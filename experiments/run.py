"""Batch runner: a list of jobs through the model, one process per batch, resumable.

    python experiments/run.py --jobs JOBS.json --out OUT_DIR [--workers N] [--root PROJECT_ROOT] [--match TEXT]
                              [--traj]

JOBS.json: [{"config": name, "tag": layout_seed, "argv": [plough.py arguments], "set": {"MODULE.NAME": value},
"root": folder}, ...] (jobs.py writes it).
  "set"   changes tunable constants for that job: the same as --set MODULE.NAME=VALUE in argv
          (singulator/tuning.py), e.g. {"control.station.APPROACH": 0.65};
  "root"  relative to JOBS.json, is the project folder that job runs -- another checkout of the code -- in place
          of PROJECT_ROOT.
Each job is one batch through singulator.simulate.run in its own process. Per job it keeps, under OUT_DIR/<config>/:
  <tag>.json            a compact summary (what analyze.py reads);
  <tag>.result.json.gz  the full result.json of the run;
  <tag>/                with --traj only: result.json, model.xml and trajectory.npz as a single run writes them
                        (about 0.5 MB a batch; experiments/replay.py replays them).
Finished jobs are skipped, so the same command resumes an interrupted sweep. A file named STOP in OUT_DIR ends the
sweep after the jobs already running. Workers opt out of Windows power throttling (see _full_speed) and use
single-threaded BLAS; a worker takes about 150 MB.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path


def _full_speed():
    """Windows on a hybrid CPU keeps a windowless (background) process on the efficiency cores unless it opts out
    of power throttling: 14 workers shared 4 cores' worth (each at 35-50 % of a core) until they did."""
    try:
        if os.name == 'nt':
            import ctypes

            class State(ctypes.Structure):
                _fields_ = [('Version', ctypes.c_ulong), ('ControlMask', ctypes.c_ulong), ('StateMask', ctypes.c_ulong)]
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p
            k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
            st = State(1, 0x1, 0)                 # execution-speed throttling: controlled, and off
            k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))   # 4: throttling
    except Exception:
        pass


def _peak_mb():
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes as w

            class PMC(ctypes.Structure):
                _fields_ = [('cb', w.DWORD), ('PageFaultCount', w.DWORD)] + [
                    (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                                   'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                                   'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
            p = PMC()
            p.cb = ctypes.sizeof(p)
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), w.DWORD]
            ctypes.windll.psapi.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
            return round(p.PeakWorkingSetSize / 2 ** 20)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        return None


def summary(job, r, wall):
    """The compact record of one batch (everything the analysis needs without opening the full result)."""
    f, tr, st, o, pc = r['feeder'], r['transfer'], r['station'], r['outcome'], r['perception']
    tips = {L['lump']: L['t_tip'] for L in tr['lumps']}
    items = []
    for it in st['items']:
        items.append(dict(item=it['item'], stage=it.get('stage'), void=it.get('void'), reasons=it.get('reasons'),
                          reason_info=it.get('reason_info'), truth_lumps=it.get('truth_lumps'),
                          truth_single=it.get('truth_single'), truth_isolated=it.get('truth_isolated'),
                          route=it.get('route'), route_as_ideal=it.get('route_as_ideal'),
                          mass_error_pct=it.get('mass_error_pct'), weigh_wait_s=it.get('weigh_wait_s'),
                          n_obj_max=it.get('n_obj_max'), n_est_max=it.get('n_est_max'), split_from=it.get('split_from'),
                          joined=it.get('joined'), t_in_s=it.get('t_in_s'),
                          t_stop_s=it.get('t_stop_s'), t_hold_s=it.get('t_hold_s'), bins=it.get('bins')))
    return dict(config=job['config'], tag=job['tag'], argv=job['argv'], set=job.get('set'),
                layout=r['config']['layout'],
                seed=r['config']['seed'], count=r['config']['count'], sensing=r['config']['sensing'],
                classification=o['classification'], end_s=o['end_time_s'], line_clear_s=o['line_clear_s'],
                left_on_belt=o['left_on_belt'], dropped=o['dropped'], sorted=o['sorted'],
                jam_stop=o['jam_stop'], unjam=o['unjam_pulses'], touched_plough=o['touched_plough'],
                together_at_cut_s=o['together_at_cut_s'], funnel=o['funnel'],
                numerics_ok=r['numerics']['ok'], pen_mm=round(1000 * r['numerics']['max_penetration_m'], 2),
                pen_landed_mm=round(1000 * ((r['numerics'].get('landed') or {}).get('max_penetration_m') or 0.), 2),
                holds=st['counts']['holds'], void=st['counts']['void'], valid=st['counts']['valid'],
                void_reasons=st['counts']['void_reasons'], n_items=st['counts']['items'], landed=st['counts']['landed'],
                route_not_as_ideal=st['counts']['route_not_as_ideal'], void_discharged=st['counts']['void_discharged'],
                station_counts={k: st['counts'][k] for k in ('section_holds', 'late', 'jogs', 'buffer_stops', 'splits',
                                                             'freezes', 's3_by_camera', 'plate_moves')},
                taken_off=sorted({a['lump'] for a in r['taken_off']}),
                false_valid=st['verification']['false_valid'], false_void=st['verification']['false_void'],
                landed_unmeasured=st['verification']['landed_unmeasured'],
                fault=(st['fault'] or {}).get('reason'), mass_error_max_pct=st['mass_error_pct_abs_max'],
                upstream_held_s=st['upstream_held_s'], buffer_stopped_s=st['buffer_stopped_s'], frozen_s=st['frozen_s'],
                items=items,
                feeder_counts=f['counts'], never_released=f['never_released'], final_phase=f['final_phase'],
                releases=[dict(lumps=x['lumps'], after_stop=x['after_stop'], in_staging=x.get('in_staging', []),
                               stop=x['stop'], jogs=x['jogs'], tip_spread_s=x['tip_spread_s']) for x in tr['releases']],
                tips=tips, transfer=tr['summary'],
                vision=pc['vision']['counts'], centroid_error=pc['vision']['centroid_error_m'],
                beam_alarms={k: (b['alarm'] or {}).get('why') for k, b in pc['beams'].items() if b['alarm']},
                masses=[round(b['mass_kg'], 2) for b in r['blocks']], materials=[b['material'] for b in r['blocks']],
                families=[b.get('family') for b in r['blocks']],
                source_sha=(r.get('provenance') or {}).get('source_sha256'), mujoco=r.get('mujoco'),
                wall_s=round(wall, 1), peak_mb=_peak_mb(), platform=sys.platform)


def _set_flags(settings):
    """A job's "set" as --set flags."""
    out = []
    for name, value in (settings or {}).items():
        text = 'none' if value is None else ','.join(repr(v) for v in value) if isinstance(value, (list, tuple)) \
            else repr(value)
        out += ['--set', '%s=%s' % (name, text)]
    return out


def work(arg):
    job, root, out, traj = arg
    _full_speed()
    for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')     # the model's arrays are small: BLAS thread pools only cost memory
    dest = Path(out) / job['config'] / (job['tag'] + '.json')
    if dest.exists():
        return job['config'], job['tag'], 'skipped', None
    if (Path(out) / 'STOP').exists():
        return job['config'], job['tag'], 'stopped', None
    sys.path.insert(0, root)
    t0 = time.time()
    try:
        from singulator import simulate
        from singulator.config import parse_config
        full = {}

        def keep(cfg, result, xml, tr, drive_names):         # no files from the run itself, unless --traj
            full['r'] = result
            if traj:
                from singulator.sim import results
                results.write(cfg, result, xml, tr, drive_names)
        cfg = parse_config(list(job['argv']) + _set_flags(job.get('set')))
        cfg['out_dir'] = Path(out) / job['config'] / job['tag']
        r = simulate.run(cfg, save=keep)
        s = summary(job, r, time.time() - t0)
        if 'side_pusher' in r:
            s['side_pusher'] = r['side_pusher']
            s['entry_audit'] = r['entry_audit']
            s['measurement_audit'] = r['measurement_audit']
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest.with_name(job['tag'] + '.result.json.gz'), 'wt', encoding='utf-8') as fh:
            json.dump(full['r'], fh, ensure_ascii=False)
        line = '%s holds %s clear %s' % (s['classification'], s['holds'], s['line_clear_s'])
    except Exception as e:                                     # a failed batch is a result too
        s = dict(config=job['config'], tag=job['tag'], argv=job['argv'], error=repr(e), tb=traceback.format_exc(),
                 wall_s=round(time.time() - t0, 1))
        dest.parent.mkdir(parents=True, exist_ok=True)
        line = 'ERROR %r' % e
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, dest)
    return job['config'], job['tag'], line, s.get('wall_s')


def main():
    here = Path(__file__).resolve()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--root', default=str(here.parents[1]), help='project folder that holds the singulator package')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 4))
    p.add_argument('--match', default='', help='only jobs whose config/tag contains this text')
    p.add_argument('--traj', action='store_true', help='also keep each run\'s model.xml and trajectory.npz')
    a = p.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(a.jobs).read_text(encoding='utf-8'))
    jobs = [j for j in jobs if a.match in '%s/%s' % (j['config'], j['tag'])]
    todo = [j for j in jobs if not (out / j['config'] / (j['tag'] + '.json')).exists()]
    log = open(out / 'progress.log', 'a', encoding='utf-8')

    def say(msg):
        msg = '%s %s' % (time.strftime('%H:%M:%S'), msg)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()
    say('start: %d jobs, %d to run, %d workers, root %s' % (len(jobs), len(todo), a.workers, a.root))
    t0, n = time.time(), 0
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        at = Path(a.jobs).resolve().parent
        root = lambda j: (at / j['root']).resolve() if j.get('root') else Path(a.root).resolve()
        args = [(j, str(root(j)), str(out), a.traj) for j in todo]
        for config, tag, line, wall in pool.imap_unordered(work, args):
            n += 1
            say('[%d/%d] %s/%s %s (%s s)' % (n, len(todo), config, tag, line, wall))
    say('done: %d jobs in %.0f s' % (n, time.time() - t0))


if __name__ == '__main__':
    main()
