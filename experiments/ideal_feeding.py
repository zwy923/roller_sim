"""The upper bound of any feeding algorithm on the side-gate hardware.

    python experiments/ideal_feeding.py jobs --out runs/side_gate_ideal_300
    python experiments/run.py --jobs runs/side_gate_ideal_300/jobs.json --out runs/side_gate_ideal_300/data --workers 14 --traj
    python experiments/ideal_feeding.py analyze --root runs/side_gate_ideal_300 [--camera-runs runs/side_gate_300]

300 batches (five layouts x 3/4/5 lumps x 20 seeds from 16000, 200 s each), the gate installed in every arm:
  parked         the feeder reads the cameras and stops on the drop beam, the gate parked: the camera line;
  oracle_parked  the feeder reads the true lumps and stops on the true centroid (S8), the gate parked: what perfect
                 perception alone gives;
  ideal          the same, and the gate holds side-by-side neighbours (--side-pusher active): perfect perception and
                 a perfect hold-back decision together.
--camera-runs reads the arms missing from --root out of an earlier camera run: 'parked', and 'active' -- the
camera-planned gate, removed 2026-10-10. The analysis compares the arms batch by batch (paired, stratified bootstrap
by layout x count); every planned batch counts, a missing or failed one stops it.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ('scatter', 'aligned', 'touching', 'flat', 'oblique')
ARMS = ('parked', 'active', 'oracle_parked', 'ideal')
PAIRS = (('ideal', 'parked'), ('ideal', 'active'), ('oracle_parked', 'parked'), ('ideal', 'oracle_parked'))


def jobs(out, seed_base=16000):
    sys.path.insert(0, str(ROOT))
    from singulator.sim.results import provenance
    rows, serial = [], 0
    for layout in LAYOUTS:
        for count in (3, 4, 5):
            for _ in range(20):
                seed = seed_base + serial
                serial += 1
                tag = '%s_n%d_%d' % (layout, count, seed)
                base = ['--seed', str(seed), '--count', str(count), '--layout', layout, '--duration', '200',
                        '--no-video']
                true = base + ['--feeder-sensing', 'oracle', '--feeder-stop', 'centroid']
                rows += [dict(config='parked', tag=tag, argv=base + ['--feeder-sensing', 'vision', '--feeder-stop',
                                                                     'beam', '--side-pusher', 'parked']),
                         dict(config='oracle_parked', tag=tag, argv=true + ['--side-pusher', 'parked']),
                         dict(config='ideal', tag=tag, argv=true + ['--side-pusher', 'active'])]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'jobs.json').write_text(json.dumps(rows, indent=0), encoding='utf-8')
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / 'manifest.json').write_text(json.dumps(dict(
        runs=len(rows), source_sha256=provenance('')['source_sha256'], git_head=git('rev-parse', 'HEAD'),
        uncommitted=git('diff', 'HEAD', '--stat')), indent=1), encoding='utf-8')
    print(len(rows), 'jobs ->', out / 'jobs.json')


def _load(roots, arm, tag):
    path = next((r / 'data' / arm / (tag + '.json') for r in roots if (r / 'data' / arm / (tag + '.json')).exists()),
                None)
    if path is None:
        raise RuntimeError('%s/%s missing; do not publish a survivor-only comparison' % (arm, tag))
    r = json.loads(path.read_text(encoding='utf-8'))
    if 'error' in r:
        raise RuntimeError('%s/%s failed: %s; do not publish a survivor-only comparison' % (arm, tag, r['error']))
    return r


def _totals(rs):
    rel = [x for r in rs for x in r['releases'] if x['lumps']]
    line_s = sum(r['line_clear_s'] if r['line_clear_s'] is not None else r['end_s'] for r in rs)
    measured = sum(r['measurement_audit']['counts']['measured'] for r in rs)
    causes = {}
    for r in rs:
        for k, v in r['measurement_audit']['counts']['failure_causes'].items():
            causes[k] = causes.get(k, 0) + v
    acts = [a for r in rs for a in r['side_pusher']['actions']]
    return dict(batches=len(rs), input=sum(r['count'] for r in rs), measured=measured, failure_causes=causes,
                releases=len(rel), multi_releases=sum(len(x['lumps']) > 1 for x in rel),
                entry_qualified=sum(r['entry_audit']['qualified'] for r in rs),
                line_s_mean=round(line_s / len(rs), 2), measured_per_min=round(60 * measured / line_s, 3),
                faults=sum(bool(r['jam_stop']) for r in rs), numerics_bad=sum(not r['numerics_ok'] for r in rs),
                taken_off=sum(len(r['taken_off']) for r in rs), dropped=sum(r['dropped'] for r in rs),
                gate_actions=len(acts), gate_batches=sum(bool(r['side_pusher']['actions']) for r in rs),
                gate_ends={e: sum(a.get('push_end', 'unfinished') == e for a in acts)
                           for e in sorted({a.get('push_end', 'unfinished') for a in acts})})


def analyze(root, camera_runs=None):
    roots = [root] + ([camera_runs] if camera_runs else [])
    tags = [j['tag'] for j in json.loads((root / 'jobs.json').read_text(encoding='utf-8')) if j['config'] == 'ideal']
    arms = [a for a in ARMS if a != 'active' or camera_runs]
    data = {tag: {arm: _load(roots, arm, tag) for arm in arms} for tag in tags}
    for tag, rs in data.items():
        assert len({(r['layout'], r['count'], r['seed']) for r in rs.values()}) == 1, tag
    batches = [data[t] for t in tags]
    out = dict(totals={a: _totals([b[a] for b in batches]) for a in arms},
               by_layout={l: {a: _totals([b[a] for b in batches if b[a]['layout'] == l]) for a in arms}
                          for l in LAYOUTS})
    m = np.array([[b[a]['measurement_audit']['counts']['measured'] for a in arms] for b in batches])
    n = np.array([b['parked']['count'] for b in batches])
    strata = [[i for i, b in enumerate(batches) if b['parked']['layout'] == l and b['parked']['count'] == c]
              for l in LAYOUTS for c in (3, 4, 5)]
    strata = [s for s in strata if s]
    rng = np.random.default_rng(20261010)
    idx = [np.concatenate([rng.choice(s, len(s), replace=True) for s in strata]) for _ in range(5000)]
    out['paired'] = {}
    for x, y in PAIRS:
        if x not in arms or y not in arms:
            continue
        d = m[:, arms.index(x)] - m[:, arms.index(y)]
        boot = [d[k].sum() / n[k].sum() for k in idx]
        out['paired']['%s-%s' % (x, y)] = dict(
            measured_rate_pp=round(100 * d.sum() / n.sum(), 2),
            ci95_pp=[round(100 * float(q), 2) for q in np.quantile(boot, [.025, .975])],
            better_batches=int((d > 0).sum()), worse_batches=int((d < 0).sum()),
            worse=[t for t, v in zip(tags, d) if v < 0])
    out['lost'] = {a: {t: [(L['lump'], L['fate']) for L in data[t][a]['measurement_audit']['lumps']
                           if L['fate'] != 'measured'] for t in tags
                       if data[t][a]['measurement_audit']['counts']['not_measured']} for a in arms}
    out['source_sha256'] = {a: sorted({b[a]['source_sha'] for b in batches}) for a in arms}
    for a, shas in out['source_sha256'].items():
        assert len(shas) == 1, 'mixed package revisions in %s: %s' % (a, shas)
    (root / 'summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('%-14s %9s %9s %7s %7s %7s %8s %7s' % ('arm', 'measured', 'multi', 'entry', 'line_s', 'per_min', 'actions',
                                                  'faults'))
    for a in arms:
        t = out['totals'][a]
        print('%-14s %4d/%-4d %4d/%-4d %7d %7.2f %7.3f %8d %7d  %s' % (
            a, t['measured'], t['input'], t['multi_releases'], t['releases'], t['entry_qualified'], t['line_s_mean'],
            t['measured_per_min'], t['gate_actions'], t['faults'] + t['numerics_bad'], t['failure_causes']))
    for k, v in out['paired'].items():
        print('%-24s %+6.2f pp  95%% CI [%+.2f, %+.2f]  better %d  worse %d %s' % (
            k, v['measured_rate_pp'], *v['ci95_pp'], v['better_batches'], v['worse_batches'], v['worse']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    j = sub.add_parser('jobs')
    j.add_argument('--out', type=Path, required=True)
    j.add_argument('--seed-base', type=int, default=16000)
    a = sub.add_parser('analyze')
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--camera-runs', type=Path)
    args = p.parse_args()
    jobs(args.out, args.seed_base) if args.command == 'jobs' else analyze(args.root, args.camera_runs)
