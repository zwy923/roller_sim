"""Feed belt head + belt below, simulated bench (singulator/trial.py): a geometry x layout matrix.

    python designs/transfer_trial/sweep.py                      # the default matrix, 6 workers
    python designs/transfer_trial/sweep.py --seeds 5 --workers 8
    python designs/transfer_trial/sweep.py --geometry sharp_5 drum250_10

Every run is `python plough.py --bench --layout CASE --seed N --count 3..5` with the transfer of one GEOMETRY;
the run ends once every lump lies on the main belt below the feed belt head. Prints one table per geometry and writes
runs/transfer_sweep_<time>/summary.json. The numbers screen geometries for the bench; they are not bench
results (rigid convex lumps, idealised belt surfaces, no belt sag or cover wear).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# name: (feeder step, head drum D, tail pulley D); the hand-over is the smallest the two allow (rounded up to 1 cm)
GEOMETRIES = {
    'sharp_5':     (.05, 0., 0.),       # the model until 2026-09-30: a sharp head edge 5 cm over a square belt start
    'sharp_10':    (.10, 0., 0.),       # the line since 2026-09-30 (user: 高度差改到 10 cm)
    'nose100_5':   (.05, .10, .10),     # a small nose drum over a small tail pulley (thin belt only)
    'drum250_5':   (.05, .25, .20),     # a wear-belt head drum, 5 cm step: the belts must stand apart
    'drum250_10':  (.10, .25, .20),
    'drum250_20':  (.20, .25, .20),
}
CASES = ('scatter', 'aligned', 'touching', 'flat', 'oblique')


def one(job):
    from singulator import feeder
    from singulator.config import parse_config
    from singulator.simulate import run
    geo, case, seed, out = job
    h, D, d = GEOMETRIES[geo]
    hand = 0. if not (D or d) else round(feeder.min_handover(h, D / 2, d / 2) + .005, 2)
    args = ['--bench', '--layout', case, '--seed', str(seed), '--count', str(3 + seed % 3),
            '--feeder-step', str(h), '--feeder-head-d', str(D), '--feeder-tail-d', str(d),
            '--feeder-handover', str(hand), '--feeder-gap', str(round(.80 + hand, 3)), '--duration', '120',
            '--no-video']
    cfg = parse_config(args)
    cfg['out_dir'] = Path(out) / ('%s_%s_%d' % (geo, case, seed))
    t0 = time.time()
    try:
        r = run(cfg)
    except Exception as e:                  # a layout that does not fit, or a numerical failure: record it
        return dict(geometry=geo, case=case, seed=seed, error=repr(e))
    tr = r['transfer']
    return dict(geometry=geo, case=case, seed=seed, handover_m=hand, count=3 + seed % 3,
                summary=tr['summary'], releases=[x['lumps'] for x in tr['releases']],
                stops=[(s['travel_m'], s.get('lump_slide_m')) for s in tr['stops']],
                numerics_ok=r['numerics']['ok'], end_s=r['outcome']['end_time_s'], wall_s=round(time.time() - t0, 1))


def table(rows):
    keys = ('lumps', 'went', 'releases', 'double_releases', 'lumps_in_doubles', 'after_stop', 'hangs', 'jogs',
            'early_contacts', 'dragged', 'not_apart')
    out = []
    for geo in dict.fromkeys(r['geometry'] for r in rows):
        g = [r for r in rows if r['geometry'] == geo and 'summary' in r]
        out.append('%s (hand-over %s m, %d batches, %d failed)' % (geo, g[0]['handover_m'] if g else '?', len(g),
                                                                 sum(r['geometry'] == geo and 'error' in r for r in rows)))
        out.append('  case       ' + ' '.join('%8s' % k[:8] for k in keys) + '  stop_max  slide_max')
        for case in CASES:
            c = [r for r in g if r['case'] == case]
            if not c:
                continue
            tot = {k: sum(r['summary'][k] or 0 for r in c) for k in keys}
            stop = max((s[0] for r in c for s in r['stops']), default=0.)
            slide = max((s[1] or 0. for r in c for s in r['stops']), default=0.)
            out.append('  %-10s ' % case + ' '.join('%8d' % tot[k] for k in keys) + '  %7.4f  %8.4f' % (stop, slide))
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--geometry', nargs='*', default=list(GEOMETRIES), choices=list(GEOMETRIES))
    p.add_argument('--case', nargs='*', default=list(CASES), choices=list(CASES))
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--seed0', type=int, default=5001)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    out = ROOT / 'runs' / ('transfer_sweep_%s' % datetime.now().strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(g, c, a.seed0 + i, str(out)) for g in a.geometry for c in a.case for i in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs))
    (out / 'summary.json').write_text(json.dumps(dict(geometries=GEOMETRIES, rows=rows), indent=1), encoding='utf-8')
    print(table(rows))
    for r in rows:
        if 'error' in r:
            print('FAILED', r['geometry'], r['case'], r['seed'], r['error'])
    print('written', out / 'summary.json')


if __name__ == '__main__':
    main()
