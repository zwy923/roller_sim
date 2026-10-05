"""Job list for the baseline line (EXPERIMENTS.md S7, 2026-10-05): one fixed line and the batches it is judged on.

    python experiments/s7/s7_jobs.py --seeds 7001-7010 [--configs baseline,ideal] [--root DIR] --out jobs.json
    python experiments/s6/s6_run.py --jobs jobs.json --out experiments/s7/out --workers 10
    python experiments/s7/s7_analyze.py experiments/s7/out --list

A job is one batch, as in S6: one configuration, one layout (scatter, aligned, touching, oblique), one seed,
3 + seed % 3 lumps, a 200 s limit. s6_run.py runs the list (resumable); s7_analyze.py counts the lumps measured alone.

The baseline is the line with its default parameters (README): the feed belt controller reads the lumps' true
centroids, the measuring device takes a fixed scan_s, the station reads its cameras and beams. 'ideal' is the same
line with every controller reading the true state: a lump the baseline loses and 'ideal' does not was lost to what
the station's sensors made of it, not to the mechanics.
"""
import argparse
import json

LAYOUTS = ('scatter', 'aligned', 'touching', 'oblique')
CONFIGS = {
    'baseline': [],                                 # the line as it is by default
    'ideal': ['--sensing', 'oracle'],               # ... with the station and the plough face reading the true state
    # the feed belt creeps on from the moment a lump's centroid is over the head edge until that lump has tipped far
    # enough to cut the drop beam (0.55 s at the median): a second lump whose centroid lies within that travel goes
    # with it. Slower creep = less travel (0.03 m/s by default)
    'creep15': ['--feeder-creep-speed', '.015'],
    'creep075': ['--feeder-creep-speed', '.0075'],
}


def jobs(configs, seeds, layouts=LAYOUTS, root=None):
    """Ordered by seed, then layout, then configuration."""
    return [dict(config=c, tag='%s_%d' % (lay, seed),
                 argv=['--layout', lay, '--seed', str(seed), '--count', str(3 + seed % 3), '--duration', '200',
                       '--no-video'] + CONFIGS[c], **({'root': root} if root else {}))
            for seed in seeds for lay in layouts for c in configs]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', required=True, metavar='FIRST-LAST')
    p.add_argument('--configs', default='baseline', help='comma-separated names from CONFIGS (default: baseline)')
    p.add_argument('--layouts', default=','.join(LAYOUTS))
    p.add_argument('--root', help='snapshot of the code these jobs run, relative to the job list (default: project)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    lo, hi = (int(x) for x in a.seeds.split('-'))
    js = jobs(a.configs.split(','), range(lo, hi + 1), a.layouts.split(','), a.root)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(js, fh, indent=0)
    print('%d jobs -> %s' % (len(js), a.out))
