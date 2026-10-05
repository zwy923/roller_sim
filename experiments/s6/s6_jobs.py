"""Job lists for s6_run.py (EXPERIMENTS.md S6, 2026-10-04).

    python experiments/s6/s6_jobs.py --add base,before,oracle:7001-7030 --add centroid,ghost:7001-7015 --out jobs.json

A job is one batch: one configuration (CONFIGS: the line with one thing changed), one layout, one seed; 3 + seed % 3
lumps and a 200 s limit, as in S4. Every configuration gets the same seeds and layouts (paired comparison). The jobs
are ordered by seed, so that at any time every configuration has been through about the same batches.

Two rounds. The first (ROUND1) ran on the code as it was at 18:29 on 2026-10-04: its batches found the tracker losing
a blob's lineage, so the second round ran on the code with that fixed and the two station rules as options.
--root names a snapshot of the code next to the job list (how the two rounds were run side by side).

A first-round configuration can only be run on a snapshot (--root ../snapshots/candidate2 or candidate3, which still
have the switches): it needs the tracker as it was before the lineage and ghost fixes (SET: SPLIT_PAD / GHOST_S =
None) and, for 'before' and 'centroid', --blob-lead centroid. Those switches were comparisons with fixed bugs and
were removed from the code on 2026-10-05; the second-round configurations run on today's code. SET names the
constants by their module path of 2026-10-04; s6_run.py finds them where they live now.

The line's defaults changed on 2026-10-05 (the baseline of S7: 0.60 m lane, the feed belt reads true centroids, the
measuring device takes a fixed second). Every S6 batch ran on the line of the day before, so each job names that
line (S6_LINE) ahead of its own change. --old-code leaves S6_LINE out: a snapshot of 2026-10-04 has no such flags.
"""
import argparse
import json

LAYOUTS = ('scatter', 'aligned', 'touching', 'oblique')
# the line S6 ran on: what the defaults were on 2026-10-04
S6_LINE = ['--lane-w', '.65', '--feeder-sensing', 'vision', '--weigh-model', 'steady']
CONFIGS = {
    'base': [],                                             # round 1: the line with the blob-lead and ghost fixes
    'before': ['--blob-lead', 'centroid'],                  # the line before the fixes of 2026-10-04 (see SET)
    'centroid': ['--blob-lead', 'centroid'],                # only the blob lead as before: led by its centroid
    'ghost': [],                                            # only the tracker as before: ghosts add to the count
    'oracle': ['--sensing', 'oracle'],                      # ideal sensing: what the mechanics alone do
    'lane_rule': ['--feeder-release', 'lane'],              # next release once the lump before is in the lane
    'lane58': ['--lane-w', '.58'],                          # single-file outlet 0.58 m
    'lane62': ['--lane-w', '.62'],                          # ... 0.62 m
    'no_side_belt': ['--no-side-belt'],                     # static low-side skirt instead of the side belt
    'plate_face': ['--face-kind', 'plate'],                 # plough face: a plate instead of free rollers
    'buffer_slow': ['--buffer-speed', '.40'],               # buffer belt at the main belt's speed: no speed-up
    # the head as built: D250 drum over a D200 tail pulley, the smallest hand-over they allow, the feed belt moved
    # back by it (as designs/transfer_trial/sweep.py drum250_10)
    'drum_head': ['--feeder-head-d', '.25', '--feeder-tail-d', '.20', '--feeder-handover', '.23',
                  '--feeder-gap', '1.03'],
    # round 2: the code as it is now
    'final': [],                                            # the line as it is
    'centre': ['--weigh-stop', 'centre'],                   # the measuring belt stops an item in the middle
    'approach': ['--buffer-approach', 'slow'],              # the buffer belt's last 0.45 m at the measuring belt speed
    'weigh5': [],                                           # 5 s for a steady reading instead of 3 s (see SET)
    's3_back': [],                                          # the staging beam S3 0.15 m further back (see SET)
    'robust': ['--weigh-stop', 'centre', '--buffer-approach', 'slow'],     # the two rules and the 5 s together
    # ... and the buffer belt faster: it parts lumps that cross the head edge closer together
    'robust_b12': ['--weigh-stop', 'centre', '--buffer-approach', 'slow', '--buffer-speed', '1.2'],
    'robust_b16': ['--weigh-stop', 'centre', '--buffer-approach', 'slow', '--buffer-speed', '1.6'],
    'rolling': ['--contact-condim', '6', '--rolling-friction', '.01'],     # rolling resistance (none by default)
    # the faster buffer belt with the slow stretch before the joint lengthened for it (see SET)
    'robust_b12a': ['--weigh-stop', 'centre', '--buffer-approach', 'slow', '--buffer-speed', '1.2'],
    # what the pairs showed, together: a plate for the plough face and the 0.58 m outlet (both stagger a pair
    # more) and the station rules -- with the side belt (proto_a) and without it (proto_b)
    'proto_a': ['--weigh-stop', 'centre', '--buffer-approach', 'slow', '--face-kind', 'plate', '--lane-w', '.58'],
    'proto_b': ['--weigh-stop', 'centre', '--buffer-approach', 'slow', '--face-kind', 'plate', '--lane-w', '.58',
                '--no-side-belt'],
    # proto_a's plate holds a lump back by friction: lump on steel at the ends of the assumed range (0.45 by default;
    # the flag also sets the skirts and the lane wall)
    'proto_a_mu30': ['--weigh-stop', 'centre', '--buffer-approach', 'slow', '--face-kind', 'plate', '--lane-w', '.58',
                     '--friction-steel', '.30'],
    'proto_a_mu60': ['--weigh-stop', 'centre', '--buffer-approach', 'slow', '--face-kind', 'plate', '--lane-w', '.58',
                     '--friction-steel', '.60'],
    'mu_belt_lo': ['--friction-belt', '.45'],               # lump on belt friction, low end of the assumed range
    'mu_belt_hi': ['--friction-belt', '.70'],               # ... high end
    'dt_coarse': ['--dt', '.0005'],                         # step size check
    'dt_fine': ['--dt', '.000125'],
}
ROUND1 = ('base', 'before', 'centroid', 'ghost', 'oracle', 'lane_rule', 'lane58', 'lane62', 'no_side_belt',
          'plate_face', 'buffer_slow', 'drum_head')
# module constants changed for a configuration (s6_run.py applies them to the job's run)
NO_LINEAGE = {'singulator.perception.SPLIT_PAD': None}      # the pieces of a parted blob are new objects (round 1)
GHOSTS_COUNT = {'singulator.perception.GHOST_S': None}      # ghosts add to a merge's count (before 2026-10-04)
WEIGH_5S = {'singulator.station.WEIGH_MAX_S': 5.}
SET = {c: dict(NO_LINEAGE) for c in ROUND1}
SET['before'].update(GHOSTS_COUNT)
SET['ghost'].update(GHOSTS_COUNT)
SET.update(weigh5=WEIGH_5S, robust=WEIGH_5S, robust_b12=WEIGH_5S, robust_b16=WEIGH_5S, proto_a=WEIGH_5S,
           proto_b=WEIGH_5S, proto_a_mu30=WEIGH_5S, proto_a_mu60=WEIGH_5S,
           s3_back={'singulator.station.STOP_BACK': .37},
           robust_b12a=dict(WEIGH_5S, **{'singulator.station.APPROACH': .65}))


def jobs(groups, layouts=LAYOUTS, root=None, line=S6_LINE):
    """groups: [(config names, seeds)]. Ordered by seed, then layout, then configuration."""
    old = sorted({c for configs, _ in groups for c in configs if c in ROUND1})
    if old and not root:
        raise ValueError('%s: first-round configurations run only on a snapshot of 2026-10-04 (--root); see the '
                         'module docstring' % ', '.join(old))
    out = []
    for seed in sorted({s for _, seeds in groups for s in seeds}):
        for lay in layouts:
            for configs, seeds in groups:
                for c in configs if seed in seeds else ():
                    argv = ['--layout', lay, '--seed', str(seed), '--count', str(3 + seed % 3), '--duration', '200',
                            '--no-video'] + list(line) + CONFIGS[c]
                    out.append(dict(config=c, tag='%s_%d' % (lay, seed), argv=argv,
                                    **({'set': SET[c]} if c in SET else {}), **({'root': root} if root else {})))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--add', action='append', required=True, metavar='CONFIGS:FIRST-LAST',
                   help='comma-separated names from CONFIGS and their seeds; repeatable')
    p.add_argument('--layouts', default=','.join(LAYOUTS))
    p.add_argument('--root', help='snapshot of the code these jobs run, relative to the job list (default: project)')
    p.add_argument('--old-code', action='store_true',
                   help='--root is a snapshot of 2026-10-04: it has no flags for S6_LINE, leave them out')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    groups = []
    for g in a.add:
        names, _, seeds = g.partition(':')
        lo, hi = (int(x) for x in seeds.split('-'))
        groups.append((names.split(','), range(lo, hi + 1)))
    js = jobs(groups, a.layouts.split(','), a.root, [] if a.old_code else S6_LINE)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(js, fh, indent=0)
    print('%d jobs -> %s' % (len(js), a.out))
