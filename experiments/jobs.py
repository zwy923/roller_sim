"""Job list: configurations x four layouts x seeds, for run.py.

    python experiments/jobs.py --seeds 7201-7260 --configs beam_stop,baseline --out runs/jobs.json
    python experiments/run.py --jobs runs/jobs.json --out runs/batch --workers 4 [--traj]
    python experiments/compare.py runs/batch --configs beam_stop,baseline --list
    python experiments/analyze.py runs/batch --list
    python experiments/feed.py runs/batch --config baseline

A job is one batch: one configuration, one layout (scatter, aligned, touching, oblique), one seed, 3 + seed % 3
lumps, a 200 s limit. A configuration is plough.py arguments plus tunable constants ("set", singulator/tuning.py),
explicit about the feed head, so that a list made today runs the same whatever the defaults become:
  baseline   the line as it is by default (S8, 2026-10-06): the release stops once its first centroid is
             control.feeder.STOP_PAST (3 mm) past the head edge and the lump tips on its own; a lump that hangs is
             jogged JOG_STEP (5 mm) at a time, feeder_stall_s apart; a drop beam cut before any centroid is over the
             edge does not stop the release (EARLY_BEAM_CREEP);
  ideal      baseline with every controller reading the true state: a lump the baseline loses and 'ideal' does not
             was lost to what the station's sensors made of it, not to the mechanics;
  beam_stop  the S7 line (the default until 2026-10-06): the release creeps on until the lump tips into the beam.
The three changes of docs/TODO.md section 1 that S8 tried, each on the S7 line:
  creep15    改法一: creep at half speed, 0.015 m/s, and the beam-miss fallback given twice as long (2 s): the
             fallback times out at the same belt travel;
  stop_past  改法二: the centroid stop of baseline, without EARLY_BEAM_CREEP;
  beam_high  改法三: the drop beam higher, 1 cm under the feed belt top (BEAM_Z 0.9 of the step; BEAM_X stays 0.10 m):
             cut after a smaller tilt, with EARLY_BEAM_CREEP (a higher beam is cut more often before any centroid is
             over the edge).
"""
import argparse
import json

LAYOUTS = ('scatter', 'aligned', 'touching', 'oblique')
BASELINE = (['--feeder-stop', 'centroid'], {'control.feeder.EARLY_BEAM_CREEP': 1})
CONFIGS = {
    'baseline': BASELINE,
    'ideal': (BASELINE[0] + ['--sensing', 'oracle'], BASELINE[1]),
    'beam_stop': (['--feeder-stop', 'beam'], {'control.feeder.EARLY_BEAM_CREEP': 0}),
    'creep15': (['--feeder-stop', 'beam', '--feeder-creep-speed', '.015'],
                {'control.feeder.EARLY_BEAM_CREEP': 0, 'control.feeder.BEAM_MISS_S': 2.}),
    'stop_past': (['--feeder-stop', 'centroid'], {'control.feeder.EARLY_BEAM_CREEP': 0}),
    'beam_high': (['--feeder-stop', 'beam'], {'machine.feed_belt.BEAM_Z': .9, 'control.feeder.EARLY_BEAM_CREEP': 1}),
}


def jobs(configs, seeds, layouts=LAYOUTS, root=None):
    """Ordered by seed, then layout, then configuration."""
    out = []
    for seed in seeds:
        for lay in layouts:
            for c in configs:
                argv, settings = CONFIGS[c]
                out.append(dict(config=c, tag='%s_%d' % (lay, seed),
                                argv=['--layout', lay, '--seed', str(seed), '--count', str(3 + seed % 3),
                                      '--duration', '200', '--no-video'] + argv,
                                **({'set': settings} if settings else {}), **({'root': root} if root else {})))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', required=True, metavar='FIRST-LAST')
    p.add_argument('--configs', default='baseline', help='comma-separated names from CONFIGS (default: baseline)')
    p.add_argument('--layouts', default=','.join(LAYOUTS))
    p.add_argument('--root', help='another checkout of the code for these jobs, relative to the job list '
                                  '(default: this project)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    lo, hi = (int(x) for x in a.seeds.split('-'))
    js = jobs(a.configs.split(','), range(lo, hi + 1), a.layouts.split(','), a.root)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(js, fh, indent=0)
    print('%d jobs -> %s' % (len(js), a.out))
