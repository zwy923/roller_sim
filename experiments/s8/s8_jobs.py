"""Job list for S8 (2026-10-06): the feed head lets more than one lump go -- the three changes of docs/TODO.md
section 1 against the baseline, then the chosen one on seeds never run before.

    python experiments/s8/s8_jobs.py --seeds 7101-7130 --configs baseline,creep15,stop_past,beam_high --out jobs.json
    python experiments/s7/s7_run.py --jobs jobs.json --out runs/s8 --workers 4 [--traj]
    python experiments/s8/s8_compare.py runs/s8 --configs baseline,creep15,stop_past,beam_high
    python experiments/s7/s7_feed.py runs/s8 --config stop_past

Same batches as S7 (s7_jobs.jobs): one layout of four, one seed, 3 + seed % 3 lumps, 200 s. A configuration is
plough.py arguments plus tunable constants ("set", singulator/tuning.py):
  baseline   the S7 line (the default until 2026-10-06): the release creeps on until the lump tips into the beam;
  creep15    改法一: creep at half speed, 0.015 m/s, and the beam-miss fallback given twice as long (2 s): the
             fallback times out at the same belt travel. (S7's creep15 kept 1 s and stopped half again as often);
  stop_past  改法二 (--feeder-stop centroid): stop the release once its first centroid is control.feeder.STOP_PAST (3 mm)
             past the head edge and let the lump tip on its own; a lump that hangs is jogged JOG_STEP (5 mm) at a time,
             feeder_stall_s apart;
  beam_high  改法三: the drop beam higher, 1 cm under the feed belt top (BEAM_Z 0.9 of the step; BEAM_X stays 0.10 m):
             cut after a smaller tilt. The replay of the baseline runs (s8_beam.py) chose it: nearer the head edge was
             no better. A higher beam is cut more often before any centroid is over the edge (a lump sagging over
             it), so it comes with EARLY_BEAM_CREEP (creep on until a centroid is over, instead of stopping and
             restarting every sample);
  proposed   stop_past, with EARLY_BEAM_CREEP as well: what the acceptance on new seeds ran against the baseline.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 's7'))
from s7_jobs import LAYOUTS  # noqa: E402

CONFIGS = {   # explicit about the feed head, so that a list made today runs the same whatever the defaults become
    'baseline': (['--feeder-stop', 'beam'], {'control.feeder.EARLY_BEAM_CREEP': 0}),       # the S7 line
    'creep15': (['--feeder-stop', 'beam', '--feeder-creep-speed', '.015'],
                {'control.feeder.EARLY_BEAM_CREEP': 0, 'control.feeder.BEAM_MISS_S': 2.}),
    'stop_past': (['--feeder-stop', 'centroid'], {'control.feeder.EARLY_BEAM_CREEP': 0}),
    'beam_high': (['--feeder-stop', 'beam'], {'machine.feed_belt.BEAM_Z': .9, 'control.feeder.EARLY_BEAM_CREEP': 1}),
    'proposed': (['--feeder-stop', 'centroid'], {'control.feeder.EARLY_BEAM_CREEP': 1}),
}


def jobs(configs, seeds, layouts=LAYOUTS):
    """Ordered by seed, then layout, then configuration."""
    out = []
    for seed in seeds:
        for lay in layouts:
            for c in configs:
                argv, settings = CONFIGS[c]
                out.append(dict(config=c, tag='%s_%d' % (lay, seed),
                                argv=['--layout', lay, '--seed', str(seed), '--count', str(3 + seed % 3),
                                      '--duration', '200', '--no-video'] + argv,
                                **({'set': settings} if settings else {})))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', required=True, metavar='FIRST-LAST')
    p.add_argument('--configs', default='baseline', help='comma-separated names from CONFIGS (default: baseline)')
    p.add_argument('--layouts', default=','.join(LAYOUTS))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    lo, hi = (int(x) for x in a.seeds.split('-'))
    js = jobs(a.configs.split(','), range(lo, hi + 1), a.layouts.split(','))
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(js, fh, indent=0)
    print('%d jobs -> %s' % (len(js), a.out))
