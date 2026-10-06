"""One batch through the machine: build the line, run it to its end, write the outputs.

    from singulator.config import default_config
    from singulator.simulate import run
    result = run(default_config(seed=392, video=False))

plough.py is the command line of run().

Outputs in cfg['out_dir']: result.json (config, derived geometry, outcome, drive loads, retract events,
per-lump record, 'feeder' (releases, stops), 'transfer' (the feed head record, verify/transfer.py),
'station' (every item: weighing, volume, route, bin, void reasons, the verification against the true state),
'perception' (vision, beams, scanner, injected faults), 'taken_off' (lumps of held items)), model.xml (the
model actually compiled), trajectory.npz (every 10 ms: body pose, centroid, front edge, contact categories,
face angle, lump-face force, drive speed factors, the measuring belt's load-cell reading 'weigh_N' -- per-step
mean while weighing, else the instant -- and the separator plate angle 'sep_deg'), video.mp4 unless --no-video.

What the controllers read: with --sensing vision (the default) the station, the plough face and the alarms read
only modelled sensor signals (singulator/sensing/); the feed belt controller reads the lumps' true centroids
(--feeder-sensing oracle, the default since 2026-10-05) or the cameras too (--feeder-sensing vision).
With --sensing oracle the same controllers read ideal sensors (the true state), for comparison.

What happens in a physics step and in a 10 ms sample is sim/line.py; how the result is put together, sim/results.py.
"""
import time

from . import tuning
from .config import validate
from .sim import results
from .sim.line import Line
from .sim.video import Recorder, overlay, plan_labels


def run(cfg, save=True):
    """One batch. cfg: a configuration (config.parse_config / default_config). save: True writes result.json,
    model.xml and trajectory.npz into cfg['out_dir']; False writes nothing; a callable gets
    (cfg, result, xml, trajectory, drive_names) instead. Returns the result (what result.json holds)."""
    cfg = validate(cfg)
    with tuning.applied(cfg['set']):          # --set MODULE.NAME=VALUE, for this run only
        line = Line(cfg)
        rec = Recorder(cfg, line.d, line.model) if cfg['video'] else None
        wall = time.time()
        while line.advance():
            if rec and rec.due(line.t):
                # with the feed belt the camera follows the released lumps, not the ones queued on it
                rec.frame(line.data, line.followed(), overlay(line), plan_labels(line))
        if rec:
            rec.close()
        out = results.assemble(line, time.time() - wall)
    if save:
        (results.write if save is True else save)(cfg, out, line.xml, line.trace.traj, line.belts.names())
    return out
