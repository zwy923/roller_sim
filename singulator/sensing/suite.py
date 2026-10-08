"""The sensors of the line as one set: the cameras, the two beams, the volume scanner and the weigher, fed with
the scene.

This is where the true state becomes sensor signals: Sensors.frame() hands the lumps' true outlines and the load on
the weigh frame to the models (vision.py, beams.py, volume.py, weigher.py) every 10 ms; what the controllers get is
its return value (the cameras' objects) and the attributes beams, scanner, weigher and feed_view. Injected faults
(--fault: scan_fail:N, beam_dirty:NAME@T, beam_dead:NAME@T, vision_off:T0-T1) act here, on the sensors.
"""
import mujoco
import numpy as np

from ..machine.assembly import weigh_frame
from ..physics.lumps import IN_LINE
from .beams import Beam
from .vision import Vision
from .volume import Scanner
from .weigher import LoadCell, make_weigher


def parse_faults(cfg):
    out = []
    for f in cfg['fault']:
        kind, _, arg = f.partition(':')
        try:
            if kind == 'scan_fail':
                out.append(dict(kind=kind, n=int(arg)))
            elif kind in ('beam_dirty', 'beam_dead'):
                name, _, t = arg.partition('@')
                if name not in ('beam_feed', 'beam_stop'):
                    raise ValueError(name)
                out.append(dict(kind=kind, beam=name, t=float(t)))
            elif kind == 'vision_off':
                t0, _, t1 = arg.partition('-')
                out.append(dict(kind=kind, t0=float(t0), t1=float(t1)))
            else:
                raise ValueError(kind)
        except ValueError:
            raise ValueError('--fault %r: use scan_fail:N, beam_dirty:NAME@T, beam_dead:NAME@T or vision_off:T0-T1'
                             % f)
    return out


class Sensors:
    """Vision, the two beams, the scanner and the weigher of the line.

        vision      the cameras (sensing.vision.Vision); frame(t) returns what the controllers see of them
        feed_view   the feed belt controller's view: the same frame, or the ideal one (--feeder-sensing oracle)
        beams       {name: sensing.beams.Beam}: beam_feed (S1), beam_stop (S3)
        scanner     the volume device over the measuring belt (sensing.volume)
        weigher     the weighing device of the measuring belt (sensing.weigher), on load_cell
        margin      the position margin the controllers put on camera-judged edges
    """

    def __init__(self, cfg, d, model, data, lumps, blocks):
        self.cfg, self.d, self.model, self.data, self.lumps, self.blocks = cfg, d, model, data, lumps, blocks
        oracle = cfg['sensing'] == 'oracle'
        rng = np.random.default_rng([cfg['seed'], 20260930])      # its own stream: layout and physics unchanged
        st, dev = d['station'], d['devices']
        self.vision = Vision(dev['cameras'], rng, exits=lambda x, y: x > st['measure']['x1'] - .02, oracle=oracle)
        self.beams = {b['name']: Beam(b['name'], b['x'], b['z'], b['max_block_s'], oracle=oracle)
                      for b in dev['beams']}
        self.scanner = Scanner(dev['scanner']['heads'], dev['scanner']['zone'], rng, oracle=oracle,
                               scan_s=cfg['scan_s'], look=self._under_scanner)
        self.load_cell = LoadCell(model, weigh_frame(model), {L.geom: L.k for L in lumps})
        self.weigher = make_weigher(cfg)
        self.margin = self.vision.margin
        # the feed belt controller's own view (--feeder-sensing oracle): every lump's true centroid and outline,
        # while the station, the plough face and the alarms go on reading the cameras. It draws no noise
        self.ideal = (Vision(dev['cameras'], rng, exits=lambda x, y: False, oracle=True)
                      if not oracle and cfg['feeder_sensing'] == 'oracle' else None)
        self.feed_view = None
        self.faults = parse_faults(cfg)
        for f in self.faults:
            if f['kind'] == 'scan_fail':
                self.scanner.fail_on.add(f['n'])
            elif f['kind'] == 'vision_off':
                self.vision.outage.append((f['t0'], f['t1']))
        self.fwd = np.zeros(6)
        self._dist = {}

    def dist(self, a, b):
        """3-D gap between two lumps (capped at 5 cm), cached per frame."""
        key = (min(a, b), max(a, b))
        if key not in self._dist:
            self._dist[key] = float(mujoco.mj_geomDistance(self.model, self.data, self.lumps[a].geom,
                                                           self.lumps[b].geom, .05, self.fwd))
        return self._dist[key]

    def step(self, weighing):
        """Every physics step. weighing: the station has an item at rest on the measuring belt -- only then is the
        load cell read at every step (it costs a pass over the contacts); otherwise once a sample, in frame()."""
        if weighing:
            self.weigher.load_step(self.load_cell.force(self.data))

    def frame(self, t):
        """One 10 ms frame: faults due now, the scene, beams, vision, the weigher's sample. Returns the vision
        frame."""
        self._dist = {}
        for f in self.faults:
            if f['kind'].startswith('beam_') and not f.get('done') and t >= f['t'] - 1e-9:
                self.beams[f['beam']].fault = f['kind'][5:]          # 'dirty' reads blocked, 'dead' never
                f['done'] = round(t, 3)
        live = [L for L in self.lumps if L.state in IN_LINE]
        self.scene = {L.k: L.V for L in live}
        outlines, boxes = [L.V for L in live], [L.box for L in live]
        for b in self.beams.values():
            b.sample(t, outlines, boxes)
        com = {L.k: self.data.xipos[L.body].copy() for L in live}
        memo = {}                                   # the two views share what they work out about the scene
        fr = self.vision.frame(t, self.scene, self.dist, com, memo)
        self.feed_view = self.ideal.frame(t, self.scene, self.dist, com, memo) if self.ideal else fr
        self.weigher.sample(self.load_cell.force(self.data))
        return fr

    def _under_scanner(self):
        """What lies in the scanner's zone now (the true scene of the last frame), as Scanner.scan takes it."""
        (x0, y0, _), (x1, y1, _) = self.d['devices']['scanner']['zone']
        seen = {k: V for k, V in self.scene.items()
                if V[:, 0].max() > x0 and V[:, 0].min() < x1 and V[:, 1].max() > y0 and V[:, 1].min() < y1
                and V[:, 2].max() > self.d['station']['top_z']}
        st = self.d['station']
        belt = ((st['measure']['x0'], st['y_c'] - st['width_m'] / 2 - .005),
                (st['measure']['x1'], st['y_c'] + st['width_m'] / 2 + .005))
        return seen, (lambda k: self.blocks[k]['volume_m3']), self.dist, belt

    def report(self):
        return dict(sensing=self.cfg['sensing'],
                    feeder_sensing='oracle' if self.ideal or self.cfg['sensing'] == 'oracle' else 'vision',
                    vision=self.vision.report(),
                    beams={n: b.report() for n, b in self.beams.items()},
                    scanner=dict(scans=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()
                                         if k != '_truth'} | dict(truth_lumps=list(s.get('_truth', ())))
                                        for s in self.scanner.scans],
                                 fail_on=sorted(self.scanner.fail_on)),
                    faults=self.faults, scenario=self.cfg['scenario'])
