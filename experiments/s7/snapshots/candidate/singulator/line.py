"""Station line glue for simulate.run(): the sensors, taking held items off the line, injected faults and the
acceptance scenarios (2026-09-30).

  Sensors      perception.Vision / Beam / Scanner fed with the scene every 10 ms (the true state enters here
               only to form measurements and the verification);
  TakeOff      a held (void) item's lumps leave the line at once and lie on the floor far away (state
               'taken_off'). This stands in for the prototype's manual separation and re-measuring, which is
               not modelled (user 2026-09-30: 不需要模拟人工操作); the stop itself is counted by the station;
  faults       --fault: scan_fail:N, beam_dirty:NAME@T, beam_dead:NAME@T, vision_off:T0-T1;
  scenarios    --scenario touching / touching_lane / off_scale_neighbour / off_scale_bridge: the acceptance
               tests (故意让两块相贴、料搭在秤外；扫描失败 = --fault scan_fail:0);
  SectionWatch the singulation section as the cameras see it, for the plough face's closing permission and
               the stall rule (vision only; the oracle path keeps tracking.stalled on the true state).
"""
import math

import mujoco
import numpy as np

from . import perception
from .lumps import block_world_vertices
from .station import V_MARGIN

PARK_GAP = 1.3        # m: lumps taken off lie this far apart on the floor, upstream of the machine
IN_LINE = ('on_belt', 'passed')   # still on the line ('passed' = tail past the singulator's plane, the head edge)


def parse_faults(cfg):
    out = []
    for f in cfg.get('fault') or []:
        kind, _, arg = f.partition(':')
        try:
            if kind == 'scan_fail':
                out.append(dict(kind=kind, n=int(arg)))
            elif kind in ('beam_dirty', 'beam_dead'):
                name, _, t = arg.partition('@')
                if name not in ('beam_feed', 'beam_in', 'beam_stop', 'beam_gangue'):
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
    """Vision, the four beams and the scanner of the line."""

    def __init__(self, cfg, d, model, data, lumps, blocks):
        self.cfg, self.d, self.model, self.data, self.lumps, self.blocks = cfg, d, model, data, lumps, blocks
        oracle = cfg['sensing'] == 'oracle'
        rng = np.random.default_rng([cfg['seed'], 20260930])      # its own stream: layout and physics unchanged
        st, dev = d['station'], d['devices']
        self.vision = perception.Vision(dev['cameras'], rng, exits=lambda x, y: x > st['measure']['x1'] - .02,
                                        oracle=oracle)
        self.beams = {b['name']: perception.Beam(b['name'], b['x'], b['z'], b['max_block_s'], oracle=oracle)
                      for b in dev['beams']}
        self.scanner = perception.Scanner(dev['scanner']['heads'], dev['scanner']['zone'], rng, oracle=oracle)
        # the feed belt controller's own view (--feeder-sensing oracle): every lump's true centroid and outline,
        # while the station, the plough face and the alarms go on reading the cameras. It draws no noise
        self.ideal = (perception.Vision(dev['cameras'], rng, exits=lambda x, y: False, oracle=True)
                      if not oracle and cfg.get('feeder_sensing', 'vision') == 'oracle' else None)
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

    def frame(self, t):
        """One 10 ms frame: faults due now, the scene, beams, vision. Returns the vision frame."""
        self._dist = {}
        for f in self.faults:
            if f['kind'].startswith('beam_') and not f.get('done') and t >= f['t'] - 1e-9:
                self.beams[f['beam']].fault = f['kind'][5:]          # 'dirty' reads blocked, 'dead' never
                f['done'] = round(t, 3)
        self.scene = {L.k: L.V for L in self.lumps if L.state in IN_LINE}
        for b in self.beams.values():
            b.sample(t, list(self.scene.values()))
        com = {L.k: self.data.xipos[L.body].copy() for L in self.lumps if L.state in IN_LINE}
        fr = self.vision.frame(t, self.scene, self.dist, com)
        self.feed_view = self.ideal.frame(t, self.scene, self.dist, com) if self.ideal else fr
        return fr

    def scan(self, t):
        """The scanner's verdict on the measuring zone now."""
        (x0, y0, _), (x1, y1, _) = self.d['devices']['scanner']['zone']
        seen = {k: V for k, V in self.scene.items()
                if V[:, 0].max() > x0 and V[:, 0].min() < x1 and V[:, 1].max() > y0 and V[:, 1].min() < y1
                and V[:, 2].max() > self.d['station']['top_z']}
        st = self.d['station']
        belt = ((st['measure']['x0'], st['y_c'] - st['width_m'] / 2 - .005),
                (st['measure']['x1'], st['y_c'] + st['width_m'] / 2 + .005))
        return self.scanner.scan(t, seen, lambda k: self.blocks[k]['volume_m3'], self.dist, belt)

    def report(self):
        return dict(sensing=self.cfg['sensing'],
                    feeder_sensing='oracle' if self.ideal or self.cfg['sensing'] == 'oracle' else 'vision',
                    vision=self.vision.report(),
                    beams={n: b.report() for n, b in self.beams.items()},
                    scanner=dict(scans=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()
                                         if k != '_truth'} | dict(truth_lumps=list(s.get('_truth', ())))
                                        for s in self.scanner.scans],
                                 fail_on=sorted(self.scanner.fail_on)),
                    faults=self.faults, scenario=self.cfg.get('scenario', 'none'))


def _pose(model, data, L, x, y, zlow, yaw=0.):
    """Put lump L flat (its generated frame), yaw about z, centred on (x, y) in plan, lowest point at zlow."""
    L.place(data, x, y, 0., yaw)
    data.qvel[L.v:L.v + 6] = 0.
    mujoco.mj_forward(model, data)
    V = block_world_vertices(data, dict(geom=L.geom, local=L.local))
    data.qpos[L.q] += x - (V[:, 0].min() + V[:, 0].max()) / 2
    data.qpos[L.q + 1] += y - (V[:, 1].min() + V[:, 1].max()) / 2
    data.qpos[L.q + 2] += zlow - V[:, 2].min()
    mujoco.mj_forward(model, data)


class TakeOff:
    """Takes a held item off the line the moment the station holds it (see the module docstring)."""

    def __init__(self, d, model, data, lumps):
        self.d, self.model, self.data, self.lumps = d, model, data, lumps
        self.st = d['station']
        self.log = []

    def act(self, t, station, sensors):
        held = station._stage('hold')
        if not held:
            return
        m, y_c, w = self.st['measure'], self.st['y_c'], self.st['width_m']
        # everything over M (straddling the joint included), and every lump of the held item (truth)
        ks = {L.k for L in self.lumps if L.state in IN_LINE and L.V[:, 0].max() > m['x0']
              and L.V[:, 0].min() < m['x1'] + .05 and abs(L.last_pose[1] - y_c) < w / 2 + .15
              and L.V[:, 2].max() > self.st['top_z']}
        ks |= {k for it in held for o in it['objs'] for k in o['_truth']}
        for k in sorted(ks):
            L = self.lumps[k]
            _pose(self.model, self.data, L, self.d['belt_x0'] - 3. - PARK_GAP * len(self.log), -3.,
                  self.st['separator']['floor_z'] + .005)
            L.state = 'taken_off'
            self.log.append(dict(t_s=round(t, 3), lump=k, items=[it['item'] for it in held]))
        sensors.vision.taken_off(t, ks)
        station.took_off(t, sorted(ks))


class Scenario:
    """--scenario: set up at t = 0 (touching) or act once while the first item is weighed (off_scale_*)."""

    def __init__(self, cfg, d, model, data, lumps, blocks):
        self.kind = cfg.get('scenario', 'none')
        self.cfg, self.d, self.model, self.data, self.lumps, self.blocks = cfg, d, model, data, lumps, blocks
        self.done, self.info = self.kind == 'none', {}

    def setup(self):
        """The two shortest lumps (long axis) laid flat, end to end, 3 mm apart (3-D) at t = 0:
          touching       on the buffer belt, from 3 cm past its start: B and M are flush, so they reach the
                         measuring belt together and are weighed together unless the station stops them;
          touching_lane  on the lane centre line after the lane entry: the step onto B parts them (the front
                         one tips first), and the station must still not let anything through as valid."""
        if self.kind not in ('touching', 'touching_lane'):
            return
        d, cfg = self.d, self.cfg
        pair = sorted(self.lumps, key=lambda L: self.blocks[L.k]['axes'][0])[:2]
        ext = lambda L: self.blocks[L.k]['axes'][0]
        st = d['station']
        if self.kind == 'touching':
            y, x_rear, zl = st['y_c'], st['buffer']['x0'] + .03, st['top_z'] + .006
            limit = st['buffer']['x1']
        else:
            y, x_rear, zl = cfg['lane_y'] + cfg['lane_w'] / 2, float(d['lane_out_start']) + .03, cfg.get('feed_drop_height', .006)
            limit = st['buffer']['x0'] - .05
        _pose(self.model, self.data, pair[0], x_rear + ext(pair[0]) / 2, y, zl)
        V0 = block_world_vertices(self.data, dict(geom=pair[0].geom, local=pair[0].local))
        _pose(self.model, self.data, pair[1], float(V0[:, 0].max()) + ext(pair[1]) / 2 + .002, y, zl)
        for _ in range(4):                                     # close the 3-D gap to about 3 mm
            gap = float(mujoco.mj_geomDistance(self.model, self.data, pair[0].geom, pair[1].geom, .5, np.zeros(6)))
            if abs(gap - .003) < .001:
                break
            self.data.qpos[pair[1].q] -= gap - .003
            mujoco.mj_forward(self.model, self.data)
        V1 = block_world_vertices(self.data, dict(geom=pair[1].geom, local=pair[1].local))
        if V1[:, 0].max() > limit:
            raise RuntimeError('%s scenario: the pair (%.2f m) does not fit' % (self.kind, float(V1[:, 0].max()) - x_rear))
        self.info = dict(lumps=[pair[0].k, pair[1].k], gap_m=round(gap, 4), front_x=round(float(V1[:, 0].max()), 3))
        self.done = True

    def act(self, t, station):
        """off_scale_*: once, 0.2 s after the first item on the measuring belt is at rest."""
        if self.done:
            return
        it = next((e for e in station.items if e['stage'] == 'measure' and t >= e['t_stop_s'] + .5), None)
        if it is None:
            return
        st = self.d['station']
        x_j = st['buffer']['x1']
        on_m = [L for L in self.lumps if L.state in IN_LINE and L.V[:, 0].min() > x_j - .02
                and L.V[:, 0].max() < st['measure']['x1'] + .02 and L.V[:, 2].max() > st['top_z']]
        if len(on_m) != 1:
            return
        W = on_m[0]
        rear = float(W.V[:, 0].min())
        busy_b = lambda x0: any(L.state in IN_LINE and L.k != W.k and L.V[:, 0].max() > x0 - .03
                                and L.V[:, 0].min() < x_j + .02 and L.V[:, 2].max() > st['top_z'] for L in self.lumps)
        if self.kind == 'off_scale_neighbour':
            queued = [L for L in self.lumps if L.state in IN_LINE and L.last_pose[0] < self.d['feeder']['x1']]
            if not queued:
                return
            N = min(queued, key=lambda L: L.last_pose[0])       # the last one on the feed belt
            back = (N.last_pose[0], N.last_pose[1], self.data.qpos[N.q:N.q + 7].copy())
            _pose(self.model, self.data, N, x_j - 1., st['y_c'], st['top_z'] + .004)
            V = block_world_vertices(self.data, dict(geom=N.geom, local=N.local))
            shift = rear - .004 - float(V[:, 0].max())
            if busy_b(float(V[:, 0].min()) + shift):
                self.data.qpos[N.q:N.q + 7] = back[2]                 # put it back: try at the next item
                mujoco.mj_forward(self.model, self.data)
                return
            self.data.qpos[N.q] += shift
            mujoco.mj_forward(self.model, self.data)
            self.info = dict(t_s=round(t, 3), item=it['item'], weighed_lump=W.k, laid_against=N.k,
                             overlap_on_measuring_belt_m=round(rear - .004 - x_j, 3))
        else:
            back = rear - (x_j - .12)
            if busy_b(x_j - .12):
                return
            self.data.qpos[W.q] -= back
            self.data.qpos[W.q + 2] += .003
            self.data.qvel[W.v:W.v + 6] = 0.
            mujoco.mj_forward(self.model, self.data)
            self.info = dict(t_s=round(t, 3), item=it['item'], weighed_lump=W.k, moved_back_m=round(back, 3))
        self.done = True


class SectionWatch:
    """Vision only: the singulation section (plough start - 0.3 m .. lane exit) as the cameras see it -- which
    objects are in it, which have not advanced jam_speed * jam_window over jam_window (the stall rule on the
    tracked outline centroid), and whether a tail left the lane recently."""

    def __init__(self, cfg, d, lo):
        self.cfg, self.lo, self.exit_x = cfg, lo, d['exit_x']
        self.hist, self.since, self.tail = {}, {}, {}

    def update(self, t, view, waiting):
        W = self.cfg['jam_window']
        section = {tid: o for tid, o in view.items() if o['x0'] < self.exit_x and o['x1'] > self.lo}
        for tid, o in view.items():
            h = self.hist.setdefault(tid, [])
            h.append((t, o['cx']))
            while len(h) > 2 and h[1][0] <= t - W + 1e-9:
                h.pop(0)
            if o['x0'] > self.exit_x and tid not in self.tail and o['x0'] < self.exit_x + .5:
                self.tail[tid] = t
        stuck = []
        for tid, o in section.items():
            if waiting(o):
                self.since.pop(tid, None)
                continue
            s = self.since.setdefault(tid, t)
            h = self.hist[tid]
            if t - s >= W - 1e-9 and h[0][0] <= t - W + .02 and h[-1][1] - h[0][1] < self.cfg['jam_speed'] * W:
                stuck.append(tid)
        for tid in [k for k in self.since if k not in section]:
            del self.since[tid]
        passing = any(tt > t - W for tt in self.tail.values())
        return section, stuck, passing

    @staticmethod
    def plans(section):
        """Outlines grown by the position margin, for the face's closing-sweep check."""
        out = {}
        for tid, o in section.items():
            P = o['pts']
            c = P.mean(0)
            r = np.maximum(np.linalg.norm(P - c, axis=1, keepdims=True), 1e-9)
            out[tid] = P + (P - c) / r * V_MARGIN
        return out
