"""What the simulation does to the scene that the machine does not: acceptance scenarios, and taking a held item
off the line.

  Scenario   --scenario touching / touching_lane / off_scale_neighbour / off_scale_bridge: the acceptance tests of
             2026-09-30 (故意让两块相贴、料搭在秤外；扫描失败 = --fault scan_fail:0);
  TakeOff    a held (void) item's lumps leave the line at once and lie on the floor far away (state 'taken_off').
             This stands in for the prototype's manual separation and re-measuring, which is not modelled (user
             2026-09-30: 不需要模拟人工操作); the stop itself is counted by the station.

Both move lumps by writing their true pose: they are the simulation's hand, not a controller.
"""
import mujoco
import numpy as np

from ..physics.lumps import IN_LINE
from .layouts import lay_flat

PARK_GAP = 1.3        # m: lumps taken off lie this far apart on the floor, upstream of the machine


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
            lay_flat(self.model, self.data, L, self.d['belt_x0'] - 3. - PARK_GAP * len(self.log), -3.,
                  self.st['separator']['floor_z'] + .005)
            L.state = 'taken_off'
            self.log.append(dict(t_s=round(t, 3), lump=k, items=[it['item'] for it in held]))
        sensors.vision.taken_off(t, ks)
        station.took_off(t, sorted(ks))


class Scenario:
    """--scenario: set up at t = 0 (touching) or act once while the first item is weighed (off_scale_*)."""

    def __init__(self, cfg, d, model, data, lumps, blocks):
        self.kind = cfg['scenario']
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
            y, x_rear, zl = cfg['lane_y'] + cfg['lane_w'] / 2, float(d['lane_out_start']) + .03, cfg['feed_drop_height']
            limit = st['buffer']['x0'] - .05
        lay_flat(self.model, self.data, pair[0], x_rear + ext(pair[0]) / 2, y, zl)
        V0 = pair[0].world(self.data)
        lay_flat(self.model, self.data, pair[1], float(V0[:, 0].max()) + ext(pair[1]) / 2 + .002, y, zl)
        for _ in range(4):                                     # close the 3-D gap to about 3 mm
            gap = float(mujoco.mj_geomDistance(self.model, self.data, pair[0].geom, pair[1].geom, .5, np.zeros(6)))
            if abs(gap - .003) < .001:
                break
            self.data.qpos[pair[1].q] -= gap - .003
            mujoco.mj_forward(self.model, self.data)
        V1 = pair[1].world(self.data)
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
            lay_flat(self.model, self.data, N, x_j - 1., st['y_c'], st['top_z'] + .004)
            V = N.world(self.data)
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
