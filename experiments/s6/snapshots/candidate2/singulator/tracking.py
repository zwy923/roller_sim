"""What is observed about each lump during a run, and the stall rule built on it.

Every 10 ms each lump's true outline is placed in the world (geom frame of the compiled mesh), and its
front edge, centroid, contacts, lane and measuring-plane crossings are recorded. Front arrival and tail
passage are kept apart: a batch is through when every TAIL has passed the measuring plane.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull

from . import station
from .lumps import block_world_vertices, exit_cut

# contact categories, by the equipment a lump touches (bit i of trajectory.npz 'touch' = CATS[i]): 'belt' is the
# feed and main belts, 'buffer' the buffer belt with its skirts, 'weigher' the measuring belt with its (the weigh
# frame), 'device' the sensor hardware (singulator/devices.py)
CATS = ('other', 'belt', 'plough', 'block', 'skirt', 'lane_wall', 'side_belt', 'buffer', 'weigher',
        'separator', 'device')


def categorise(model):
    """Category index of every geom (see CATS)."""
    import mujoco
    G, B = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
    cat = np.zeros(model.ngeom, dtype=np.int8)
    for g in range(model.ngeom):
        gname = mujoco.mj_id2name(model, G, g) or ''
        bname = mujoco.mj_id2name(model, B, model.geom_bodyid[g]) or ''
        if bname in ('mfloor', 'feeder', 'fdrum', 'mtail'):
            cat[g] = 1
        elif gname.startswith('plough') or (bname.startswith('fr') and bname[2:].isdigit()):
            cat[g] = 2
        elif gname.startswith('bg'):
            cat[g] = 3
        elif gname.startswith('skirt'):
            cat[g] = 4
        elif gname == 'lane_out':
            cat[g] = 5
        elif bname.startswith('vslat'):
            cat[g] = 6
        elif bname == 'bbelt' or gname.startswith('bskirt'):
            cat[g] = 7
        elif bname == 'mbelt' or gname.startswith('mskirt'):
            cat[g] = 8
        elif bname.startswith('sep_'):
            cat[g] = 9
        elif gname.startswith('dev_'):
            cat[g] = 10
    return cat


def yaw_deg(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def misalignment(yaw):
    """Angle between the lump's long axis and the flow direction, 0..90 deg."""
    return abs(((yaw + 90.) % 180.) - 90.)


class Lump:
    def __init__(self, model, k):
        import mujoco
        nid = lambda obj, s: mujoco.mj_name2id(model, obj, s)
        J, B, G = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
        g = nid(G, 'bg%d' % k)
        mesh = model.geom_dataid[g]
        adr, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        self.k, self.geom, self.body = k, g, nid(B, 'b%d' % k)
        self.q, self.v = model.jnt_qposadr[nid(J, 'bj%d' % k)], model.jnt_dofadr[nid(J, 'bj%d' % k)]
        self.local = model.mesh_vert[adr:adr + n].copy()      # geom-frame vertices: see block_world_vertices
        tri = ConvexHull(self.local).simplices
        self.edges = np.array(sorted({tuple(sorted((int(fc[i]), int(fc[(i + 1) % 3])))) for fc in tri for i in range(3)}))
        self.state = 'on_belt'                  # on_belt / passed (tail past the head edge) / sorted / dropped /
                                                # taken_off (a held item's lump)
        self.bin = None                         # the separator bin it landed in (coal / gangue)
        self.enter = self.leave = self.pass_t = None   # measuring plane: front arrival, last cut, tail past
        self.lane_t = self.lane_tail_t = self.lane_yaw = self.lane_cut = None
        self.cut_n = self.face_yaw = None
        self.drop_t = self.drop_x = None
        self.s_hist, self.c_hist = [], []       # (t, front x), (t, centroid x)
        self.eligible_since = None
        self.touched_plough = False
        self.stalled_s = 0.
        self.feeder_wait_s = 0.                 # queued on the stopped feed belt
        self.station_wait_s = 0.                # held for, or taken by, the station
        self.last_pose = None
        self.plan = self.ztop = None

    def place(self, data, x, y, z, yaw):
        data.qpos[self.q:self.q + 3] = (x, y, z)
        data.qpos[self.q + 3:self.q + 7] = (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))

    def observe(self, data, t, touch, cfg, d, waiting, dt_sample, slow):
        """One 10 ms sample. A sample whose centroid advances less than `slow` counts as stalled, or as
        planned waiting when waiting(lump) says so ('feeder': queued on the feed belt; 'station': held).
        Returns (cut at the measuring plane or None, True if the front reached it just now)."""
        R = data.xmat[self.body].reshape(3, 3)            # body frame: long axis = body x, for yaw
        V = block_world_vertices(data, dict(geom=self.geom, local=self.local))
        self.V = V                                         # world outline; the feeder's drop beam tests it
        com = data.xipos[self.body]
        self.s_hist.append((t, float(V[:, 0].max())))
        self.c_hist.append((t, float(com[0])))
        self.plan, self.ztop = V[:, :2].copy(), float(V[:, 2].max())
        if 'plough' in touch:
            self.touched_plough = True
        self.last_pose = (float(com[0]), float(com[1]), float(com[2]), float(V[:, 0].min()), float(V[:, 0].max()),
                          float(V[:, 1].min()), float(V[:, 1].max()), yaw_deg(R), sorted(touch))
        if V[:, 2].min() < d['drop_z'] and self.drop_t is None:
            self.drop_t, self.drop_x = t, float(V[:, 0].max())
            self.state, self.bin = station.landing(d['station'], com)
        if len(self.c_hist) > 1 and self.state == 'on_belt' and t > cfg['ramp'] + 1.:
            if self.c_hist[-1][1] - self.c_hist[-2][1] < slow:
                w = waiting(self)
                if w == 'feeder':
                    self.feeder_wait_s += dt_sample
                elif w == 'station':
                    self.station_wait_s += dt_sample
                else:
                    self.stalled_s += dt_sample
        lane_cut = exit_cut(V, self.edges, d['exit_x'])
        if lane_cut is not None and self.lane_t is None:
            self.lane_t, self.lane_cut = t, lane_cut
            self.lane_yaw = misalignment(yaw_deg(R))
        if self.lane_tail_t is None and V[:, 0].min() > d['exit_x']:
            self.lane_tail_t = t
        cut = exit_cut(V, self.edges, d['final_x'])
        first = False
        if cut is not None:
            if self.enter is None:
                self.enter, self.cut_n, first = t, cut, True
                self.face_yaw = misalignment(yaw_deg(R))
            self.leave = t
        elif self.enter is not None and V[:, 0].min() > d['final_x'] and self.state == 'on_belt':
            self.state = 'passed'
            self.pass_t = t
        return cut, first

    def centroid_progress(self, n_win):
        h = self.c_hist
        return h[-1][1] - h[-1 - n_win][1] if len(h) > n_win else math.inf

    def summary(self):
        p = self.last_pose
        return dict(state=self.state, front_at_plane_s=self.enter, tail_past_plane_s=self.pass_t,
                    left_s=self.leave,
                    cut_n_extent_m=None if self.cut_n is None else round(self.cut_n[1] - self.cut_n[0], 3),
                    yaw_misalignment_deg=None if self.face_yaw is None else round(self.face_yaw, 1),
                    touched_plough=self.touched_plough, stalled_s=round(self.stalled_s, 2),
                    feeder_wait_s=round(self.feeder_wait_s, 2), station_wait_s=round(self.station_wait_s, 2),
                    lane_front_s=self.lane_t, lane_tail_s=self.lane_tail_t,
                    lane_yaw_misalignment_deg=None if self.lane_yaw is None else round(self.lane_yaw, 1),
                    lane_cut_width_m=None if self.lane_cut is None else round(self.lane_cut[1] - self.lane_cut[0], 3),
                    drop_x_m=None if self.drop_x is None else round(self.drop_x, 2),
                    **({} if self.bin is None else dict(bin=self.bin, landed_s=round(self.drop_t, 3))),
                    final_pose=None if p is None else dict(
                        x=round(p[0], 3), y=round(p[1], 3), z=round(p[2], 3),
                        x_range=[round(p[3], 3), round(p[4], 3)], y_range=[round(p[5], 3), round(p[6], 3)],
                        yaw_deg=round(p[7], 1), touching=p[8]))


def stalled(lumps, in_section, waiting, t, cfg, n_win):
    """Stall rule (2026-09-22): a lump that has sat in the singulation section for a whole jam_window
    without its CENTROID advancing jam_speed * jam_window. Centroid, not front edge: a lump turning in
    place moves its front, not its centre. Per lump, not the batch maximum: one moving lump used to mask
    an arch among the others. Planned waiting (waiting: on the feed belt, held by the station) is not a
    candidate."""
    for L in lumps:
        if L.k not in in_section or waiting(L):
            L.eligible_since = None
        elif L.eligible_since is None:
            L.eligible_since = t
    return [L.k for L in lumps if L.eligible_since is not None
            and t - L.eligible_since >= cfg['jam_window'] - 1e-9
            and L.centroid_progress(n_win) < cfg['jam_speed'] * cfg['jam_window']]


def lane_discharging(lumps, t, window):
    """A lump's tail left the lane within the window: the section is still emptying."""
    return any(L.lane_tail_t is not None and L.lane_tail_t > t - window for L in lumps)
