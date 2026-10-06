"""The lumps in the running model: each one's true state, sampled every 10 ms.

Every sample each lump's true outline is placed in the world (geom frame of the compiled mesh), and its
front edge, centroid, contacts, lane and measuring-plane crossings are recorded. Front arrival and tail
passage are kept apart: a batch is through when every TAIL has passed the measuring plane.

This is the truth: the sensors are formed from it (sensing/), the record and the verification are written from it.
No controller reads it.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull

from ..machine import station
from ..machine.assembly import LUMP_BODY, LUMP_GEOM, LUMP_JOINT

IN_LINE = ('on_belt', 'passed')   # still on the line ('passed' = tail past the singulator's plane, the head edge)
END_STATES = ('sorted', 'dropped', 'taken_off')   # the states that end a lump's run


def world_vertices(data, geom, local):
    """World vertices of a lump. `local` are the COMPILED mesh vertices, which MuJoCo re-centres on the
    centroid and turns onto the principal axes, compensating in the geom's own pos/quat -- so they must
    be placed with the geom frame. Placing them with the body frame (as before 2026-09-22) was off by up
    to 0.18 m on seed 81 and corrupted every observation built on it."""
    return data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T


def exit_cut(V, edges, x, span=None):
    """y-interval of a convex block's cross-section with the vertical plane at x, or None if it does not reach it.
    span: (min, max) of V's x, when known -- a block wholly on one side of the plane has no edge crossing it."""
    if span is not None and not span[0] <= x <= span[1]:
        return None
    P, Q = V[edges[:, 0]], V[edges[:, 1]]
    dp, dq = P[:, 0] - x, Q[:, 0] - x
    m = (dp * dq <= 0) & (dp != dq)
    if not m.any():
        return None
    y = P[m, 1] + dp[m] / (dp[m] - dq[m]) * (Q[m, 1] - P[m, 1])
    return float(y.min()), float(y.max())


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
        g = nid(G, LUMP_GEOM % k)
        mesh = model.geom_dataid[g]
        adr, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        self.k, self.geom, self.body = k, g, nid(B, LUMP_BODY % k)
        self.q, self.v = model.jnt_qposadr[nid(J, LUMP_JOINT % k)], model.jnt_dofadr[nid(J, LUMP_JOINT % k)]
        self.local = model.mesh_vert[adr:adr + n].copy()      # geom-frame vertices: see world_vertices
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
        self.touched_plough = False
        self.stalled_s = 0.
        self.feeder_wait_s = 0.                 # queued on the stopped feed belt
        self.station_wait_s = 0.                # held for, or taken by, the station
        self.last_pose = None
        self.plan = self.ztop = None

    def place(self, data, x, y, z, yaw):
        data.qpos[self.q:self.q + 3] = (x, y, z)
        data.qpos[self.q + 3:self.q + 7] = (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))

    def world(self, data):
        """Its outline now: world vertices (n, 3), from the poses of the last mj_forward."""
        return world_vertices(data, self.geom, self.local)

    def observe(self, data, t, touch, cfg, d, waiting, dt_sample, slow):
        """One 10 ms sample. A sample whose centroid advances less than `slow` counts as stalled, or as
        planned waiting when waiting(lump) says so ('feeder': queued on the feed belt; 'station': held).
        Returns (cut at the measuring plane or None, True if the front reached it just now)."""
        R = data.xmat[self.body].reshape(3, 3)            # body frame: long axis = body x, for yaw
        V = self.world(data)
        self.V = V                                         # world outline; the feeder's drop beam tests it
        (x0, y0, z0), (x1, y1, z1) = self.box = V.min(0).tolist(), V.max(0).tolist()   # the beams test it too
        com = data.xipos[self.body]
        self.s_hist.append((t, x1))
        self.c_hist.append((t, float(com[0])))
        self.plan, self.ztop = V[:, :2].copy(), z1
        if 'plough' in touch:
            self.touched_plough = True
        self.last_pose = (float(com[0]), float(com[1]), float(com[2]), x0, x1, y0, y1, yaw_deg(R), sorted(touch))
        if z0 < d['drop_z'] and self.drop_t is None:
            self.drop_t, self.drop_x = t, x1
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
        lane_cut = exit_cut(V, self.edges, d['exit_x'], (x0, x1))
        if lane_cut is not None and self.lane_t is None:
            self.lane_t, self.lane_cut = t, lane_cut
            self.lane_yaw = misalignment(yaw_deg(R))
        if self.lane_tail_t is None and x0 > d['exit_x']:
            self.lane_tail_t = t
        cut = exit_cut(V, self.edges, d['final_x'], (x0, x1))
        first = False
        if cut is not None:
            if self.enter is None:
                self.enter, self.cut_n, first = t, cut, True
                self.face_yaw = misalignment(yaw_deg(R))
            self.leave = t
        elif self.enter is not None and x0 > d['final_x'] and self.state == 'on_belt':
            self.state = 'passed'
            self.pass_t = t
        return cut, first

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
