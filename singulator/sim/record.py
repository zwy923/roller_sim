"""What a run records of the true state every 10 ms: each lump's own record, who is at the measuring plane
together, the funnel's occupancy, and the trajectory that goes to trajectory.npz.

None of it reaches a controller. The measuring plane is the main belt's head edge; a 'cut' is a lump's
cross-section with that plane.
"""
import math

import mujoco
import numpy as np

from ..config import SAMPLE_S
from ..machine.assembly import CATS, categorise

RIDING_OVERLAP = .05      # m: two lumps at the plane whose cuts overlap by more than this across the belt: one rides
                          # over the other (they are not side by side)


class Trace:
    def __init__(self, cfg, d, model, lumps):
        self.cfg, self.d, self.model, self.lumps = cfg, d, model, lumps
        self.cat = categorise(model)
        self.cat_list = self.cat.tolist()
        self.geom_lump = {L.geom: L.k for L in lumps}
        dt = model.opt.timestep
        stride = max(1, round(SAMPLE_S / dt))
        self.dt_sample = stride * dt                # s between samples
        self.slow = .02 * stride * dt               # m: a centroid advancing less than this in a sample stands
        self.crossings = []                         # (t, lump): fronts reaching the measuring plane
        self.together, self.riding = dict(s=0., pairs={}, max_total_n=0.), 0.
        # the plough funnel, plough start to lane entry: how many lumps reach into it at once
        self.funnel = dict(x_from_m=round(float(d['P0'][0]), 4), x_to_m=round(float(d['lane_out_start']), 4),
                           max_lumps=0, two_or_more_s=0.)
        self.touch = {}                             # {lump: contact categories} of this sample
        self.traj = dict(t=[], front_x=[], pos=[], quat=[], com=[], touch=[], face_deg=[], face_contact_N=[],
                         drive_f=[], weigh_N=[], sep_deg=[])

    def observe(self, t, data, waiting):
        """Every lump's sample (physics.lumps.Lump.observe), and what they are doing together. waiting(lump):
        'feeder' / 'station' / False -- is the lump waiting by plan, for its stall record."""
        lumps, dt_sample = self.lumps, self.dt_sample
        touch = self.touch = {L.k: set() for L in lumps}
        for g1, g2 in data.contact.geom[:data.ncon].tolist():
            for ga, gb in ((g1, g2), (g2, g1)):
                if ga in self.geom_lump:
                    touch[self.geom_lump[ga]].add(CATS[self.cat_list[gb]])
        live = []
        for L in lumps:
            if L.state == 'taken_off':
                continue                       # a held item's lump, off the line
            cut, first = L.observe(data, t, touch[L.k], self.cfg, self.d, waiting, dt_sample, self.slow)
            if first:
                self.crossings.append((t, L.k))
            if cut is not None:
                live.append((L.k, cut))
        together = self.together
        if len(live) >= 2:
            together['s'] += dt_sample
            together['max_total_n'] = max(together['max_total_n'], sum(c[1] - c[0] for _, c in live))
            for i in range(len(live)):
                for j in range(i + 1, len(live)):
                    (ka, ca), (kb, cb) = live[i], live[j]
                    key = '%d-%d' % (min(ka, kb), max(ka, kb))
                    together['pairs'][key] = together['pairs'].get(key, 0.) + dt_sample
                    if min(ca[1], cb[1]) - max(ca[0], cb[0]) > RIDING_OVERLAP:
                        self.riding += dt_sample
        funnel = self.funnel
        n_funnel = sum(1 for L in lumps if L.state == 'on_belt' and L.last_pose[4] > funnel['x_from_m']
                       and L.last_pose[3] < funnel['x_to_m'])
        funnel['max_lumps'] = max(funnel['max_lumps'], n_funnel)
        funnel['two_or_more_s'] += dt_sample * (n_funnel >= 2)

    def face_force(self, data):
        """Total lump-face normal force now, N."""
        face_N, wrench = 0., np.zeros(6)
        pair, cat = {CATS.index('block'), CATS.index('plough')}, self.cat_list
        for ci, (g1, g2) in enumerate(data.contact.geom[:data.ncon].tolist()):
            if {cat[g1], cat[g2]} != pair:
                continue
            mujoco.mj_contactForce(self.model, data, ci, wrench)
            face_N += float(wrench[0])
        return face_N

    def frame(self, t, data, face_theta, face_N, drive_f, weigh_N, sep_deg):
        """One row of the trajectory: body pose (pos + quat, which with model.xml reconstructs each lump exactly),
        centroid, front edge, touched categories as a bit mask over CATS, face angle, total lump-face normal force,
        drive speed factors, load-cell reading and plate angle."""
        lumps, traj = self.lumps, self.traj
        traj['t'].append(t)
        traj['front_x'].append([L.s_hist[-1][1] for L in lumps])
        traj['pos'].append([data.xpos[L.body].copy() for L in lumps])
        traj['quat'].append([data.xquat[L.body].copy() for L in lumps])
        traj['com'].append([data.xipos[L.body].copy() for L in lumps])
        traj['touch'].append([sum(1 << CATS.index(c) for c in self.touch[L.k]) for L in lumps])
        traj['face_deg'].append(math.degrees(face_theta))
        traj['face_contact_N'].append(face_N)
        traj['drive_f'].append(drive_f)
        traj['weigh_N'].append(weigh_N)
        traj['sep_deg'].append(sep_deg)
