"""Retract action of the hinged plough face, and the bookkeeping of what it met.

The default face is a finite-torque physical hinge with a rate-limited position servo and soft joint
limits. Carrier mass/inertia and controller gains are assumptions. A kinematic diagnostic option is
retained for model comparisons. Torque divided by chord is not cylinder force or total contact force.
Whether the face touches material is measured separately, from the lump-face contact pairs.

Phases: idle -> out (swing to face_swing_deg) -> hold (retracted) -> back (closing) -> idle, or fault.
  * Closing starts only when the section is clear AND no lump reaches into the area the face will sweep
    on its way home (the whole lump outline).
  * While closing the same sweep check runs every sample; a lump entering it pauses the face (back to
    hold) until it is clear again.
  * Retracted for face_hold_max_s without being allowed to close: the batch stops for the operator with
    the face open. It never closes onto material.
  * Only a face that actually reaches home goes idle. If it has not got home 6 x its swing time after
    closing started, that is a reset fault: the batch stops and no further action is taken.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull

from .drives import load_stats, make_motor, motor_update
from .machine import MOTION_CLEARANCE, face_parts


def polygons_overlap(P, H, margin):
    """Separating-axis test of the convex polygons P (N, k, 2) against one convex polygon H (m, 2).
    True where P[i] and H come within `margin` of each other."""
    def normals(E):
        n = np.stack([-E[..., 1], E[..., 0]], -1)
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    A = normals(np.roll(P, -1, axis=1) - P)                       # (N, k, 2) axes of P
    pP, pH = np.einsum('nkd,njd->nkj', P, A), np.einsum('md,njd->nmj', H, A)
    sep = ((pP.max(1) + margin < pH.min(1)) | (pH.max(1) + margin < pP.min(1))).any(-1)
    a = normals(np.roll(H, -1, axis=0) - H)                       # (m, 2) axes of H
    qP, qH = np.einsum('nkd,jd->nkj', P, a), H @ a.T
    sep |= ((qP.max(1) + margin < qH.min(0)) | (qH.max(0) + margin < qP.min(1))).any(-1)
    return ~sep


class Leaf:
    """The face as a kinematically driven hinged part: its angle and commanded rate, a force-limited drive
    referred to the far end (lever = hinge to far end), and the loads it met."""

    def __init__(self, joint, parts, piv, lever, delta, swing_s, force_max, slip, dt):
        self.joint, self.parts, self.piv = joint, parts, piv      # joint = (qpos adr, dof adr)
        self.lever, self.delta, self.swing_s, self.dt = lever, delta, swing_s, dt
        v_tip = abs(delta) * lever / swing_s
        self.drive = make_motor(True, force_max, 50., max(v_tip, 1e-6), slip)
        self.theta = self.omega = 0.
        self.reached = False
        self.loads, self.hold, self.by_phase = [], [], dict(out=[], back=[])

    def write(self, phase, data):
        """Write the hinge position/velocity for the coming step."""
        q, v = self.joint
        if phase in ('out', 'back'):
            w = self.omega * self.drive['f']
            nxt = self.theta + w * self.dt
            if phase == 'back' and nxt * self.theta < 0:     # do not overshoot home
                nxt, w = 0., -self.theta / self.dt
            if phase == 'out' and abs(nxt) > abs(self.delta):
                nxt, w = self.delta, (self.delta - self.theta) / self.dt
            data.qpos[q] = self.theta
            data.qvel[v] = w
            self.theta = nxt
        else:
            data.qpos[q] = self.theta
            data.qvel[v] = 0.

    def record(self, t, phase, data, ramp):
        """After the step: hinge moment while moving (drive load) or while standing (pin/stop/frame)."""
        tau = float(data.qfrc_constraint[self.joint[1]])
        if phase in ('out', 'back') and self.omega:
            tip_load = tau * math.copysign(1., self.omega) / self.lever    # negative resists
            motor_update(self.drive, 1., tip_load, self.dt)
            self.drive['loads'].append(tip_load)
            self.loads.append(tip_load)
            self.by_phase[phase].append(tip_load)
        else:
            self.drive['f'] = 0.
            # what the pin, the stop and the frame carry while the part just stands there: the material
            # leans on it the whole time, and that load is NOT what the actuator has to supply
            if t > ramp + .2:
                self.hold.append(abs(tau) / self.lever)

    def sweep(self):
        """Plan-view polygons of the part at angles from the present one back to home (<= 1 deg apart),
        i.e. everything it will pass through while closing."""
        n = max(2, int(math.ceil(abs(self.theta) / math.radians(1.))) + 1)
        out = []
        for th in np.linspace(self.theta, 0., n):
            c, s = math.cos(th), math.sin(th)
            out.append(self.piv + (self.parts - self.piv) @ np.array([[c, s], [-s, c]]))
        return np.concatenate(out) if out else self.parts


class DynamicLeaf(Leaf):
    """Finite-torque position servo. MuJoCo alone advances the hinge position and velocity.

    The reference moves at the configured swing rate but stays close to the measured angle, so a
    stalled actuator cannot accumulate a whole stroke of position error. Joint limits are soft stops.
    """
    def __init__(self, *args, actuator, cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.actuator, self.cfg = actuator, cfg
        self.reference = 0.
        self.last_phase = 'idle'
        self.saturation_s = 0.
        self.torque_peak = self.power_peak = 0.
        self.work_J = 0.

    def write(self, phase, data):
        q, v = self.joint
        self.theta = float(data.qpos[q])
        self.start_omega = float(data.qvel[v])
        if phase in ('out', 'back'):
            goal = self.delta if phase == 'out' else 0.
            rate = abs(self.delta) / self.swing_s
            self.reference += float(np.clip(goal - self.reference, -rate*self.dt, rate*self.dt))
            # Position error corresponding to the force limit; prevents reference wind-up at a jam.
            max_error = self.drive['F_max'] * self.lever / self.cfg['face_kp']
            self.reference = float(np.clip(self.reference, self.theta-max_error, self.theta+max_error))
            self.reference = float(np.clip(self.reference, min(0., self.delta), max(0., self.delta)))
        elif phase != self.last_phase:
            self.reference = self.theta if phase != 'idle' else 0.
        data.ctrl[self.actuator] = self.reference
        self.last_phase = phase

    def record(self, t, phase, data, ramp):
        q, v = self.joint
        self.theta = float(data.qpos[q])
        self.actual_omega = float(data.qvel[v])
        torque = float(data.actuator_force[self.actuator])
        self.torque_peak = max(self.torque_peak, abs(torque))
        power = torque*(self.start_omega + self.actual_omega)/2
        self.power_peak = max(self.power_peak, abs(power))
        self.work_J += power*self.dt
        self.saturation_s += self.dt * (abs(torque) >= .999*self.drive['F_max']*self.lever)
        ratio = abs(self.actual_omega) / (abs(self.delta)/self.swing_s)
        self.drive['f'] = ratio
        if phase in ('out', 'back'):
            # Applied actuator torque, expressed at the reference lever; includes acceleration.
            direction = math.copysign(1., self.delta) * (1 if phase == 'out' else -1)
            load = -torque*direction/self.lever
            self.loads.append(load)
            self.drive['loads'].append(load)
            self.by_phase[phase].append(load)
            self.drive['min_ratio'] = min(self.drive['min_ratio'], ratio)
            self.drive['stall_s'] += self.dt*(ratio < .5)
        else:
            self.hold.append(abs(torque)/self.lever)


class FaceRetract:
    def __init__(self, cfg, d, dt, joint=None, actuator=None):
        self.cfg, self.dt, self.joint = cfg, dt, joint        # joint = (qpos adr, dof adr) or None
        parts, piv = face_parts(cfg, d)
        # moment arm: the chord between the hinge and the far end -- NOT the polyline length, which for a
        # curved face is longer than the straight distance the tip actually travels
        self.dynamic = actuator is not None
        leaf = DynamicLeaf if self.dynamic else Leaf
        extra = dict(actuator=actuator, cfg=cfg) if self.dynamic else {}
        self.face = leaf(joint, parts, piv, float(np.linalg.norm(d['P1'] - d['P0'])),
                         math.radians(cfg['face_swing_deg']), cfg['face_swing_s'], cfg['face_force_max'],
                         cfg['motor_slip'], dt, **extra)
        self.phase, self.t0 = 'idle', 0.
        self.hold_since = None                              # first entry into hold of the current action
        self.closed_on_empty, self.fault = None, None
        self.contact_s = self.contact_max_N = 0.            # during the current action
        self.all_contact_s = self.all_contact_max_N = 0.    # whole run
        self.zone_busy = True                               # the section is not clear (set every sample)
        self.in_sweep = []                                  # lumps reaching into the closing sweep
        self.events, self.next_ok = [], 0.

    # the face's own state, as the rest of the model reads it
    theta = property(lambda self: self.face.theta)
    lever = property(lambda self: self.face.lever)
    drive = property(lambda self: self.face.drive)

    @property
    def at_home(self):
        """A phase label alone is insufficient now that load can deflect the physical actuator."""
        return self.phase == 'idle' and (not self.dynamic or
            (abs(self.theta) <= self.cfg['face_position_tol']
             and abs(getattr(self.face, 'actual_omega', 0.)) <= self.cfg['face_velocity_tol']))

    # ---- actions ------------------------------------------------------------------------------
    def start(self, t, **info):
        """Begin a retract; info is stored with the event."""
        if self.events:
            self.events[-1].update(self.action_summary())
        self.phase, self.t0 = 'out', t
        F = self.face
        F.reached, F.loads = False, []
        F.drive['f'] = 0.
        self.hold_since, self.contact_max_N, self.contact_s = None, 0., 0.
        c = self.cfg
        self.next_ok = t + 2 * c['face_swing_s'] + c['face_hold_s'] + c['jam_window'] + .5
        self.events.append(dict(t_s=round(t, 2), **info, pauses=0))

    def command(self, t, data):
        """Advance the phase machine and write the hinge positions/velocities for the coming step."""
        if self.joint is None:
            return
        c, F = self.cfg, self.face
        if self.phase == 'out':
            F.omega = F.delta / F.swing_s
            reached = (abs(F.theta-F.delta) <= c['face_position_tol']
                       and abs(getattr(F, 'actual_omega', 0.)) <= c['face_velocity_tol']) if self.dynamic else abs(F.theta) >= abs(F.delta)
            if reached or t - self.t0 > (6 if self.dynamic else 2) * F.swing_s:
                F.reached = reached
                F.omega = 0.
                self.phase, self.t0, self.hold_since = 'hold', t, t
                if self.dynamic and not reached:
                    self.fault = dict(t_s=round(t, 3), reason='retract_not_reached',
                                      angle_deg=math.degrees(F.theta))
                    self.phase = 'fault'
        elif self.phase == 'hold' and t - self.t0 >= c['face_hold_s'] and not self.zone_busy and not self.in_sweep:
            self.phase, self.t0, self.closed_on_empty = 'back', t, True
        elif self.phase == 'back' and self.in_sweep:
            # a lump has come into the path while closing: stop pushing, wait retracted again
            self.phase, self.t0 = 'hold', t
            F.omega = 0.
            self.events[-1]['pauses'] += 1
        if self.phase == 'back':
            F.omega = -math.copysign(abs(F.delta) / F.swing_s, F.theta) if F.theta else 0.
            home = (abs(F.theta) <= c['face_position_tol']
                    and abs(getattr(F, 'actual_omega', 0.)) <= c['face_velocity_tol']) if self.dynamic else abs(F.theta) < 1e-4
            if home and not self.dynamic:
                F.theta, F.omega = 0., 0.
            if home:
                self.phase = 'idle'
            elif t - self.t0 > 6 * F.swing_s:
                self.fault = dict(t_s=round(t, 2), reason='reset_not_reached', angle_deg=round(math.degrees(F.theta), 2))
                self.phase = 'fault'
                F.omega = 0.
        F.write(self.phase, data)

    def record(self, t, data):
        """After the step: hinge moment of the face."""
        if self.joint is None:
            return
        self.face.record(t, self.phase, data, self.cfg['ramp'])

    # ---- observation --------------------------------------------------------------------------
    def check_sweep(self, plans):
        """plans: {lump id: plan-view points}. Records which lumps reach into the face's closing sweep
        (within MOTION_CLEARANCE); checked only while the face is out of its home position."""
        self.in_sweep = []
        if self.joint is None or self.phase not in ('hold', 'back') or not len(self.face.parts):
            return self.in_sweep
        P = self.face.sweep()
        lo, hi = P.reshape(-1, 2).min(0) - MOTION_CLEARANCE, P.reshape(-1, 2).max(0) + MOTION_CLEARANCE
        for k, pts in plans.items():
            if (pts.max(0) < lo).any() or (pts.min(0) > hi).any():
                continue
            if polygons_overlap(P, pts[ConvexHull(pts).vertices], MOTION_CLEARANCE).any():
                self.in_sweep.append(k)
        return self.in_sweep

    def contact(self, face_N, dt_sample):
        """Total lump-face contact normal force at a sample."""
        if face_N > 0.:
            self.all_contact_s += dt_sample
            self.all_contact_max_N = max(self.all_contact_max_N, face_N)
            if self.phase != 'idle':
                self.contact_s += dt_sample
                self.contact_max_N = max(self.contact_max_N, face_N)

    def hold_timed_out(self, t):
        """Retracted for face_hold_max_s without being allowed to close: stop for the operator."""
        return (self.phase == 'hold' and (self.zone_busy or bool(self.in_sweep))
                and t - self.hold_since > self.cfg['face_hold_max_s'])

    # ---- report -------------------------------------------------------------------------------
    def action_summary(self):
        return dict(reached_full_angle=self.face.reached,
                    hinge_moment_over_chord=load_stats(self.face.loads, self.dt, self.cfg['face_force_max']),
                    face_contact_s=round(self.contact_s, 2),
                    face_contact_peak_sample_N=round(self.contact_max_N, 1))

    def report(self):
        if self.events and 'reached_full_angle' not in self.events[-1]:
            self.events[-1].update(self.action_summary())
        moving = self.joint is not None
        face = self.face
        hold = load_stats([-x for x in face.hold], self.dt)
        return dict(
            face_drive_model='dynamic' if self.dynamic else 'kinematic',
            face_actuator=(dict(peak_torque_Nm=face.torque_peak, peak_mechanical_power_W=face.power_peak,
                                net_work_J=face.work_J, saturation_s=face.saturation_s,
                                note='Finite torque servo and soft joint stops; uncalibrated carrier inertia and gains; not cylinder sizing.')
                           if self.dynamic else None),
            face_hinge_moment_over_chord=(dict(
                note=('applied actuator torque / chord (includes inertia); no independent hard-stop reaction measurement. '
                      if self.dynamic else 'hinge constraint moment / chord: the moment referred to the free end. Not '
                     'the contact force on the face, not a cylinder force; the face is a '
                     'kinematically driven joint with no modelled stop or frame stiffness. '
                     '50 ms averages are model diagnostics, not design loads'),
                hold_avg50ms_max_N=hold['resist_avg50ms_max_N'], hold_peak_step_N=hold['resist_peak_step_N'],
                stroke_out=load_stats(face.by_phase['out'], self.dt, self.cfg['face_force_max']),
                stroke_back=load_stats(face.by_phase['back'], self.dt, self.cfg['face_force_max']))
                if moving and face.hold else None),
            face_contact=dict(contact_s=round(self.all_contact_s, 2),
                              peak_sample_total_normal_N=round(self.all_contact_max_N, 1),
                              note='sum of lump-face contact normal forces, sampled every '
                                   '10 ms; the peak sample is impulsive, not a design load'),
            face_closed_on_empty=self.closed_on_empty,
            face_fault=self.fault,
            face_last_action=(dict(self.action_summary(), phase=self.phase,
                                   final_angle_deg=round(math.degrees(face.theta), 2)) if moving else None))
