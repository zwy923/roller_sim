"""Retract action of the hinged plough face, and the bookkeeping of what it met.

The face is a finite-torque physical hinge with a rate-limited position servo and soft joint limits (FaceServo).
Carrier mass/inertia and controller gains are assumptions. Torque divided by chord is not cylinder force or total
contact force. Whether the face touches material is measured separately, from the lump-face contact pairs.

FaceRetract is the phase machine; it drives any servo with FaceServo's interface (the checks use an ideal one).
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

from .drives import load_stats, make_motor
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


class FaceServo:
    """The face's hinge in the MuJoCo model: a finite-torque position servo. MuJoCo alone advances the hinge
    position and velocity.

    The reference moves at the configured swing rate but stays close to the measured angle, so a stalled
    actuator cannot accumulate a whole stroke of position error. Joint limits are soft stops.

    What FaceRetract uses of it (the servo interface): parts, piv (plan polygons of the moving parts at rest and
    the hinge point), lever, delta, swing_s; theta and actual_omega (measured); reached; command(phase, data)
    before a physics step and record(t, phase, data, ramp) after it; sweep(); drive, loads, hold, by_phase and
    the actuator totals for the report.
    """

    def __init__(self, cfg, d, dt, joint, actuator):
        """joint = (qpos adr, dof adr) of the face hinge; actuator = its position actuator's id."""
        self.cfg, self.dt, self.joint, self.actuator = cfg, dt, joint, actuator
        self.parts, self.piv = face_parts(cfg, d)
        # moment arm: the chord between the hinge and the far end -- NOT the polyline length, which for a
        # curved face is longer than the straight distance the tip actually travels
        self.lever = float(np.linalg.norm(d['P1'] - d['P0']))
        self.delta, self.swing_s = math.radians(cfg['face_swing_deg']), cfg['face_swing_s']
        v_tip = abs(self.delta) * self.lever / self.swing_s
        self.drive = make_motor(cfg['face_force_max'], 50., max(v_tip, 1e-6), cfg['motor_slip'])
        self.theta = self.actual_omega = 0.
        self.reached = False
        self.loads, self.hold, self.by_phase = [], [], dict(out=[], back=[])
        self.reference = 0.
        self.last_phase = 'idle'
        self.saturation_s = 0.
        self.torque_peak = self.power_peak = 0.
        self.work_J = 0.

    def command(self, phase, data):
        """Write the servo reference for the coming step."""
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
        """After the step: the actuator torque while moving (drive load) or while standing."""
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

    def sweep(self):
        """Plan-view polygons of the part at angles from the present one back to home (<= 1 deg apart),
        i.e. everything it will pass through while closing."""
        n = max(2, int(math.ceil(abs(self.theta) / math.radians(1.))) + 1)
        out = []
        for th in np.linspace(self.theta, 0., n):
            c, s = math.cos(th), math.sin(th)
            out.append(self.piv + (self.parts - self.piv) @ np.array([[c, s], [-s, c]]))
        return np.concatenate(out) if out else self.parts

    def report(self):
        return dict(peak_torque_Nm=self.torque_peak, peak_mechanical_power_W=self.power_peak,
                    net_work_J=self.work_J, saturation_s=self.saturation_s,
                    note='Finite torque servo and soft joint stops; uncalibrated carrier inertia and gains; not cylinder sizing.')


class FaceRetract:
    """The retract phase machine on a face servo (FaceServo, or anything with its interface)."""

    def __init__(self, cfg, servo):
        self.cfg, self.face, self.dt = cfg, servo, servo.dt
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

    def _settled_at(self, angle):
        """At `angle` and at rest, by the measured position and speed."""
        F, c = self.face, self.cfg
        return abs(F.theta - angle) <= c['face_position_tol'] and abs(F.actual_omega) <= c['face_velocity_tol']

    @property
    def at_home(self):
        """A phase label alone is insufficient: load can deflect the physical actuator."""
        return self.phase == 'idle' and self._settled_at(0.)

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
        """Advance the phase machine and write the servo reference for the coming step."""
        c, F = self.cfg, self.face
        if self.phase == 'out':
            reached = self._settled_at(F.delta)
            if reached or t - self.t0 > 6 * F.swing_s:
                F.reached = reached
                self.phase, self.t0, self.hold_since = 'hold', t, t
                if not reached:
                    self.fault = dict(t_s=round(t, 3), reason='retract_not_reached',
                                      angle_deg=math.degrees(F.theta))
                    self.phase = 'fault'
        elif self.phase == 'hold' and t - self.t0 >= c['face_hold_s'] and not self.zone_busy and not self.in_sweep:
            self.phase, self.t0, self.closed_on_empty = 'back', t, True
        elif self.phase == 'back' and self.in_sweep:
            # a lump has come into the path while closing: stop pushing, wait retracted again
            self.phase, self.t0 = 'hold', t
            self.events[-1]['pauses'] += 1
        if self.phase == 'back':
            if self._settled_at(0.):
                self.phase = 'idle'
            elif t - self.t0 > 6 * F.swing_s:
                self.fault = dict(t_s=round(t, 2), reason='reset_not_reached', angle_deg=round(math.degrees(F.theta), 2))
                self.phase = 'fault'
        F.command(self.phase, data)

    def record(self, t, data):
        """After the step: the actuator torque on the face."""
        self.face.record(t, self.phase, data, self.cfg['ramp'])

    # ---- observation --------------------------------------------------------------------------
    def check_sweep(self, plans):
        """plans: {lump id: plan-view points}. Records which lumps reach into the face's closing sweep
        (within MOTION_CLEARANCE); checked only while the face is out of its home position."""
        self.in_sweep = []
        if self.phase not in ('hold', 'back') or not len(self.face.parts):
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
        face = self.face
        hold = load_stats([-x for x in face.hold], self.dt)
        return dict(
            face_drive_model='dynamic',
            face_actuator=face.report(),
            face_hinge_moment_over_chord=(dict(
                note='applied actuator torque / chord (includes inertia); no independent hard-stop reaction '
                     'measurement. ',
                hold_avg50ms_max_N=hold['resist_avg50ms_max_N'], hold_peak_step_N=hold['resist_peak_step_N'],
                stroke_out=load_stats(face.by_phase['out'], self.dt, self.cfg['face_force_max']),
                stroke_back=load_stats(face.by_phase['back'], self.dt, self.cfg['face_force_max']))
                if face.hold else None),
            face_contact=dict(contact_s=round(self.all_contact_s, 2),
                              peak_sample_total_normal_N=round(self.all_contact_max_N, 1),
                              note='sum of lump-face contact normal forces, sampled every '
                                   '10 ms; the peak sample is impulsive, not a design load'),
            face_closed_on_empty=self.closed_on_empty,
            face_fault=self.fault,
            face_last_action=dict(self.action_summary(), phase=self.phase,
                                  final_angle_deg=round(math.degrees(face.theta), 2)))
