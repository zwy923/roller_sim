"""Retract action of the hinged plough face, and the bookkeeping of what it met.

FaceRetract is the phase machine. It drives a face servo -- physics.actuators.FaceServo in a run, an ideal one in
the checks -- through that class's interface, and reads only the servo's measured angle and speed.

Phases: idle -> out (swing to face_swing_deg) -> hold (retracted) -> back (closing) -> idle, or fault.
  * Closing starts only when the section is clear AND no lump reaches into the area the face will sweep
    on its way home (the whole lump outline).
  * While closing the same sweep check runs every sample; a lump entering it pauses the face (back to
    hold) until it is clear again.
  * Retracted for face_hold_max_s without being allowed to close: the batch stops for the operator with
    the face open. It never closes onto material.
  * Only a face that actually reaches home goes idle. If it has not got home 6 x its swing time after
    closing started, that is a reset fault: the batch stops and no further action is taken.

Torque divided by chord is not cylinder force or total contact force. Whether the face touches material is
measured separately, from the lump-face contact pairs (contact()).
"""
import math

from scipy.spatial import ConvexHull

from ..geom2d import overlap_many
from ..machine import plough
from ..series import load_stats


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

    def command(self, t):
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
        F.command(self.phase)

    def record(self, t):
        """After the step: the actuator torque on the face."""
        self.face.record(t, self.phase, self.cfg['ramp'])

    # ---- observation --------------------------------------------------------------------------
    def check_sweep(self, plans):
        """plans: {lump id: plan-view points}. Records which lumps reach into the face's closing sweep
        (within plough.MOTION_CLEARANCE); checked only while the face is out of its home position."""
        self.in_sweep = []
        if self.phase not in ('hold', 'back') or not len(self.face.parts):
            return self.in_sweep
        P = self.face.sweep()
        clear = plough.MOTION_CLEARANCE
        lo, hi = P.reshape(-1, 2).min(0) - clear, P.reshape(-1, 2).max(0) + clear
        for k, pts in plans.items():
            if (pts.max(0) < lo).any() or (pts.min(0) > hi).any():
                continue
            if overlap_many(P, pts[ConvexHull(pts).vertices], clear).any():
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
