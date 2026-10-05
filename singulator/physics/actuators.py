"""The two position-controlled mechanisms of the line in the MuJoCo model: the plough face's hinge and the
separator plate's cylinder. What decides their motion is control/face.py and control/station.py.
"""
import math

import numpy as np

from ..machine import plough, separator
from .drives import make_motor


class FaceServo:
    """The face's hinge in the MuJoCo model: a finite-torque position servo. MuJoCo alone advances the hinge
    position and velocity.

    The reference moves at the configured swing rate but stays close to the measured angle, so a stalled
    actuator cannot accumulate a whole stroke of position error. Joint limits are soft stops.

    What control.face.FaceRetract uses of it (the servo interface): parts, piv (plan polygons of the moving parts
    at rest and the hinge point), lever, delta, swing_s; theta and actual_omega (measured); reached;
    command(phase) before a physics step and record(t, phase, ramp) after it; sweep(); drive, loads, hold,
    by_phase and the actuator totals for the report.
    """

    def __init__(self, cfg, d, model, data):
        j = model.joint(plough.FACE_JOINT).id
        self.cfg, self.data, self.dt = cfg, data, model.opt.timestep
        self.joint = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])      # qpos adr, dof adr of the hinge
        self.actuator = model.actuator(plough.FACE_ACTUATOR).id                 # its position actuator
        self.parts, self.piv = plough.face_parts(cfg, d)
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

    def command(self, phase):
        """Write the servo reference for the coming step."""
        data = self.data
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

    def record(self, t, phase, ramp):
        """After the step: the actuator torque while moving (drive load) or while standing."""
        data = self.data
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


class PlateDrive:
    """The separator plate in the model: the cylinder's position servo follows the plate angle the station asks
    for (through the linkage's kinematics), and the plate hinge is the encoder."""

    def __init__(self, model, sep):
        """sep: the separator's dimensions (machine.separator.Config, d['station']['sep'])."""
        self.model, self.sep = model, sep
        deg = np.linspace(sep.closed_deg, sep.open_deg, 221)
        self.ext = (deg, np.array([separator.state(sep, a)['piston_extension_m'] for a in deg]))
        self.q = model.jnt_qposadr[model.joint(separator.PREFIX + 'plate_hinge').id]
        self.actuator = model.actuator(separator.PREFIX + 'cylinder_position').id

    def place(self, data):
        """Plate down (coal), linkage closed: the pose at t = 0."""
        separator.place(self.model, data, self.sep, self.sep.closed_deg)

    def command(self, data, deg):
        """Ask for the plate angle `deg` (the station's ramped reference) for the coming step."""
        data.ctrl[self.actuator] = float(np.interp(deg, *self.ext))

    def angle(self, data):
        """The plate encoder, deg."""
        return math.degrees(float(data.qpos[self.q]))
