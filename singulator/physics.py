"""Explicit, uncalibrated physics controls and numerical diagnostics.

These tolerances are screening gates, not material measurements or a convergence certificate.
"""
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

DEFAULTS = dict(solver_iterations=100, solver_tolerance=1e-8,
                penetration_limit=.005, contact_condim=3, torsional_friction=0., rolling_friction=0.,
                feed_drop_height=.006, face_bearing_drag=0.,
                face_drive_model='dynamic', face_kp=30000., face_kv=3000.,
                face_carrier_mass=20., face_position_tol=.002, face_velocity_tol=.01)


def validate(cfg):
    from .lumps import validate_material
    c = dict(DEFAULTS, **cfg)
    validate_material(c)
    for key, value in c.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            raise ValueError('%s must be finite' % key)
    positive = ('dt', 'duration', 'solref', 'dampratio', 'v_belt', 'belt_mass', 'motor_slip',
                'size_min', 'size_max', 'size_long_max', 'belt_force_max',
                'side_belt_force_max', 'face_force_max', 'face_swing_s',
                'jam_window', 'feed_len', 'solver_tolerance', 'penetration_limit',
                'face_kp', 'face_kv', 'face_carrier_mass', 'face_position_tol', 'face_velocity_tol')
    nonnegative = ('friction_belt', 'friction_steel', 'friction_block', 'ramp', 'row_gap',
                   'row_stagger', 'jam_speed', 'torsional_friction', 'rolling_friction',
                   'feed_drop_height', 'face_bearing_drag')
    for key in positive + nonnegative:
        x = c[key]
        if not math.isfinite(x) or (x <= 0 if key in positive else x < 0):
            raise ValueError('%s must be finite and %s' % (key, 'positive' if key in positive else 'nonnegative'))
    if not 0 <= c['gangue_fraction'] <= 1:
        raise ValueError('gangue_fraction must be in [0, 1]')
    if c['size_min'] > c['size_max'] or c['size_long_max'] < c['size_max']:
        raise ValueError('need size_min <= size_max <= size_long_max (otherwise the long-axis cap is false)')
    for key in ('count', 'solver_iterations'):
        if int(c[key]) != c[key] or c[key] < 1:
            raise ValueError('%s must be a positive integer' % key)
    if c['contact_condim'] not in (3, 4, 6):
        raise ValueError('contact_condim must be 3, 4 or 6')
    if c['torsional_friction'] and c['contact_condim'] < 4:
        raise ValueError('torsional friction requires contact_condim >= 4')
    if c['rolling_friction'] and c['contact_condim'] < 6:
        raise ValueError('rolling friction requires contact_condim = 6')
    if c['solref'] < 2 * c['dt']:
        raise ValueError('solref must be >= 2*dt; changing dt must not silently change contact stiffness')
    if c['dt'] > .01 or abs(.01 / c['dt'] - round(.01 / c['dt'])) > 1e-7:
        raise ValueError('dt must divide the 10 ms controller interval exactly')
    if c['face_drive_model'] not in ('dynamic', 'kinematic'):
        raise ValueError('face_drive_model must be dynamic or kinematic')
    if c['unjam'] and not 0 < abs(c['face_swing_deg']) < 90:
        raise ValueError('face_swing_deg must have a nonzero magnitude below 90 degrees')
    return c


def provenance(xml, sensing='vision'):
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.glob('*.py')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return dict(source_sha256=h.hexdigest(),
                model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                calibrated=False, convergence_verified=False,
                limitations=['rigid convex unbreakable lumps; shape and material distributions uncalibrated',
                             ('modelled sensors (perception.py): camera outlines with white placeholder noise and '
                              '50 ms latency, no occlusion by equipment; debounced beams; scanner validity on convex '
                              'lumps' if sensing == 'vision' else
                              'ideal full-outline sensing at 100 Hz; no occlusion or processing latency'),
                             'held belt surfaces; no pulley transfer, belt compliance or thermal trips',
                             'weighing = ideal sum of contact forces on the weigh belt and its skirts (no load-cell '
                             'dynamics, belt tension or vibration); volume = the true hull volume'])


class Diagnostics:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.peak, self.peak_steady = 0., 0.
        self.worst = None
        self.over_s = 0.
        self.steps = self.at_limit = self.max_iterations = 0
        self.failure = None

    def observe(self, data, t):
        """Called before integration, on the contact/force state that actually advances this step."""
        m = self.model
        self.steps += 1
        it = int(np.max(data.solver_niter))
        self.max_iterations = max(self.max_iterations, it)
        self.at_limit += it >= m.opt.iterations
        warnings = self.warnings(data)
        if warnings or not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc)):
            self.failure = dict(t_s=float(t), warnings=warnings, reason='warning_or_nonfinite_state')
        if data.ncon:
            distances = data.contact.dist[:data.ncon]
            i = int(np.argmin(distances))
            pen = max(0., -float(distances[i]))
            if pen > self.peak:
                self.peak = pen
                pair = data.contact[i].geom
                self.worst = dict(t_s=float(t), geoms=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                                  or '@' + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or str(g))
                                  for g in pair])
            if t > self.cfg['ramp'] + .2:
                self.peak_steady = max(self.peak_steady, pen)
            self.over_s += m.opt.timestep * (pen >= self.cfg['penetration_limit'])

    def warnings(self, data):
        return {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}

    def report(self, data):
        warnings = self.warnings(data)
        return dict(max_penetration_m=self.peak, max_steady_penetration_m=self.peak_steady,
                    worst_contact=self.worst, penetration_limit_m=self.cfg['penetration_limit'],
                    penetration_over_limit_s=self.over_s, warnings=warnings, failure=self.failure,
                    solver_iteration_limit=int(self.model.opt.iterations), max_solver_iterations=self.max_iterations,
                    solver_at_limit_steps=int(self.at_limit), checked_steps=self.steps,
                    ok=bool(not self.failure and not warnings and self.peak < self.cfg['penetration_limit']
                            and not self.at_limit),
                    convergence_verified=False,
                    note='Numerical screening only. Full startup included. Halved-step and contact calibration still required.')
