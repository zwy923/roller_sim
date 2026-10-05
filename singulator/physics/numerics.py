"""Numerical screening of a run: contact penetration, solver iterations, MuJoCo warnings, non-finite states.

These tolerances are screening gates, not material measurements or a convergence certificate.
"""
import mujoco
import numpy as np


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
