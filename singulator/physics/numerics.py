"""Numerical screening of a run: contact penetration, solver iterations, MuJoCo warnings, non-finite states.

These tolerances are screening gates, not material measurements or a convergence certificate.

What the penetration gate screens is the machine: every contact except those of lumps that have left it. A contact
with the floor (it stands in for the bins: none are modelled) or between two lumps that have both left the line
(landed, dropped or taken off: lying on the floor) is recorded apart under 'landed' and does not fail the screen.
Until 2026-10-06 it did: a lump landing on the floor after the separator is a 0.7-1 m fall onto a rigid plane, and
in S6 41 of the 46 batches over the 5 mm limit were over it only for the instant of such a landing, with nothing
of the line behind it (docs/TODO.md). Warnings and non-finite states still fail the run wherever they come from.
"""
import mujoco
import numpy as np

FLOOR = 'floor'      # the geom the separator stands on and every lump that leaves the line falls onto


class Diagnostics:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.peak, self.peak_steady = 0., 0.
        self.worst = None
        self.over_s = 0.
        self.steps = self.at_limit = self.max_iterations = 0
        self.failure = None
        # per geom: off the machine (the floor), and the geoms of lumps that have left the line (set_landed)
        self.off = np.zeros(model.ngeom, dtype=bool)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR)
        if floor >= 0:
            self.off[floor] = True
        self.gone = np.zeros(model.ngeom, dtype=bool)
        self.landed = dict(peak=0., worst=None, over_s=0.)

    def set_landed(self, geoms):
        """The geoms of every lump that has left the line (sorted into a bin, dropped, taken off). Contacts
        between two of them, or with the floor, are recorded under 'landed' and not screened."""
        self.gone[:] = False
        self.gone[list(geoms)] = True

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
            g = data.contact.geom[:data.ncon]
            landed = self.off[g].any(1) | self.gone[g].all(1)
            if landed.any():
                self._screen(self.landed, distances, g, landed, t)
                if landed.all():
                    return
                machine = ~landed
                distances, g = distances[machine], g[machine]
            i = int(np.argmin(distances))
            pen = max(0., -float(distances[i]))
            if pen > self.peak:
                self.peak = pen
                self.worst = dict(t_s=float(t), geoms=self._names(g[i]))
            if t > self.cfg['ramp'] + .2:
                self.peak_steady = max(self.peak_steady, pen)
            self.over_s += m.opt.timestep * (pen >= self.cfg['penetration_limit'])

    def _screen(self, rec, distances, g, mask, t):
        d = np.where(mask, distances, np.inf)
        i = int(np.argmin(d))
        pen = max(0., -float(d[i]))
        if pen > rec['peak']:
            rec['peak'] = pen
            rec['worst'] = dict(t_s=float(t), geoms=self._names(g[i]))
        rec['over_s'] += self.model.opt.timestep * (pen >= self.cfg['penetration_limit'])

    def _names(self, pair):
        m = self.model
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                or '@' + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or str(g))
                for g in pair]

    def warnings(self, data):
        return {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}

    def report(self, data):
        warnings = self.warnings(data)
        lr = self.landed
        return dict(max_penetration_m=self.peak, max_steady_penetration_m=self.peak_steady,
                    worst_contact=self.worst, penetration_limit_m=self.cfg['penetration_limit'],
                    penetration_over_limit_s=self.over_s, warnings=warnings, failure=self.failure,
                    solver_iteration_limit=int(self.model.opt.iterations), max_solver_iterations=self.max_iterations,
                    solver_at_limit_steps=int(self.at_limit), checked_steps=self.steps,
                    ok=bool(not self.failure and not warnings and self.peak < self.cfg['penetration_limit']
                            and not self.at_limit),
                    landed=dict(max_penetration_m=lr['peak'], worst_contact=lr['worst'],
                                penetration_over_limit_s=lr['over_s'],
                                note='contacts with the floor (the bins) or between two lumps that have left the line: '
                                     'recorded, not screened'),
                    convergence_verified=False,
                    note='Numerical screening only. Full startup included. Penetration is screened on the machine: '
                         'the landing of lumps that left it is under landed. Halved-step and contact calibration '
                         'still required.')
