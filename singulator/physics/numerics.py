"""Numerical screening of a run: contact penetration, solver iterations, MuJoCo warnings, non-finite states.

These tolerances are screening gates, not material measurements or a convergence certificate.

What the penetration gate screens is the machine: every contact except those of lumps that have left it. A contact
with the floor (it stands in for the bins: none are modelled) or between two lumps that have both left the line
(landed, dropped or taken off: lying on the floor) is recorded apart under 'landed' and does not fail the screen.
Until 2026-10-06 it did: a lump landing on the floor after the separator is a 0.7-1 m fall onto a rigid plane, and
in S6 41 of the 46 batches over the 5 mm limit were over it only for the instant of such a landing, with nothing
of the line behind it (docs/TODO.md). Warnings and non-finite states still fail the run wherever they come from.
"""
import math

import mujoco
import numpy as np

FLOOR = 'floor'      # the geom the separator stands on and every lump that leaves the line falls onto


def _finite(*arrays):
    """Every element of every array is finite. A sum is finite only when every term is: one sum answers the common
    case, np.isfinite the rest."""
    return math.isfinite(sum(x.sum() for x in arrays)) or all(np.isfinite(x).all() for x in arrays)


class Diagnostics:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.dt, self.iterations = model.opt.timestep, model.opt.iterations
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
        # both of the above in one number per geom: a contact is landed when the two of its geoms add up to 2 or
        # more (off the machine: 2; a lump that has left the line: 1, so two of them)
        self.landing = 2 * self.off.astype(np.int8)
        self.landing_list = self.landing.tolist()
        self.landed = dict(peak=0., worst=None, over_s=0.)
        self.data = None

    def set_landed(self, geoms):
        """The geoms of every lump that has left the line (sorted into a bin, dropped, taken off). Contacts
        between two of them, or with the floor, are recorded under 'landed' and not screened."""
        self.gone[:] = False
        self.gone[list(geoms)] = True
        self.landing = 2 * self.off.astype(np.int8) + self.gone
        self.landing_list = self.landing.tolist()

    def _bind(self, data):
        """The arrays of data observe() reads every step (MuJoCo's arrays are views that stay valid)."""
        self.data = data
        self.state = data.qpos, data.qvel, data.qacc
        self.niter, self.warned = data.solver_niter, data.warning.number

    def observe(self, data, t):
        """Called before integration, on the contact/force state that actually advances this step.

        Every step, right after mj_step: each numpy call on these small arrays costs microseconds there, so they
        are read as lists, and the screen below runs in Python. A step with a landed contact or a distance that is
        not finite takes the numpy path (_contacts), which gives the same answer for every step."""
        if data is not self.data:
            self._bind(data)
        self.steps += 1
        it = max(self.niter.tolist())
        self.max_iterations = max(self.max_iterations, it)
        self.at_limit += it >= self.iterations
        # the warning counters as one array: reading them as structs (warnings()) cost more than the step itself
        if any(self.warned.tolist()) or not _finite(*self.state):
            self.failure = dict(t_s=float(t), warnings=self.warnings(data), reason='warning_or_nonfinite_state')
        ncon = data.ncon
        if not ncon:
            return
        contact = data.contact
        dist, g = contact.dist[:ncon].tolist(), contact.geom[:ncon].tolist()
        landing = self.landing_list
        if any(landing[a] + landing[b] >= 2 for a, b in g) or not math.isfinite(sum(dist)):
            self._contacts(contact.dist[:ncon], contact.geom[:ncon], t)
            return
        least = min(dist)                               # the first smallest, as argmin finds it
        self._penetration(max(0., -least), lambda: g[dist.index(least)], t)

    def _contacts(self, distances, g, t):
        landed = self.landing[g].sum(1) >= 2         # = self.off[g].any(1) | self.gone[g].all(1)
        if landed.any():
            self._screen(self.landed, distances, g, landed, t)
            if landed.all():
                return
            machine = ~landed
            distances, g = distances[machine], g[machine]
        i = int(distances.argmin())
        self._penetration(max(0., -float(distances[i])), lambda: g[i], t)

    def _penetration(self, pen, pair, t):
        """The deepest machine contact of the step: pen (m) between the geoms pair() names."""
        if pen > self.peak:
            self.peak = pen
            self.worst = dict(t_s=float(t), geoms=self._names(pair()))
        if t > self.cfg['ramp'] + .2:
            self.peak_steady = max(self.peak_steady, pen)
        self.over_s += self.dt * (pen >= self.cfg['penetration_limit'])

    def _screen(self, rec, distances, g, mask, t):
        d = np.where(mask, distances, np.inf)
        i = int(d.argmin())
        pen = max(0., -float(d[i]))
        if pen > rec['peak']:
            rec['peak'] = pen
            rec['worst'] = dict(t_s=float(t), geoms=self._names(g[i]))
        rec['over_s'] += self.dt * (pen >= self.cfg['penetration_limit'])

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
