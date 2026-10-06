"""Virtual drives: every conveyor behind a force-limited motor, and the load bookkeeping.

A conveyor here is a plate (or a chain of slats) whose position is held and whose surface velocity is written
before every physics step -- for contact exactly a belt surface. Its motor is one speed factor f (1 = rated
speed): the contact forces of the step load it, and it answers along a stiff droop line up to a force limit
(make_motor, motor_update), or with a finite brake when it is told to stand (brake_update).

    Belt, SlatChain   one conveyor: its joints in the model, its motor, the target its controller sets
    Conveyors         the five of the line; the run sets their targets, calls command() before a step and
                      after_step() after it

No current, heating or protective trip is modelled, and nothing is calibrated.
"""
import math

import numpy as np

from ..machine import feed_belt, plough, station
from ..series import load_stats


def make_motor(force_max, mass, v_ref, slip):
    """A drive reduced to one speed factor f (1 = rated speed) with a force limit and reflected inertia.

    Force is the equivalent tangential force at the reference surface speed v_ref. Below the limit the
    speed droops by at most `slip` (stiff induction-motor-like line); above it the drive decelerates and,
    being non-backdrivable, stops at f = 0 instead of reversing.
    """
    return dict(F_max=float(force_max), M_eff=float(mass) * v_ref, v_ref=v_ref,
                gain=float(force_max) / float(slip), f=0., stall_s=0., min_ratio=1., loads=[],
                brake_overload_s=0., contact_power_peak_W=0., contact_work_J=0.)


def motor_update(m, f_target, load, dt):
    """Advance the speed factor one step. load = contact force on the drive at v_ref (negative resists)."""
    a, f, gain, F_max = dt / m['M_eff'], m['f'], m['gain'], m['F_max']
    # implicit in the proportional term, so a very stiff (or effectively unlimited) drive stays stable
    f_new = (f + a * (gain * f_target + load)) / (1 + a * gain)
    force = gain * (f_target - f_new)
    if abs(force) > F_max:
        f_new = f + a * (math.copysign(F_max, force) + load)
    m['f'] = max(0., f_new)


def brake_update(m, load, dt):
    """Finite Coulomb brake, using the drive force limit as an explicit uncalibrated brake assumption.

    Integrate inertia before applying the brake impulse. Hold exactly at rest only within capacity;
    an overload can drag the belt in either direction. Never teleport speed to zero.
    """
    free = m['f'] + dt*load/m['M_eff']
    impulse = dt*m['F_max']/m['M_eff']
    m['f'] = math.copysign(max(0., abs(free)-impulse), free)
    m['brake_overload_s'] += dt*(abs(load) > m['F_max'])


def record_load(m, load, dt):
    m['loads'].append(load)
    power = load*m['v_ref']*m['f']
    m['contact_power_peak_W'] = max(m['contact_power_peak_W'], abs(power))
    m['contact_work_J'] += power*dt


class Belt:
    """A plate belt held at its place, its surface moving at speed * f.

    target      the speed factor its controller asks for (1 = rated speed); set before after_step()
    brakes      at target 0 a finite brake holds it (True), or the motor alone runs it down (False)
    rated_only  what counts as being held back. True: only while it is meant to run at rated speed -- a belt
                being stopped, jogged or run slow on purpose is not stalled. False: whenever it is meant to run,
                relative to its target (the feed belt creeps and stages on purpose)
    pulleys     [(dof, radius)] of drums that turn with the belt: their surface runs at the belt speed and their
                contact torque loads this drive. None: a belt that never has any
    """

    def __init__(self, name, motor, speed, q, dof, brakes, rated_only, pulleys=None):
        self.name, self.motor, self.speed, self.q, self.dof = name, motor, speed, q, dof
        self.brakes, self.rated_only, self.pulleys = brakes, rated_only, pulleys
        self.target = 0.

    f = property(lambda self: self.motor['f'], doc='speed factor now (1 = rated speed)')

    def command(self, data):
        """Hold the plate and write its surface velocity for the coming step."""
        qvel, v = data.qvel, self.speed * self.motor['f']
        data.qpos[self.q] = 0.
        qvel[self.dof] = v
        for dof, R in self.pulleys or ():
            qvel[dof] = v / R

    def load(self, data):
        """Contact force of the last step on the drive, at the belt surface (negative resists), N."""
        return self._load(data.qfrc_constraint)

    def _load(self, qfrc):
        """load() from qfrc_constraint, as the array or as a list."""
        F = float(qfrc[self.dof])
        if self.pulleys is None:
            return F
        return F + sum((float(qfrc[dof]) / R for dof, R in self.pulleys), 0.)

    def after_step(self, data, counted, dt):
        """Let the motor answer the load of the step. counted: the start-up is over (stalls are counted)."""
        self._after_step(self._load(data.qfrc_constraint), counted, dt)

    def _after_step(self, load, counted, dt):
        m, target = self.motor, self.target
        record_load(m, load, dt)
        if self.brakes and target <= 0.:
            brake_update(m, load, dt)
        else:
            motor_update(m, target, load, dt)
        if not counted:
            return
        if self.rated_only:
            if target >= 1.:
                m['min_ratio'] = min(m['min_ratio'], m['f'])
                m['stall_s'] += dt * (m['f'] < .5)
        elif target > 0.:
            m['min_ratio'] = min(m['min_ratio'], m['f'] / target)
            if m['f'] < .5 * target:
                m['stall_s'] += dt


class SlatChain(Belt):
    """The side belt: slats on slide joints, carried round their chain. q and dof are arrays (one per slat);
    with no slats (--no-side-belt) the drive still exists and runs unloaded."""

    def __init__(self, name, motor, speed, q, dof, x0, pitch, length):
        super().__init__(name, motor, speed, q, dof, brakes=False, rated_only=True)
        self.x0, self.length = x0, length
        self.base = np.arange(len(q)) * pitch
        self.travel = 0.
        # the slats' joints follow one another in the model: slices address them without a gather
        run = lambda a: slice(int(a[0]), int(a[-1]) + 1) if len(a) and (np.diff(a) == 1).all() else a
        self.q_run, self.dof_run = run(q), run(dof)
        self._at = np.empty(len(q))

    def command(self, data):
        if len(self.q):
            at = self._at
            np.add(self.base, self.travel, out=at)
            np.mod(at, self.length, out=at)
            at += self.x0
            data.qpos[self.q_run] = at
            data.qvel[self.dof_run] = self.speed * self.motor['f']

    def _load(self, qfrc):
        return float(qfrc[self.dof_run].sum()) if len(self.q) else 0.

    def after_step(self, data, counted, dt):
        self._after_step(self._load(data.qfrc_constraint), counted, dt)

    def _after_step(self, load, counted, dt):
        self.travel += self.speed * self.motor['f'] * dt
        super()._after_step(load, counted, dt)


class Conveyors:
    """The driven conveyors of the line, each behind its own motor:

        main, side          the main belt and the side belt. Their target is the start-up ramp times the station's
                            hold of the section upstream of the buffer belt (1 = run);
        feed                the feed belt; the feeder control sets its target every step;
        buffer, measure     the station's two belts; the station sets their targets.
    """

    def __init__(self, cfg, d, model):
        import mujoco
        jid = lambda joint: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        q, dof = (lambda joint: model.jnt_qposadr[jid(joint)]), (lambda joint: model.jnt_dofadr[jid(joint)])
        plate = lambda body: (q(body + 'j'), dof(body + 'j'))       # a plate belt or a pulley `name` rides on `name`j
        self.cfg = cfg
        mass, slip = cfg['belt_mass'], cfg['motor_slip']
        fd, st, n = d['feeder'], d['station'], d['n_sbslat']
        hd = fd.get('head') or {}
        drum = [(plate(feed_belt.DRUM)[1], hd['drum_d_m'] / 2)] if hd.get('drum_d_m') else []
        tail = [(plate(plough.TAIL)[1], hd['tail_d_m'] / 2)] if hd.get('tail_d_m') else []
        self.main = Belt('belt', make_motor(cfg['belt_force_max'], mass, cfg['v_belt'], slip), cfg['v_belt'],
                         *plate(plough.MAIN_BELT), brakes=False, rated_only=True, pulleys=tail)
        self.side = SlatChain('side_belt', make_motor(cfg['side_belt_force_max'], mass / 4, max(d['v_side'], 1e-6), slip),
                              d['v_side'], np.array([q('vj%d' % i) for i in range(n)], dtype=int),
                              np.array([dof('vj%d' % i) for i in range(n)], dtype=int),
                              d['sb_x0'], plough.SIDE_BELT['pitch'], d['sb_chain'])
        self.feed = Belt('feeder', make_motor(cfg['feeder_force_max'], mass / 2, fd['speed_m_s'], slip), fd['speed_m_s'],
                         *plate(feed_belt.BODY), brakes=True, rated_only=False, pulleys=drum)
        self.buffer = Belt('buffer_belt', make_motor(station.FORCE_MAX, mass / 4, st['buffer_speed_m_s'], slip),
                           st['buffer_speed_m_s'], *plate(station.BUFFER), brakes=True, rated_only=True)
        self.measure = Belt('measure_belt', make_motor(station.FORCE_MAX, mass / 4, st['speed_m_s'], slip),
                            st['speed_m_s'], *plate(station.MEASURE), brakes=True, rated_only=True)
        self.all = (self.main, self.side, self.feed, self.buffer, self.measure)

    def targets(self, section, feed, buffer, measure):
        """The speed factors the controllers ask for this step. section: the main and side belts."""
        self.main.target = self.side.target = section
        self.feed.target, self.buffer.target, self.measure.target = feed, buffer, measure

    def command(self, data):
        """Write the surface velocities (and the slat positions along their chain) for the coming step."""
        for b in self.all:
            b.command(data)

    def after_step(self, data, t, dt):
        """Read the loads the step put on each drive and let the motors respond."""
        counted, qfrc = t > self.cfg['ramp'] + .2, data.qfrc_constraint
        listed = qfrc.tolist()          # the plates read single entries: from a list, not one numpy call each
        for b in self.all:
            b._after_step(b._load(qfrc if b is self.side else listed), counted, dt)

    def factors(self):
        return [b.f for b in self.all]

    def names(self):
        return [b.name for b in self.all]

    def report(self, dt, face_drive):
        """Load summary per drive (see load_stats). No current, heating or protective trip is modelled:
        time below half speed says the simulated drive was held back, not when a real one would trip."""
        drives = ((('belt', self.main.motor), ('side_belt', self.side.motor), ('face_actuator', face_drive))
                  + tuple((b.name, b.motor) for b in self.all[2:]))
        return {nm: dict(limit_N=mo['F_max'], **load_stats(mo['loads'], dt, mo['F_max']),
                         load_includes_startup=True,
                         **({} if nm == 'face_actuator' else dict(
                             brake_overload_s=round(mo['brake_overload_s'], 5),
                             contact_power_peak_W=round(mo['contact_power_peak_W'], 2),
                             contact_work_J=round(mo['contact_work_J'], 2))),
                         min_speed_ratio=round(mo['min_ratio'], 3),
                         below_half_speed_s=round(mo['stall_s'], 3),
                         final_speed_factor=round(mo['f'], 3))
                for nm, mo in drives}
