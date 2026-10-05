"""Virtual drives and load bookkeeping."""
import math

import numpy as np


def make_motor(limited, force_max, mass, v_ref, slip):
    """A drive reduced to one speed factor f (1 = rated speed) with a force limit and reflected inertia.

    Force is the equivalent tangential force at the reference surface speed v_ref. Below the limit the
    speed droops by at most `slip` (stiff induction-motor-like line); above it the drive decelerates and,
    being non-backdrivable, stops at f = 0 instead of reversing. limited=False reproduces the old
    kinematic drive (f follows the ramp exactly, unlimited force).
    """
    return dict(limited=bool(limited), F_max=float(force_max), M_eff=float(mass) * v_ref, v_ref=v_ref,
                gain=float(force_max) / float(slip), f=0., stall_s=0., min_ratio=1., loads=[],
                brake_overload_s=0., contact_power_peak_W=0., contact_work_J=0.)


def motor_update(m, f_target, load, dt):
    """Advance the speed factor one step. load = contact force on the drive at v_ref (negative resists)."""
    if not m['limited']:
        m['f'] = f_target
        return
    a = dt / m['M_eff']
    # implicit in the proportional term, so a very stiff (or effectively unlimited) drive stays stable
    f_new = (m['f'] + a * (m['gain'] * f_target + load)) / (1 + a * m['gain'])
    force = m['gain'] * (f_target - f_new)
    if abs(force) > m['F_max']:
        f_new = m['f'] + a * (math.copysign(m['F_max'], force) + load)
    m['f'] = max(0., f_new)


def brake_update(m, load, dt):
    """Finite Coulomb brake, using the drive force limit as an explicit uncalibrated brake assumption.

    Integrate inertia before applying the brake impulse. Hold exactly at rest only within capacity;
    an overload can drag the belt in either direction. Never teleport speed to zero.
    """
    if not m['limited']:
        m['f'] = 0.
        return
    free = m['f'] + dt*load/m['M_eff']
    impulse = dt*m['F_max']/m['M_eff']
    m['f'] = math.copysign(max(0., abs(free)-impulse), free)
    m['brake_overload_s'] += dt*(abs(load) > m['F_max'])


def record_load(m, load, dt):
    m['loads'].append(load)
    power = load*m['v_ref']*m['f']
    m['contact_power_peak_W'] = max(m['contact_power_peak_W'], abs(power))
    m['contact_work_J'] += power*dt


def load_stats(loads, dt, limit=None):
    """Load summary from a signed per-step series F (negative = resisting the motion).

    net_*      50 ms moving average of F itself: resistance and assistance cancel, i.e. the net effect.
    resist_*   computed on max(-F, 0) BEFORE averaging, and assist_* on max(F, 0), so a load that
               alternates in sign shows up in both instead of averaging away (a +/-2000 N alternation
               has zero net average but a 1000 N average resistance and assistance).
    *_peak_step_N     largest single-step value -- a contact impulse, not a load a part holds.
    *_over_10pct_limit_s  time the one-sided load exceeds 10 % of the drive limit (any value > 0 without one).
    The 50 ms averages are model diagnostics, not design loads for a drive or a cylinder.
    """
    a = np.asarray(loads, float)
    keys = ('net_avg50ms_resist_max_N', 'net_avg50ms_assist_max_N', 'resist_avg50ms_max_N', 'resist_peak_step_N',
            'resist_over_10pct_limit_s', 'assist_avg50ms_max_N', 'assist_peak_step_N', 'assist_over_10pct_limit_s')
    if not len(a):
        return dict.fromkeys(keys, 0.)
    w = max(1, int(round(.05 / dt)))

    def avg(x):
        c = np.concatenate([[0.], np.cumsum(x)])
        return (c[w:] - c[:-w]) / w if len(x) >= w else np.array([x.mean()])
    thr = .1 * limit if limit else 0.
    net, r, s = avg(a), np.maximum(-a, 0.), np.maximum(a, 0.)
    return dict(zip(keys, (round(float(max(0., -net.min())), 1), round(float(max(0., net.max())), 1),
                           round(float(avg(r).max()), 1), round(float(r.max()), 1), round(float((r > thr).sum() * dt), 3),
                           round(float(avg(s).max()), 1), round(float(s.max()), 1), round(float((s > thr).sum() * dt), 3))))


class Conveyors:
    """The driven conveyors, each behind its own force-limited virtual motor: the main belt (a held plate
    with a prescribed surface velocity), the side belt (a slat chain), the feed belt (a held plate; its target
    speed factor, feed_target, is set every step by the feeder control instead of following the start-up
    ramp), and the buffer and measuring belts (held plates whose targets, buffer_target and measure_target,
    the station sets). `upstream` (the station's hold of the section upstream of the buffer belt, 1 = run)
    scales the main belt and side belt targets."""

    def __init__(self, cfg, d, model):
        import mujoco
        from .machine import SIDE_BELT
        J = mujoco.mjtObj.mjOBJ_JOINT
        jid = lambda s: mujoco.mj_name2id(model, J, s)
        self.cfg, self.d = cfg, d
        self.main_q, self.main_dof = model.jnt_qposadr[jid('mfloorj')], model.jnt_dofadr[jid('mfloorj')]
        self.side_q = np.array([model.jnt_qposadr[jid('vj%d' % i)] for i in range(d['n_sbslat'])], dtype=int)
        self.side_dofs = np.array([model.jnt_dofadr[jid('vj%d' % i)] for i in range(d['n_sbslat'])], dtype=int)
        lim, mass, slip = cfg['drive_limit'], cfg['belt_mass'], cfg['motor_slip']
        self.main = make_motor(lim, cfg['belt_force_max'], mass, cfg['v_belt'], slip)
        self.side = make_motor(lim, cfg['side_belt_force_max'], mass / 4, max(d['v_side'], 1e-6), slip)
        self.side_base = np.arange(d['n_sbslat']) * SIDE_BELT['pitch']
        self.side_travel = 0.
        self.pulleys = []                          # (dof, radius, 'feed' / 'main'): head drum, tail pulley below
        self.feed_q, self.feed_dof = model.jnt_qposadr[jid('feederj')], model.jnt_dofadr[jid('feederj')]
        self.feed = make_motor(lim, cfg['feeder_force_max'], mass / 2, d['feeder']['speed_m_s'], slip)
        self.feed_target = 0.
        hd = d['feeder'].get('head') or {}
        for j, dia, who in (('fdrumj', hd.get('drum_d_m'), 'feed'), ('mtailj', hd.get('tail_d_m'), 'main')):
            if dia:
                self.pulleys.append((model.jnt_dofadr[jid(j)], dia / 2, who))
        self.upstream = 1.
        self.station = []                          # (name, motor, qpos adr, dof adr, target attribute)
        from .station import FORCE_MAX
        for name, body, attr, v in (('buffer_belt', 'bbeltj', 'buffer_target', d['station']['buffer_speed_m_s']),
                                    ('measure_belt', 'mbeltj', 'measure_target', d['station']['speed_m_s'])):
            self.station.append((name, make_motor(lim, FORCE_MAX, mass / 4, v, slip),
                                 model.jnt_qposadr[jid(body)], model.jnt_dofadr[jid(body)], attr))
            setattr(self, attr, 0.)

    def command(self, data):
        """Write the surface velocities (and the slat positions along their chains) for the coming step."""
        d = self.d
        data.qpos[self.main_q] = 0.
        data.qvel[self.main_dof] = self.cfg['v_belt'] * self.main['f']
        if d['n_sbslat']:
            data.qpos[self.side_q] = d['sb_x0'] + np.mod(self.side_base + self.side_travel, d['sb_chain'])
            data.qvel[self.side_dofs] = d['v_side'] * self.side['f']
        data.qpos[self.feed_q] = 0.
        data.qvel[self.feed_dof] = d['feeder']['speed_m_s'] * self.feed['f']
        for dof, R, who in self.pulleys:            # surface speed = its belt's speed
            data.qvel[dof] = (d['feeder']['speed_m_s'] * self.feed['f'] if who == 'feed'
                              else self.cfg['v_belt'] * self.main['f']) / R
        for _, mo, q, dof, _ in self.station:
            data.qpos[q] = 0.
            data.qvel[dof] = mo['v_ref'] * mo['f']

    def after_step(self, data, t, f_target, dt):
        """Read the loads the step put on each drive and let the motors respond."""
        d, cfg = self.d, self.cfg
        self.side_travel += d['v_side'] * self.side['f'] * dt
        load = float(data.qfrc_constraint[self.main_dof])
        pulley_load = dict(feed=0., main=0.)
        for dof, R, who in self.pulleys:            # a pulley's contact torque as a force at its belt surface
            pulley_load[who] += float(data.qfrc_constraint[dof]) / R
        load += pulley_load['main']
        s_load = float(data.qfrc_constraint[self.side_dofs].sum()) if d['n_sbslat'] else 0.
        for mo, force in ((self.main, load), (self.side, s_load)):
            record_load(mo, force, dt)
        up = f_target * self.upstream               # the station holds the section upstream of the buffer belt
        motor_update(self.main, up, load, dt)
        motor_update(self.side, up, s_load, dt)
        for _, mo, _, dof, attr in self.station:
            target, st_load = getattr(self, attr), float(data.qfrc_constraint[dof])
            record_load(mo, st_load, dt)
            if target <= 0.:
                brake_update(mo, st_load, dt)
            else:
                motor_update(mo, target, st_load, dt)
            if t > cfg['ramp'] + .2 and target >= 1.:     # held back while running at speed (stopped is not a stall)
                mo['min_ratio'] = min(mo['min_ratio'], mo['f'])
                mo['stall_s'] += dt * (mo['f'] < .5)
        feed_load = float(data.qfrc_constraint[self.feed_dof]) + pulley_load['feed']
        record_load(self.feed, feed_load, dt)
        if self.feed_target <= 0.:
            brake_update(self.feed, feed_load, dt)
        else:
            motor_update(self.feed, self.feed_target, feed_load, dt)
        if t > cfg['ramp'] + .2:
            if self.feed_target > 0.:              # held back while it is meant to run (stopped is not a stall)
                self.feed['min_ratio'] = min(self.feed['min_ratio'], self.feed['f'] / self.feed_target)
                if self.feed['f'] < .5 * self.feed_target:
                    self.feed['stall_s'] += dt
        if t > cfg['ramp'] + .2:
            for mo in (self.main, self.side):
                if self.upstream < 1.:
                    continue                           # held or jogged by the station: not a stall
                mo['min_ratio'] = min(mo['min_ratio'], mo['f'] / max(f_target, 1e-9))
                if mo['f'] < .5 * f_target:
                    mo['stall_s'] += dt

    def factors(self):
        return [self.main['f'], self.side['f'], self.feed['f']] + [mo['f'] for _, mo, _, _, _ in self.station]

    def names(self):
        return ['belt', 'side_belt', 'feeder'] + [nm for nm, _, _, _, _ in self.station]

    def report(self, dt, face_drive):
        """Load summary per drive (see load_stats). No current, heating or protective trip is modelled:
        time below half speed says the simulated drive was held back, not when a real one would trip."""
        drives = ((('belt', self.main), ('side_belt', self.side), ('face_actuator', face_drive), ('feeder', self.feed))
                  + tuple((nm, mo) for nm, mo, _, _, _ in self.station))
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
