"""The experimental side gate's control (hardware: machine/side_gate.py). It is the upper bound of any feeding algorithm
on this hardware, not a controller for the real machine.

It reads the true lumps (the feeder's ideal view, --feeder-sensing oracle) and decides whenever the feed belt runs
forward -- in a release and while staging, before the next row overhangs the plate. When queued lumps lie side by side
with the lead (centroids within PAIR_DX), it brakes, inserts the plate from the side away from the lump that is to go
and holds the others; the release is the feeder's own (staging, funnel check, centroid stop), led by the lumps not
held (Feeder.hold_back). Once the free lump's centroid is over the edge the plate retracts -- only partway when the
next to go is one of the held lumps and the rest still lie on the plate's side: the plate is already in the plane,
and lumps pressed against it would block a new insertion.

    startup -> idle -> brake -> deploy -> hold (-> shift -> hold ...) -> stop -> retract -> settle -> idle

A controller that read the cameras instead (planned from contours, metered the free lump blind until the drop beam)
acted at most once a batch -- after the first release the staged row already overhung the plate -- and was removed on
2026-10-10; its result is in designs/side_pusher/RESULTS.md.
"""
from . import feeder
from ..machine import side_gate as hardware

STARTUP_S = .50         # s: the feeder waits this long for the gate's first look
SETTLE_S = .30          # s: after a stop, and after the plate is home again
MARGIN = .025           # m: the plate's tip stays this far off the free lump
PAIR_DX = .10           # m: a queued lump whose centroid is within this of the lead's would go with it (S8: level
                        # centroids; side_gate_300: every two-lump release was side by side, 0-9.6 cm apart)
REACH = .20             # m: brake for the plate only once the held fronts are this close to it ...
PLAN_MARGIN = .02       # m: ... and still this far short of it (the feed belt runs on a little while braking)
MOVE_S = 4.             # s: a brake, insertion, shift or retraction taking longer is aborted (or a fault)
HOLD_TIMEOUT_S = 60.    # s: hold at most this long while the feed belt is free to run


class SideGate:
    def __init__(self, cfg, d):
        self.cfg, self.fd, self.g = cfg, d['feeder'], hardware.geometry(cfg, d)
        self.phase, self.t0, self.last_t = 'startup', 0., 0.
        self.targets = [0., 0.]
        self.actions, self.events, self.reasons = [], [], {}
        self.blocking = True        # the gate has the feed belt: the feeder is paused
        self.fault = self.current = None
        self.held_s = self.out_s = self.hold_s = 0.
        self.next_decision = 0.

    def _phase(self, t, p):
        self.phase, self.t0 = p, t
        self.events.append(dict(t_s=round(t, 3), phase=p))

    def plan(self, view, margin=0., side=None, within=None):
        """Hold every queued lump whose centroid is within PAIR_DX of the lead's, from the side away from the one that
        goes, so that the feeder's own release takes it alone. Nothing may lie in the plate's swept volume except
        beyond the free lump's side. side, within: the plate is in on that side at that stroke; only a shorter stroke
        on it is a plan (retracting partway sweeps nothing)."""
        x1 = self.fd['x1']
        queued = [o for o in view.values() if o['cx'] <= x1 and o['x1'] > self.fd['x0']]
        if len(queued) < 2:
            return None, 'no_pair'
        lead = max(o['cx'] for o in queued)
        row = [o for o in queued if o['cx'] > lead - PAIR_DX]
        if len(row) < 2:
            return None, 'no_pair'
        plane = self.g['x_m'] - self.g['thickness_m'] / 2 - .012
        beyond = self.g['x_m'] + self.g['thickness_m'] / 2 + self.g['buffer_stroke_m']
        best, why, rank = None, 'neighbours_on_both_sides_or_out_of_stroke', None
        # the free lump: any of the row with every other on the plate's side; the lead first, then the shorter stroke
        for free in row:
            held = [o for o in row if o is not free]
            for i, s in enumerate(self.g['sides']):
                tip = free['y0'] - MARGIN if i == 0 else free['y1'] + MARGIN
                stroke = (tip - s['tip_y']) * s['sign']
                if not .08 < stroke <= self.g['stroke_m'] or not all(
                        (o['cy'] < tip - .04) if i == 0 else (o['cy'] > tip + .04) for o in held):
                    continue
                if within is not None:
                    if i != side or stroke >= within - .01:
                        continue
                elif any(o is not free and ((o['y0'] < tip) if i == 0 else (o['y1'] > tip)) and o['x1'] > plane - margin
                         and o['x0'] < beyond for o in view.values()):
                    why = 'insertion_plane_occupied'
                    continue
                elif max(o['x1'] for o in held) < plane - REACH:
                    why = 'row_far_from_gate'
                    continue
                r = (free['cx'] < lead, stroke)
                if rank is None or r < rank:
                    rank = r
                    best = dict(side=i, stroke_m=stroke, plan='hold_true_neighbours' + ('_shift' if within else ''),
                                free=free['tid'], held=[o['tid'] for o in held])
        return (best, 'planned') if best else (None, why)

    def observe(self, t, view, fb, belt_speed, inhibit=False, beam=False):
        """One 10 ms sample. view: the true lumps; fb: the drive's feedback (GateDrive.feedback); belt_speed: the feed
        belt's, m/s; inhibit: no new decision now (the station holds, or the feed belt is not running forward); beam:
        the drop beam is cut."""
        dt, self.last_t = t - self.last_t, t
        self.held_s += dt * self.blocking
        home = all(abs(p) < .004 and abs(v) < .02 for p, v in zip(fb['position_m'], fb['speed_m_s']))
        self.out_s += dt * (not home)
        overload = max(fb['load_N']) > self.g['load_trip_N'] or max(fb['buffer_m']) > self.g['buffer_stroke_m'] * .95
        if self.phase == 'startup' and t >= STARTUP_S:
            self._phase(t, 'idle')
            self.blocking = False
        if self.phase == 'idle':
            if self.cfg['side_pusher'] != 'active' or inhibit or t < self.next_decision \
                    or len(self.actions) >= self.cfg['pusher_max_actions']:
                return
            self.next_decision = t + .1
            plan, why = self.plan(view, PLAN_MARGIN)
            self.reasons[why] = self.reasons.get(why, 0) + 1
            if plan:
                self.current = plan
                self.blocking = True
                self._phase(t, 'brake')
        elif self.phase == 'brake':
            if abs(belt_speed) < .002 and t - self.t0 >= SETTLE_S:
                plan, why = self.plan(view)
                if plan is None or inhibit or beam:
                    self.blocking = False
                    self.next_decision = t + .5
                    self._phase(t, 'idle')
                    return
                self.current = dict(plan, t_start_s=round(t, 3))
                self.actions.append(self.current)
                self.targets[plan['side']] = plan['stroke_m']
                self._phase(t, 'deploy')
            elif t - self.t0 > MOVE_S:
                self.fault = dict(t_s=round(t, 3), reason='gate_brake_timeout')
                self._phase(t, 'fault')
        elif self.phase == 'deploy':
            i = self.current['side']
            reached = abs(fb['position_m'][i] - self.targets[i]) < .004 and abs(fb['speed_m_s'][i]) < .03
            if reached and not inhibit:
                self.blocking, self.hold_s = False, 0.      # the feeder runs on its own; the plate holds the others
                self._phase(t, 'hold')
            elif t - self.t0 > MOVE_S or overload or inhibit:
                self.current['push_end'] = 'deployment_aborted'
                self._phase(t, 'stop')
        elif self.phase == 'hold':
            self.hold_s += dt * (not inhibit)
            free, x1 = view.get(self.current['free']), self.fd['x1']
            went = free is None or free['cx'] > x1 + feeder.STOP_PAST
            held_went = any(k in view and view[k]['cx'] > x1 for k in self.current['held'])
            if went or held_went or overload or self.hold_s > HOLD_TIMEOUT_S:
                self.current['push_end'] = ('held_went' if held_went else 'went' if went else
                                            'load_trip' if overload else 'hold_timeout')
                self.current['hold_s'] = round(self.hold_s, 3)
                self.blocking = True
                i = self.current['side']
                nxt = (self.plan(view, side=i, within=self.targets[i])[0] if self.current['push_end'] == 'went'
                       and len(self.actions) < self.cfg['pusher_max_actions'] else None)
                if nxt:
                    self.current['t_end_s'] = round(t, 3)
                    self.current = dict(nxt, t_start_s=round(t, 3))
                    self.actions.append(self.current)
                    self.targets[i] = nxt['stroke_m']
                    self._phase(t, 'shift')
                else:
                    self._phase(t, 'stop')
        elif self.phase == 'shift':
            i = self.current['side']
            if abs(fb['position_m'][i] - self.targets[i]) < .004 and abs(fb['speed_m_s'][i]) < .03:
                self.blocking, self.hold_s = False, 0.
                self._phase(t, 'hold')
            elif t - self.t0 > MOVE_S or overload:
                self.current['push_end'] = 'shift_aborted'
                self._phase(t, 'stop')
        elif self.phase == 'stop':
            if abs(belt_speed) < .002 and t - self.t0 >= SETTLE_S:
                self.targets = [0., 0.]
                self._phase(t, 'retract')
            elif t - self.t0 > MOVE_S:
                self.fault = dict(t_s=round(t, 3), reason='gate_stop_timeout')
                self._phase(t, 'fault')
        elif self.phase == 'retract':
            if home:
                self._phase(t, 'settle')
            elif t - self.t0 > MOVE_S:
                self.fault = dict(t_s=round(t, 3), reason='gate_retract_timeout')
                self._phase(t, 'fault')
        elif self.phase == 'settle' and t - self.t0 >= SETTLE_S:
            self.current['t_end_s'] = round(t, 3)
            self.blocking = False
            self.next_decision = t + .1
            self._phase(t, 'idle')

    def report(self):
        return dict(mode=self.cfg['side_pusher'], geometry=self.g, actions=self.actions, events=self.events,
                    decision_reasons=self.reasons, final_phase=self.phase, fault=self.fault,
                    held_s=round(self.held_s, 3), plate_out_s=round(self.out_s, 3),
                    sensing='true lumps: an upper bound, not a sensor',
                    limitations='uncalibrated materials, friction, servo and mount load cell; rigid convex lumps')
