"""Control of the measuring and sorting station behind the lane: buffer belt B, measuring belt M, the separator
plate (third version 2026-09-30). The hardware is machine/station.py. The two measuring devices on M are handed in:
a weigher (sensing/weigher.py) and a volume device (sensing/volume.py); this controller stops an item for them,
asks each for its result and decides on what they return.

    lane -> main belt head edge -> buffer belt (one step down) -> measuring belt (weigh + volume) -> flip separator

History. First version (2026-09-24 morning): weigh belt -> volume belt, the whole section upstream stopped
for every lump. Second version (afternoon): the former weigh belt became a buffer, weighing and the volume
scan share one belt, sensors as hardware, the plate moves as soon as the density is known. Third version
(2026-09-30, user): 优先让无效测量真正停止分选 -- a void measurement no longer only gets a flag and is sorted
anyway; and the controllers read what real sensors deliver (singulator/sensing/), not the true state.
Same day (user): the station became the line itself (the gated mainline was deleted), the step onto B went to
10 cm, the separator plate got side walls, and B got its own speed (buffer_speed) so that it can run faster
than the main belt and pull apart lumps that come over the head edge together.

What it reads (--sensing vision): the camera objects of sensing.vision.Vision (estimated outline, extents,
outline centroid, velocity, confidence, estimated lump count, track lineage), the beams S1 and S3 as debounced
signals, the weigher's result, the volume device's verdict, the belt drives' own speed and the plate angle. It
never reads the true state. The verification of its decisions against the truth is written into its record by a
witness (verify/station.py) that it calls at fixed points; without one (NoWitness) it runs the same.

Items. An object whose outline centroid passes the head edge joins an item: the item of an object it
descends from (track lineage), else the last item on B if it is closer than GROUP_GAP behind it, else a
new one. Two items whose objects run together become one. An item on B whose objects have drawn apart by
GROUP_GAP (B running faster than the main belt pulls lumps that came over the edge together apart) becomes
two items, each keeping only its own objects.

Void = the measurement cannot stand for one lump. Reasons (all recorded, any one is enough):
  multi        the item had two objects at once, or an object the cameras count as two lumps;
  outside      while weighing, the item's outline is not inside the weigh zone with ISO_MARGIN, or another
               object is within ISO_GAP of it (it may lean on something off the scale);
  unsteady     the weigher has a result but refuses it: no steady reading in time (--weigh-model steady only; the
               default since 2026-10-05 is a device that reads scan_s after M came to rest, whatever the reading
               does);
  scan         the volume device's verdict is not usable (failed, not one lump, not wholly in the zone, coverage);
  implausible  measured density outside DENSITY_OK (a part of the weight carried elsewhere reads low);
  tare         the weigher says the scale was not clear before this item (something left on M);
  track_lost   the cameras lost the item's object on the way and found it again (identity not certain).
An item found good waits at rest on M until the plate is in position for it. It is still watched (nothing else
within ISO_GAP, one object of one lump): a lump that reaches it in that time makes it void after all (2026-10-04;
until then nothing looked once the route was known, and a round lump that rolled on over the joint after its stop
at S3 went into the same bin with it, unmeasured).
A void item is HELD: M does not discharge it, the plate does not move for it, the feed belt releases nothing
more, B stages its lead at S3 and stops, and the section upstream stops as soon as B cannot take the next
lump (the second version's occupancy rule). On the prototype someone then separates, cleans and re-measures
by hand (no automatic return). The simulation does not model that (user 2026-09-30: 不需要模拟人工操作): it
takes the held item's lumps off the line at once, counts the stop, and the line runs on (took_off).

Control (as in the second version, now on the sensors):
  section upstream (main belt, side belt, feed belt): runs; stops only when B cannot take the next lump
     (an object's centroid within HOLD_ZONE of the head edge while B is stopped or has less than ENTRY_ROOM
     free behind its last object, or an object across the head edge while B is stopped);
  B  runs, except that it stops when its lead item is at S3 -- the beam blocked, or the cameras see its front
     STAGE_CAM past the line (then S3 is suspect) -- and M cannot take it;
  M  runs while empty, receiving or discharging; stops once an item is wholly on it (estimated rear
     CLEAR + the cameras' position margin past the joint) and the item before has left; at rest it is weighed
     and scanned;
  two optional rules (2026-10-04; EXPERIMENTS.md S5 tried them as a patch, S6 as options; off by default):
     --weigh-stop centre: M carries an item that is wholly on it further, until the item is in the middle of the
        weigh zone (or its front is at the stop position: a long item stops as before). A round lump rocks after
        the stop; stopped with its rear 5-12 cm past the joint it can roll back to the joint ('outside');
     --buffer-approach slow: B runs at M's speed while the front of its lead item is within APPROACH of the
        joint. The lead then meets S3 and the joint at 0.4 m/s whatever M is doing: it stops shorter at S3 (at
        0.8 m/s a large round lump cut the 5 cm high beam late and its nose came to rest over the joint, in the
        scanner's zone: the item being weighed was void), and it is not thrown onto the slower M;
  plate  moves to the route's position the moment the route is known, once everything sent before has left
     its path: the healthy cameras see the plate zone empty, for both coal and gangue. Not cleared within
     PATH_TIMEOUT_S after its discharge: stop the line.
Alarms that stop everything (all belts, the plate stays): vision not healthy (camera outage, a track lost
and not found again) -- the line resumes once it is healthy again, and stops for good after FREEZE_MAX_S;
a beam diagnosed dirty, dead or blocked too long, and a plate-path timeout -- the run stops there (no
recovery modelled).

Nothing here is calibrated: the sorting threshold, the density window and every tolerance are placeholders.
"""
import math

from ..config import SAMPLE_S
from ..geom2d import box_gap
from ..machine import station as hw
from ..sensing import vision

STAGE_CAM = .03       # m: the cameras see the lead's front this far past S3 while S3 is clear: stop B anyway
ACCEPT_AFTER = .30    # m: M takes the next item while discharging once the item on it has its rear this far on M
APPROACH = .45        # m: --buffer-approach slow: B runs at M's speed while its lead's front is this close to the joint
                      # (S3 is 0.22 m before the joint at 0.8 m/s, and B needs 0.09 m to come down to 0.4 m/s)
GROUP_GAP = .10       # m: a lump going over the head edge this close behind the last one joins its item
HOLD_ZONE = .10       # m: a lump whose centroid is this close to the head edge is about to go over it
ENTRY_ROOM = .55      # m: B takes a lump only with this much free belt behind its last lump
PLATE_MARGIN_S = .30  # s: the plate is in position this long before the lump's front reaches M's head edge
HANG_S = .50          # s: a lump across the head edge without progress this long gets a creep jog of the section
JOG = .075            # jog speed / belt speed: 0.03 m/s at 0.40, the feed belt's creep
PLATE_TOL_DEG = .5    # plate in position within this
PLATE_FAULT = 3.      # plate not in position after this many swing times: fault, stop
STALL_S = 20.         # s: a lump in hand and no station state change this long: stop the run
# ---- void rules (placeholders, see the module docstring) ----
ISO_MARGIN = .02      # m: while weighing the item's outline keeps this far inside the weigh zone (x) ...
ISO_GAP = .05         # m: ... and no other object comes this close to it
DENSITY_OK = (1000., 3200.)   # kg/m3: a density outside this is not a lump weighed whole
ZERO_CLEAR = .10      # m: the weigher tracks its zero only with nothing this close to the measuring belt
PATH_TIMEOUT_S = 6.   # s: a sent item must be out of the plate's path this long after its discharge started
FREEZE_MAX_S = 5.     # s: the vision not healthy this long (not found again): stop the run
DEAD_AFTER = 2       # releases (S1) in a row without the beam: the beam is dead
ACROSS_IN = .10      # m: the cameras see a lump's body across a beam when the line is this far inside its outline


class NoWitness:
    """Where the verification hooks go when nobody verifies. verify.station.StationWitness is the one that does;
    it documents each hook."""

    def seen(self, objs):
        pass

    def landed(self, t, sent):
        pass

    def at_rest(self, it):
        pass

    def decided(self, it):
        pass

    def routed(self, it):
        pass

    def revoked(self, it):
        pass


class Station:
    """B, M and the plate. This sets the targets of their drives from what the sensors report: step() returns
    the belt speed factors and moves plate_ref, the plate angle asked for (physics.drives.Conveyors and
    physics.actuators.PlateDrive follow them in a run)."""

    def __init__(self, cfg, d, weigher, scanner, margin=0.):
        """weigher, scanner: the two measuring devices on M (the interfaces of sensing.weigher.Weigher and
        sensing.volume.Scanner); margin: the position margin on the edges the cameras judge, m (the sensors'
        own figure: sensing.suite.Sensors.margin; 0 on an ideal view)."""
        self.cfg, self.d, self.g = cfg, d, d['station']
        self.weigher, self.scanner, self.vm = weigher, scanner, margin
        self.witness = NoWitness()                  # verify.station.StationWitness.attach() puts itself here
        c = self.sep = self.g['sep']
        self.rate = (c.open_deg - c.closed_deg) / cfg['separator_swing_s']      # plate, deg/s
        self.v, self.vb = cfg['station_speed'], cfg['buffer_speed']     # M, B
        self.centre = cfg['weigh_stop'] == 'centre'
        self.slow = cfg['buffer_approach'] == 'slow'
        # drive factors (1 = rated speed) and their goals; u is the whole section upstream of B
        self.u, self.b, self.m = 1., 0., 0.
        self.u_goal, self.b_goal, self.m_goal = 1., 1., 1.
        self.plate = 'closed'
        self.plate_ref = self.plate_goal = self.plate_deg = c.closed_deg
        self.plate_t0 = 0.
        self.items, self.line, self.sent = [], [], []
        self.jog, self.fault, self.alarm = None, None, None
        self.hist = {}                              # (t, centroid x) per object, HANG_S long
        self.counts = dict(holds=0, section_holds=0, late=0, jogs=0, buffer_stops=0, plate_moves=0, discharge_early=0,
                           s3_by_camera=0, freezes=0, splits=0)
        self.events, self.held_s, self.buffer_stopped_s, self.frozen_s = [], 0., 0., 0.
        self.key, self.since = None, 0.
        self.frozen, self.frozen_since = False, None
        self.objs = {}

    def _stage(self, *stages):
        return [it for it in self.line if it['stage'] in stages]

    @property
    def path_pending(self):
        """A discharged item still needs the plate-zone camera's clearance confirmation."""
        return bool(self._stage('discharge')) or any('t_clear_s' not in it for it in self.sent)

    @property
    def weighing(self):
        """An item is at rest on M (being measured, or held): the weigher is fed every physics step."""
        return bool(self._stage('measure', 'hold'))

    @property
    def section_held(self):
        """The section upstream of B is stopped (or jogging) for the station."""
        return self.u_goal < 1.

    @property
    def inhibit_release(self):
        """The feed belt must not let another lump go: a void item is held, or an alarm stops the line."""
        return bool(self._stage('hold')) or self.frozen

    # ---- per physics step ------------------------------------------------------------------------
    def step(self, dt):
        """One physics step: ramp the drive factors toward their goals and move the plate reference. Returns
        (u, b, m): the speed factors of the section upstream, of B and of M."""
        r = dt / hw.RAMP_S
        ramp = lambda x, goal: min(goal, x + r) if goal > x else max(goal, x - r)
        self.u, self.b, self.m = ramp(self.u, self.u_goal), ramp(self.b, self.b_goal), ramp(self.m, self.m_goal)
        # float(np.clip(...)) on scalars, as min(max(...)): the same value without numpy's overhead every step
        self.plate_ref += float(min(max(self.plate_goal - self.plate_ref, -self.rate * dt), self.rate * dt))
        return self.u, self.b, self.m

    # ---- every camera sample ---------------------------------------------------------------------
    def observe(self, t, view, beams, plate_deg):
        """One 10 ms sample. view: the cameras' frame {tid: object} (health in view.health); beams: {name:
        sensing.beams.Beam}; plate_deg: the plate encoder. The weigher and the volume device have had their sample
        (the run feeds them)."""
        g = self.g
        self.plate_deg = plate_deg
        self.held_s += SAMPLE_S * self.section_held
        self.buffer_stopped_s += SAMPLE_S * (self.b_goal == 0.)
        self.frozen_s += SAMPLE_S * self.frozen
        x_e, x_j, m_end = g['buffer']['x0'], g['buffer']['x1'], g['measure']['x1']
        self.objs = objs = {tid: o for tid, o in view.items() if o['cx'] > x_e and o['x0'] < g['plate_zone'][1][0]}
        self.witness.seen(objs)
        for tid, o in view.items():
            h = self.hist.setdefault(tid, [])
            h.append((t, o['cx']))
            while len(h) > 2 and h[1][0] < t - HANG_S - .02:
                h.pop(0)

        # ---- objects -> items --------------------------------------------------------------------
        self._assign(t, objs)
        for it in self.line + [e for e in self.sent if 't_clear_s' not in e]:
            it['objs'] = [objs[tid] for tid in it['now']]
            if it['objs']:
                it['box'] = (min(o['x0'] for o in it['objs']), max(o['x1'] for o in it['objs']),
                             min(o['y0'] for o in it['objs']), max(o['y1'] for o in it['objs']))
                it['n_obj_max'] = max(it['n_obj_max'], len(it['objs']))
                it['n_est_max'] = max(it['n_est_max'], max(o['n_est'] for o in it['objs']))
        self._split(t)

        # ---- stage changes on B and M ------------------------------------------------------------
        on_m = lambda it: it['stage'] in ('transfer', 'onm', 'measure', 'decided', 'discharge', 'hold')
        for i, it in enumerate(self.line):
            if 'box' not in it:
                continue
            x0, x1 = it['box'][:2]
            if not it['objs'] and it['stage'] == 'discharge':
                x0 = m_end + 1.                                   # gone over the head edge and out of view
            if it['stage'] == 'buffer' and x1 > x_j:
                it.update(stage='transfer', t_transfer_s=round(t, 3))
            if it['stage'] == 'transfer' and x0 > x_j + hw.CLEAR + self.vm:
                it['stage'] = 'onm'
            first = not any(on_m(e) for e in self.line[:i])
            at_front = x1 >= m_end - hw.FRONT_MARGIN - self.v * (hw.RAMP_S / 2 + SAMPLE_S)
            # --weigh-stop centre: an item wholly on M, M free for it, rides on to the middle of the weigh zone
            riding = self.centre and it['stage'] == 'onm' and first
            centred = (x0 + x1) / 2 >= (x_j + m_end) / 2 - self.v * (hw.RAMP_S / 2 + SAMPLE_S)
            if it['stage'] in ('transfer', 'onm') and self.m_goal == 1. and at_front and not riding:
                # M is carrying it and its front is at the stop position before the item is wholly on (it is longer
                # than the belt, or its lumps have drawn apart; seed 4014): stop now -- nothing leaves M unmeasured --
                # and hold it
                self._void(it, 'outside', detail='front at the measuring belt head before the item was on it')
                it.update(stage='measure', t_stop_s=round(t, 3), tare_N=self.weigher.begin())
            if it['stage'] == 'onm' and first and (not self.centre or centred or at_front):
                it.update(stage='measure', t_stop_s=round(t, 3), tare_N=self.weigher.begin())
            if it['stage'] == 'discharge' and x0 > m_end + hw.CLEAR:
                it.update(stage='sent', t_off_s=round(t, 3))
        for it in [e for e in self.line if e['stage'] == 'sent']:
            self.line.remove(it)
            self.sent.append(it)

        # ---- the plate's path, and what has landed ------------------------------------------------
        path_clear = self._path(t, view)
        # ---- M: tare while empty, weigh and scan at rest -----------------------------------------------
        if not self.frozen and not any(on_m(e) for e in self.line) \
                and not any(o['x1'] > x_j - ZERO_CLEAR and o['x0'] < m_end + ZERO_CLEAR for o in objs.values()):
            self.weigher.track_zero()                         # M seen empty
        for it in self._stage('measure'):
            self._measure(t, it)
        for it in self._stage('decided'):                     # found good, waiting for the plate: still watched
            self._isolation(it, zone=False)
            if it['reasons']:
                it.update(void=True, revoked=dict(t_s=round(t, 3), route=it.pop('route')))
                self.witness.revoked(it)
                self._hold(t, it)
        # ---- plate: move the moment the route is known and the path is clear ---------------------------
        self._plate(t, path_clear)
        # ---- M: discharge as soon as the plate will be in position in time -----------------------------
        for it in self._stage('decided'):
            if self.frozen:
                break
            want = 'open' if it['route'] == 'gangue' else 'closed'
            t_front = max(0., m_end - it['box'][1]) / self.v + hw.RAMP_S / 2
            moving = self.plate == ('opening' if want == 'open' else 'closing')
            remaining = abs(self.plate_goal - self.plate_deg) / self.rate
            if self.plate == want or (moving and remaining + PLATE_MARGIN_S <= t_front):
                it.update(stage='discharge', t_discharge_s=round(t, 3), plate_at_start=self.plate)
                self.counts['discharge_early'] += self.plate != want
            break                                             # one at a time, in order

        # ---- alarms, belt goals ------------------------------------------------------------------------
        self._alarms(t, view, beams)
        self.m_goal = 0. if (self._stage('measure', 'decided', 'hold') or self.frozen) else 1.
        busy = [e for e in self.line if on_m(e)]
        accepting = all(e['stage'] == 'discharge' and e['box'][0] > x_j + ACCEPT_AFTER for e in busy)
        lead = next((e for e in self.line if e['stage'] == 'buffer'), None)
        staged = False
        if lead is not None and 'box' in lead:
            by_cam = lead['box'][1] > g['beam_stop']['x'] + STAGE_CAM
            staged = beams['beam_stop'].blocked or by_cam
            if by_cam and not beams['beam_stop'].blocked and not lead.get('s3_by_camera'):
                lead['s3_by_camera'] = round(t, 3)
                self.counts['s3_by_camera'] += 1
                self._event(t, 's3_by_camera', item=lead['item'])
        if self.frozen:
            b_goal = 0.
        elif self._stage('transfer'):
            b_goal = self.m_goal * self.v / self.vb           # an item crossing moves only with M, at its speed
        else:
            b_goal = 0. if staged and not accepting else 1.
            if b_goal and self.slow and lead is not None and 'box' in lead and lead['box'][1] > x_j - APPROACH:
                b_goal = self.v / self.vb                     # --buffer-approach slow: the last stretch at M's speed
        if b_goal == 0. and self.b_goal > 0.:
            self.counts['buffer_stops'] += 1
            self._event(t, 'buffer_stop', item=None if lead is None else lead['item'])
        self.b_goal = b_goal
        self._section(t, view)
        self._watch(t)

    # ---- items -------------------------------------------------------------------------------------
    def _new_item(self, t):
        it = dict(item=len(self.items), tids=set(), now=[], objs=[], joined=[], t_in_s=round(t, 3), stage='buffer',
                  reasons=[], n_obj_max=0, n_est_max=0)
        self.items.append(it)
        self.line.append(it)
        return it

    def _void(self, it, why, **info):
        if why not in it['reasons']:
            it['reasons'].append(why)
            if info:
                it.setdefault('reason_info', {})[why] = info

    def _assign(self, t, objs):
        """Every object past the head edge to an item: by track lineage (items still on the line, or sent
        and not yet out of the plate's path), else as a new arrival on B."""
        g = self.g
        live = self.line + [e for e in self.sent if 't_clear_s' not in e]
        for it in live:
            it['now'] = []
        new = []
        for tid in sorted(objs, key=lambda k: -objs[k]['x1']):
            o = objs[tid]
            its = [it for it in live if it['tids'] & o['lineage']]
            line_its = [it for it in its if it in self.line]
            if len(line_its) > 1:                             # two items ran together: one item now
                keep = line_its[0]
                for it in line_its[1:]:
                    keep['tids'] |= it['tids']
                    keep['joined'].append('item %d' % it['item'])
                    self.line.remove(it)
                    live.remove(it)
                    it['stage'] = 'merged_into_%d' % keep['item']
                    self._void(keep, 'multi', detail='items ran together', absorbed=it['item'])
                    self._event(t, 'items_merged', item=keep['item'], absorbed=it['item'])
                its = [keep]
            if its:
                its[0]['tids'].add(tid)
                its[0]['now'].append(tid)
            elif o['cx'] < g['measure']['x1']:
                new.append(tid)                               # past M's head edge unclaimed: nothing to do
        for tid in new:
            o = objs[tid]
            orphan = [it for it in self.line if not it['now'] and 'box' in it
                      and it['box'][0] - .3 < o['cx'] < it['box'][1] + .3]
            last = self.line[-1] if self.line else None
            if orphan:                                        # found again after the cameras lost it
                it = orphan[0]
                self._void(it, 'track_lost')
            elif last is not None and last['stage'] == 'buffer' and 'box' in last \
                    and o['x1'] >= last['box'][0] - GROUP_GAP and o['cx'] < g['buffer']['x1']:
                it = last
                it['joined'].append(tid)
            else:
                it = self._new_item(t)
                if o['cx'] > g['buffer']['x0'] + .5:
                    self._void(it, 'track_lost', detail='appeared on the station, not over the head edge')
                if self.section_held:
                    self.counts['late'] += 1                  # went over while the section was held
                self._event(t, 'went', item=it['item'], late=self.section_held)
            it['tids'].add(tid)
            it['now'].append(tid)

    def _split(self, t):
        """An item on B whose objects have drawn apart by GROUP_GAP (in x) becomes two: the lead objects keep
        the item, the ones behind become a new item right after it. Each keeps only its own objects' track
        ids (a blob's id stays with neither, or its halves would pull the two back together) and its object
        counts restart from what the cameras see now."""
        for it in [e for e in self.line if e['stage'] == 'buffer' and len(e['objs']) > 1]:
            objs = sorted(it['objs'], key=lambda o: o['x0'])
            reach, cut = objs[0]['x1'], None
            for i, o in enumerate(objs[1:], 1):
                if o['x0'] - reach >= GROUP_GAP:
                    cut = i                                       # the last clear gap, from behind
                reach = max(reach, o['x1'])
            if cut is None:
                continue
            behind, ahead = objs[:cut], objs[cut:]
            new = dict(item=len(self.items), tids={o['tid'] for o in behind}, now=[o['tid'] for o in behind],
                       objs=behind, joined=[], t_in_s=it['t_in_s'], stage='buffer', reasons=[],
                       n_obj_max=len(behind), n_est_max=max(o['n_est'] for o in behind),
                       split_from=it['item'])
            self.items.append(new)
            self.line.insert(self.line.index(it) + 1, new)
            it.update(tids={o['tid'] for o in ahead}, now=[o['tid'] for o in ahead], objs=ahead,
                      n_obj_max=len(ahead), n_est_max=max(o['n_est'] for o in ahead))
            it.setdefault('split', []).append(new['item'])
            for e in (it, new):
                e['box'] = (min(o['x0'] for o in e['objs']), max(o['x1'] for o in e['objs']),
                            min(o['y0'] for o in e['objs']), max(o['y1'] for o in e['objs']))
            self.counts['splits'] += 1
            self._event(t, 'item_split', item=it['item'], new=new['item'])

    # ---- the plate's path ------------------------------------------------------------------------
    def _in_zone(self, view):
        (x0, y0, z0), (x1, y1, _) = self.g['plate_zone']
        return [o for o in view.values() if o['x1'] > x0 and o['x0'] < x1 and o['y1'] > y0 and o['y0'] < y1
                and o['ztop'] > z0]

    def _path(self, t, view):
        """The healthy cameras see nothing of the item or of anything unknown in the plate zone.
        Objects of items sent later do not count against an earlier one. Not out within PATH_TIMEOUT_S
        of its discharge: stop the line."""
        zone = self._in_zone(view)
        order = {it['item']: i for i, it in enumerate(self.sent)}
        owner = lambda o: next((it for it in self.sent if it['tids'] & o['lineage']), None)
        for it in self.sent:
            if 't_clear_s' in it:
                continue
            mine = [o for o in zone if owner(o) is None or order[owner(o)['item']] <= order[it['item']]]
            occupied = bool(mine) or not view.health['ok']
            if not occupied:
                it['t_clear_s'] = round(t, 3)
                it['path_confirmed_by'] = 'plate-zone camera'
            elif t - it['t_discharge_s'] > PATH_TIMEOUT_S and self.fault is None:
                self.fault = dict(t_s=round(t, 2), reason='separator_path_timeout', item=it['item'],
                                  route=it['route'], zone_occupied=occupied,
                                  note='the plate-zone camera did not confirm the path clear')
        self.witness.landed(t, self.sent)
        return all('t_clear_s' in it for it in self.sent)

    # ---- weighing and scanning -------------------------------------------------------------------
    def _isolation(self, it, zone=True):
        """An item at rest on M, every sample: inside the weigh zone (zone: while it is weighed -- once its
        measurement stands, a lump that rocks back towards the joint is still the lump that was measured), nothing
        else close, one object of one lump."""
        g, box = self.g, it.get('box')
        x_j, m_end = g['buffer']['x1'], g['measure']['x1']
        if zone and box is not None and (box[0] < x_j + ISO_MARGIN or box[1] > m_end - ISO_MARGIN):
            self._void(it, 'outside', detail='outline %.3f..%.3f m, weigh zone %.3f..%.3f m'
                       % (box[0], box[1], x_j + ISO_MARGIN, m_end - ISO_MARGIN))
        mine = set(it['now'])
        if it['objs'] and any(min(box_gap(o, m) for m in it['objs']) < ISO_GAP
                              for tid, o in self.objs.items() if tid not in mine):
            self._void(it, 'outside', detail='another object within %.2f m' % ISO_GAP)
        if it['n_obj_max'] > 1 or it['n_est_max'] > 1:
            self._void(it, 'multi', detail='objects %d, estimated lumps %d' % (it['n_obj_max'], it['n_est_max']))

    def _measure(self, t, it):
        """An item at rest on M: the void checks every sample, and the two devices asked for their results until
        each has given one."""
        rest = it['t_stop_s'] + hw.RAMP_S                 # M at rest from here
        if t < rest - 1e-9:
            return
        self._isolation(it)
        if not self.weigher.zero_ok(it['tare_N']):
            self._void(it, 'tare', detail='empty reading %.1f N' % it['tare_N'])
        self.witness.at_rest(it)
        if 'mass_kg' not in it:
            got = self.weigher.read(t, rest, it['tare_N'])
            if got is not None:
                refused = got.pop('void', None)
                it.update(got)
                if refused:
                    self._void(it, refused)
        if 'scan' not in it:
            res = self.scanner.read(t, rest)
            if res is not None:
                it['scan'] = res
                if not res['valid']:
                    self._void(it, 'scan', detail=','.join(res['reasons']))
                else:
                    it['volume_m3'] = res['volume_m3']
        if 'mass_kg' in it and 'scan' in it:
            self._decide(t, it)

    def _decide(self, t, it):
        thr = self.cfg['sort_density']
        vol = it.get('volume_m3')
        rho = it['mass_kg'] / vol if vol else float('nan')
        it.update(t_decided_s=round(t, 3), density_kg_m3=round(rho, 1) if math.isfinite(rho) else None)
        self.witness.decided(it)
        if math.isfinite(rho) and not DENSITY_OK[0] <= rho <= DENSITY_OK[1]:
            self._void(it, 'implausible', detail='%.0f kg/m3' % rho)
        it['void'] = bool(it['reasons'])
        if it['void']:
            self._hold(t, it)
            return
        it.update(stage='decided', route='gangue' if rho >= thr else 'coal')
        self.witness.routed(it)
        self._event(t, 'decided', item=it['item'], mass_kg=it['mass_kg'], volume_m3=it['volume_m3'],
                    density_kg_m3=it['density_kg_m3'], route=it['route'])

    def _hold(self, t, it):
        it.update(stage='hold', t_hold_s=round(t, 3))
        self.counts['holds'] += 1
        self._event(t, 'hold', item=it['item'], reasons=list(it['reasons']))

    # ---- items taken off the line by hand ---------------------------------------------------------------
    def took_off(self, t, items, lumps):
        """Someone has taken these items off the line -- the held ones, and with them whatever else lay on M
        (the simulation's stand-in for that is sim.scenarios.TakeOff, which also says which items those are). The
        stop is over; M was cleaned, so the scale is clear again. lumps: what was removed, for the record."""
        for it in items:
            self.line.remove(it)
            it.update(stage='taken_off', t_taken_off_s=round(t, 3), taken_off=list(lumps))
        self.weigher.clear()
        self._event(t, 'taken_off', lumps=list(lumps))

    # ---- alarms ------------------------------------------------------------------------------------
    def _alarms(self, t, view, beams):
        """Beam diagnoses stop the run; the vision not healthy freezes the line until it is again (at most
        FREEZE_MAX_S, then the run stops)."""
        d, g = self.d, self.g
        fd = d['feeder']
        belt = dict(beam_feed=self.u, beam_stop=self.b)
        x_on = dict(beam_feed=fd['x1'], beam_stop=g['buffer']['x0'])
        for name, b in beams.items():
            near = [o for o in view.values() if o['x0'] - .05 < b.x < o['x1'] + .05]
            # a lump's body across the line (not a raised nose or tail: ACROSS_IN inside its outline both ways),
            # on the belt under the beam, seen with confidence
            across = [o for o in near if o['x0'] + ACROSS_IN < b.x < o['x1'] - ACROSS_IN and o['cx'] > x_on[name]
                      and o['conf'] >= vision.CONF_OK]
            seen_clear = view.health['ok'] and not near
            expected = name == 'beam_stop' or belt[name] < .5    # staged, or the belt under it stopped
            # 'dead' by the cameras only for S3: a lump overhanging the edge above S1 can look like one that went
            # over; S1 is judged dead by repeated releases it did not see
            a = b.diagnose(t, seen_clear, bool(across) and name == 'beam_stop', expected)
            if a is not None and self.alarm is None:
                self.alarm = dict(a)
                self._event(t, 'beam_alarm', **{k: v for k, v in a.items() if k != 't_s'})
                if self.fault is None:
                    self.fault = dict(t_s=round(t, 2), reason='beam_%s' % a['why'], beam=name, note=a['note'])
        frozen = self.alarm is not None or not view.health['ok']
        if frozen and not self.frozen:
            self.counts['freezes'] += 1
            self.frozen_since = t
            self._event(t, 'freeze', why=view.health.get('why') if not view.health['ok'] else self.alarm['why'])
        elif not frozen and self.frozen:
            self._event(t, 'unfreeze')
        elif frozen and t - self.frozen_since > FREEZE_MAX_S and self.fault is None:
            self.fault = dict(t_s=round(t, 2), reason='vision_lost', why=view.health.get('why'),
                              at=view.health.get('at'), note='not found again within %.0f s' % FREEZE_MAX_S)
        self.frozen = frozen

    def _no_progress(self, tid, t):
        h = self.hist.get(tid, [])
        if not h or h[0][0] > t - HANG_S + 1e-9:
            return False
        past = next(x for tt, x in reversed(h) if tt <= t - HANG_S + 1e-9)
        return h[-1][1] - past < .02 * HANG_S

    def _section(self, t, view):
        """Hold the section upstream only when B cannot take the next lump; jog a lump hanging on the edge;
        everything stops while the line is frozen."""
        x_e = self.g['buffer']['x0']
        if self.frozen:
            self.u_goal, self.jog = 0., None
            return
        on_b = [o for it in self.line if it['stage'] in ('buffer', 'transfer') for o in it['objs']]
        if self.jog is not None:
            j = self.jog
            if j not in view or view[j]['x0'] > x_e + hw.CLEAR or self.b_goal == 0.:
                self.jog = None
                self._event(t, 'jog_done', tid=j)
            else:
                return
        bridging = [o for o in on_b if o['x0'] < x_e]
        room = not on_b or min(o['x0'] for o in on_b) >= x_e + ENTRY_ROOM
        taken = {tid for it in self.line for tid in it['now']}
        approaching = any(tid not in taken and x_e - HOLD_ZONE < o['cx'] <= x_e for tid, o in view.items())
        hold = (approaching and (self.b_goal == 0. or not room)) or (bridging and self.b_goal == 0.)
        hanging = [o['tid'] for o in bridging if o['x1'] > x_e and self._no_progress(o['tid'], t)] \
            if self.b_goal > 0. else []
        if hanging and not hold:
            self.jog, self.u_goal = hanging[0], JOG
            self.counts['jogs'] += 1
            self._event(t, 'jog', tid=hanging[0])
            return
        if hold and self.u_goal == 1.:
            self.counts['section_holds'] += 1
            self._event(t, 'section_hold', why='buffer stopped' if self.b_goal == 0. else 'no room on the buffer')
        elif not hold and self.u_goal < 1.:
            self._event(t, 'resume')
        self.u_goal = 0. if hold else 1.

    def _plate(self, t, path_clear):
        c = self.sep
        if self.plate in ('opening', 'closing'):
            if abs(self.plate_deg - self.plate_goal) <= PLATE_TOL_DEG:
                self.plate = 'open' if self.plate_goal == c.open_deg else 'closed'
                self._event(t, 'plate_' + self.plate, angle_deg=round(self.plate_deg, 2))
            elif t - self.plate_t0 > PLATE_FAULT * self.cfg['separator_swing_s'] and self.fault is None:
                self.fault = dict(t_s=round(t, 2), reason='separator_fault', plate=self.plate,
                                  angle_deg=round(self.plate_deg, 2), goal_deg=self.plate_goal)
                return
        if self._stage('discharge') or not path_clear or self.frozen:
            return                                      # a lump is on its way over the plate: do not move
        nxt = next((e for e in self.line if e['stage'] == 'decided'), None)
        if nxt is not None:
            want = 'open' if nxt['route'] == 'gangue' else 'closed'
        elif self._stage('transfer', 'onm', 'measure', 'hold'):
            return                                      # the next route is about to be known: stay put
        else:
            want = 'closed'                             # rest position
        goal = c.open_deg if want == 'open' else c.closed_deg
        if goal != self.plate_goal or self.plate not in (want, 'opening', 'closing'):
            # at rest in the other position, or on its way there: go (a swing reverses at once)
            self.plate = 'opening' if want == 'open' else 'closing'
            self.plate_goal, self.plate_t0 = goal, t
            self.counts['plate_moves'] += 1
            self._event(t, 'plate_' + self.plate, item=None if nxt is None else nxt['item'])
            if nxt is not None:
                nxt['t_plate_move_s'] = round(t, 3)

    def _watch(self, t):
        """A lump in hand and no state change for STALL_S: stop the run."""
        key = (tuple(it['stage'] for it in self.line), self.plate, len(self.items),
               sum('t_clear_s' in it for it in self.sent), self.jog, self.b_goal, self.u_goal)
        busy = (bool(self.line) or any('t_clear_s' not in it for it in self.sent)) and not self.frozen
        if key != self.key or not busy:
            self.key, self.since = key, t
        elif t - self.since > STALL_S and self.fault is None:
            self.fault = dict(t_s=round(t, 2), reason='station_stall', plate=self.plate,
                              items=[(it['item'], it['stage']) for it in self.line],
                              not_cleared=[it['item'] for it in self.sent if 't_clear_s' not in it])

    def geometry(self):
        """The station for result.json: the hardware (machine.station.report) with what the two measuring devices
        and this controller add to it -- how M weighs and scans, and the void rules."""
        g = hw.report(self.g)
        measure = dict(g['measure'], **self.weigher.describe(), **self.scanner.describe())
        return dict(g, measure=measure,
                    void_rules=dict(density_ok_kg_m3=list(DENSITY_OK), iso_margin_m=ISO_MARGIN, iso_gap_m=ISO_GAP,
                                    path_timeout_s=PATH_TIMEOUT_S, **self.weigher.limits(),
                                    on_void='hold: no discharge, no plate move, no release; the simulation then takes '
                                            'the lumps off the line (manual re-measuring is not modelled)'))

    def _event(self, t, event, **info):
        self.events.append(dict(t_s=round(t, 3), event=event, **info))

    # ---- output ----------------------------------------------------------------------------------
    def report(self):
        """The station's record for result.json. (A witness adds the verification: verify.station.)"""
        done = [it for it in self.items if 't_decided_s' in it]
        valid = [it for it in done if not it['void']]
        clean = lambda it: {k: (sorted(v) if isinstance(v, (set, frozenset)) else v) for k, v in it.items()
                            if k not in ('objs', 'now', 'tids')}
        reasons = {}
        for it in done:
            for r in it['reasons']:
                reasons[r] = reasons.get(r, 0) + 1
        w = self.weigher.describe()
        return dict(geometry=self.geometry(), items=[clean(it) for it in self.items], events=self.events,
                    counts=dict(self.counts, items=len(self.items), measured=len(done), valid=len(valid),
                                void=len(done) - len(valid), void_reasons=reasons,
                                gangue=sum(it.get('route') == 'gangue' for it in valid),
                                coal=sum(it.get('route') == 'coal' for it in valid),
                                void_discharged=sum(it['void'] and 't_discharge_s' in it for it in done)),
                    upstream_held_s=round(self.held_s, 2), buffer_stopped_s=round(self.buffer_stopped_s, 2),
                    frozen_s=round(self.frozen_s, 2), fault=self.fault,
                    final=dict(plate=self.plate, plate_deg=round(self.plate_deg, 2),
                               items=[(it['item'], it['stage']) for it in self.line]),
                    control='void items are held on the measuring belt (no discharge, no plate move, no release), '
                            'then taken off the line by the simulation (manual re-measuring not modelled); section '
                            'upstream stops only when the buffer cannot take the next lump; the buffer stages its '
                            'lead item at S3 (or where the cameras see it past S3) while the measuring belt is busy; '
                            'the measuring belt weighs (indicator: %.1f s average steady within %.0f %% for %.1f s) and '
                            'scans (%.1f s) at the same time; route = gangue if mass/volume >= %.0f kg/m3; the plate '
                            'moves as soon as the route is known and the healthy plate-zone camera says its path is '
                            'clear; the discharge starts once the plate will be in position %.1f s before the lump '
                            'reaches it' % (w['filter_s'], 100 * w['steady_band'], w['steady_s'], self.cfg['scan_s'],
                                            self.cfg['sort_density'], PLATE_MARGIN_S))
