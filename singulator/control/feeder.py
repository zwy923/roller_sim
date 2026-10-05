"""Control of the feed belt: the batch is let go lump by lump over the step-down transfer in front of the plough
(third version 2026-09-23; the line's feed since the gated mainline was deleted, 2026-09-30). The hardware is
machine/feed_belt.py.

    feed belt (whole batch laid on it, own drive), its top feeder_step ABOVE the main belt
    -> main belt, straight run of feeder_gap -> plough face -> lane (0.60 m)

Why the step. On a feed belt butted flat to the main belt (the second version) a lump straddling the joint
is held or dragged by friction on two surfaces, so hold-back had to be judged by the FRONT edge; the row
plan then pulled in every lump the feed travel brought near the joint, and echelon layouts went five in
a row (all 77 five-lump rows in 1000 seeds, EXPERIMENTS.md). With the feed belt a step above the main belt,
a lump overhanging the head edge still lies wholly on the feed belt -- the overhang is in the air, clear of
the main belt -- until its CENTROID passes the edge; then it tips onto the main belt, which takes it away.
Hold-back is gravity, not friction, and it is by centroid for every lump: a lump beside the one that goes
stays behind even when its front already overhangs.

Control (one lump at a time; the camera only sets the creep, the beam stops):
  1. the whole batch is laid on the feed belt;
  2. a release runs the feed belt at feeder_speed, creeps once the leading lump's centroid is within
     CREEP_ZONE of the edge, and stops as soon as the beam is cut;
  3. a lump is released once its centroid is past the edge. Every lump released between the start of one
     release and the start of the next belongs to it; one that goes after the stop is counted
     ('after_stop'): its centroid was level with the one that went, within the stopping distance;
  4. the next release starts once every released lump is through the funnel -- its rear edge past the lane
     entry, where the lane's outer wall starts (lane_out_start), or off the belt -- and the plough face is
     back home. Until 2026-09-24 it waited for the tail to pass the lane EXIT, 0.82 m further on (user: go as
     soon as the lump is in the straight lane). The next lump cannot catch up: it still has the creep zone,
     the drop and feeder_gap + the face to cover, over 1.5 m at no more than the main belt speed;
  5. fallbacks, counted: a released lump left hanging on the edge (rear still on the feed belt) with no
     progress for feeder_stall_s gets a creep jog until its rear is CLEAR past the edge; a release whose
     lump went without cutting the beam stops BEAM_MISS_S after it went. That fallback says the lump is slow
     to tip, not that the beam is dead: the beam is taken for dead (the line stops) when two released lumps in
     a row have passed wholly beyond it while it never read blocked;
  6. while the station holds the section upstream, the feed belt brakes and this controller is frozen
     (pause / resume); its clocks do not run during the hold.

Predictive release (--feeder-release predict, the default since 2026-09-24, user: as many lumps as possible
without piling them up in the funnel; 'lane' is item 4 above). Item 4 is replaced and a staging step added:
  4'. between releases the feed belt STAGES the next lump: once the released lumps are off the edge it runs
      until the leading queued centroid is at the creep zone, so a release only has the creep left. A release
      starts only from there, so the decision below always looks the same short time ahead;
  5'. the next release starts as soon as it cannot put two lumps in the funnel: by the time the next lump's
      front reaches the funnel mouth (the plough start, x of P0), every lump released before must have its
      rear past the lane entry, with RELEASE_MARGIN_S to spare. How long they take is predicted from the
      camera: each one's centroid speed over the last feeder_stall_s, but no faster than V_FACE in the
      funnel while the plough face is still ahead of a lump that reaches above the lane band; a lump slower
      than V_STALL holds the release, and so does one still lying across the drop beam (it would stop the
      new release at once). The face must be home, as before.

What it reads. --feeder-sensing oracle (the default since 2026-10-05, user: 给料直接读质心): one object per lump
with the true centroid and hull (observe), and the drop beam. --feeder-sensing vision (the default from 2026-09-30
to 2026-10-04): the cameras' objects (outline centroid, not the centre of mass) and the debounced beam, judged by
region instead of by track id -- see _observe_vision. --sensing oracle makes the whole line ideal, this controller
included. What is still open about feeding by camera is listed in TODO.md.

A blob -- lumps lying against each other, one object to the cameras -- is led by its FRONT EDGE, not by its outline
centroid (since 2026-10-04; see _lead_x). The lump in front of a blob has its centroid
0.2-0.5 m ahead of the blob's. Staging the blob's centroid at the creep zone therefore ran the lump in front over
the edge at full feed speed while no release was on: nothing had judged whether the funnel was clear for it, and it
was booked to the release before (EXPERIMENTS.md S6). Led by the front edge a blob is staged with its front at the
edge and crept from there until the beam is cut.

The feed belt's own drive is force-limited with a finite brake (physics/drives.py): it runs at feeder_speed on
approach and at feeder_creep_speed once the leading lump's centroid is within CREEP_ZONE of the edge. Nothing here
is calibrated.
"""
from ..sensing import vision

CREEP_ZONE = .05     # m: creep once the leading centroid is this close to the edge (covers camera error)
BEAM_MISS_S = 1.     # s: a release whose lump went without cutting the beam stops this long after it went
STAGE_STOP = .016    # m: predictive release: staging stops this far short of the creep zone (stop from 0.10 m/s) ...
STAGE_SLACK = .03    # ... and starts again only if the lead lies this much further back
V_FACE = .15         # m/s: x-speed assumed in the funnel while the plough face is still ahead of a lump
V_STALL = .03        # m/s: a released lump slower than this holds the next release
RELEASE_MARGIN_S = 1.  # s: the next lump reaches the funnel mouth at least this long after the one before has left it
TIP_S = .3           # s: from centroid over the edge to lying on the main belt
CLEAR = .02          # m: a jog runs until the jogged lump's rear edge is this far past the edge
WENT_MARGIN = .05    # m: vision: an object is over the edge once its outline centroid is this far past it
                     # (the camera's centroid is up to ~4 cm off the centre of mass: seed 392, p95 2.6 cm) ...
JUST_WENT = .25      # m: ... and has only just gone while it is less than this past it
JOG_MAX_S = 3.       # s: vision: a hang jog runs at most this long without the beam being cut
STAGE_REACH = .80    # m: vision: objects from this far behind the edge to the lane entry must be recognised
NOSE_SHARE = 1 / 3   # vision: the lump in front of a blob has its centroid at least this share of size_min behind the
                     # blob's front edge (a convex outline's centroid lies at least a third of its width from every
                     # side; 3000 generated lumps at any heading: 0.085 m at the least, 0.12 m for 999 in 1000)
BLOB_FRAMES = 2      # vision: an outline that is not one convex lump this many frames running is a blob ...
UNBLOB_FRAMES = 20   # ... until it has looked like one lump for this many



class Feeder:
    """Lump-by-lump release. The drive itself is a Conveyors motor; this sets its target speed factor."""

    def __init__(self, cfg, d, n):
        self.cfg, self.g, self.n = cfg, d['feeder'], n
        self.lane_in = float(d['lane_out_start'])       # a released lump whose rear is past this is in the lane
        self.creep = self.g['creep_speed_m_s'] / self.g['speed_m_s']
        self.released, self.releases, self.events = [], [], []
        self.phase, self.goal, self.target = 'stopped', 0., 0.
        self.jog = None                                 # the released lump being jogged off the edge
        self.hist = {}                                  # (t, centroid x) per object, feeder_stall_s long
        self.counts = dict(after_stop=0, jogs=0, beam_missed=0)
        self.paused_at, self.held_s, self.miss_shift = None, 0., 0.   # station hold (see pause)
        self.predict = cfg['feeder_release'] == 'predict'
        self.staging = False
        # the cameras' objects, or the ideal view: one object per lump with its true centroid (--feeder-sensing
        # oracle, the default since 2026-10-05; --sensing oracle makes the whole line ideal)
        self.vision = cfg['sensing'] == 'vision' and cfg['feeder_sensing'] == 'vision'
        self.lumps_went = set()                         # vision: true lumps over the edge (verification)
        self.miss_streak = 0                            # released lumps in a row that passed the beam unseen
        self.across_hist = []                           # vision: (t, centroid x of the most advanced object across the edge)
        self.nose_min = NOSE_SHARE * cfg['size_min']
        self.blobs, self.blob_run = set(), {}           # vision: track ids that may hold several lumps (_mark_blobs)
        self.staged = False                             # the belt was run for staging since the last release stopped
        self.hang, self.jog_t0 = False, 0.              # vision: a release ended on the beam-miss fallback
        # a lump that went has to cross the hand-over gap before it reaches the beam (dragged at about half the
        # main belt speed while its rear still rests on the head drum): the fallback waits that much longer
        hand = self.g.get('x_lower', self.g['x1']) - self.g['x1']
        self.beam_miss_s = BEAM_MISS_S + (hand / (.5 * cfg['v_belt']) if hand > 0. else 0.)
        if self.predict:
            self.funnel_in, self.bend, self.lane_top = float(d['P0'][0]), float(d['P1'][0]), float(d['lane_top'])
            self.counts['stagings'] = 0

    def waiting(self, k, state, cx):
        """Planned wait (stall bookkeeping only): the lump still lies on the feed belt, centroid behind the edge."""
        return state == 'on_belt' and k not in (self.lumps_went if self.vision else self.released) and cx <= self.g['x1']

    def start(self, t):
        self.releases.append(dict(release=len(self.releases), members=[], after_stop=[], t_start_s=round(t, 3),
                                  t_creep_s=None, t_stop_s=None, stop='', jogs=0))
        self.phase, self.goal, self.jog, self.miss_shift = 'feeding', 1., None, 0.
        self.staged = False
        self.events.append(dict(t_s=round(t, 3), event='release_start', release=len(self.releases) - 1))

    # ---- station hold ----------------------------------------------------------------------------
    @property
    def paused(self):
        return self.paused_at is not None

    def pause(self, t):
        """The station holds the section upstream of the weigh belt: brake now, re-ramp after the hold."""
        if self.paused_at is None:
            self.paused_at, self.target = t, 0.

    def resume(self, t):
        """End of a hold: the beam-miss clock skips it, and progress is judged afresh."""
        if self.paused_at is None:
            return
        held, self.paused_at = t - self.paused_at, None
        self.held_s += held
        if self.releases and (self.releases[-1]['members'] or 't_first_s' in self.releases[-1]):
            self.miss_shift += held
        self.hist, self.across_hist = {}, []

    def _stop(self, t, why):
        rel = self.releases[-1]
        if rel['t_stop_s'] is None:
            rel['t_stop_s'], rel['stop'] = round(t, 3), why
        self.phase, self.goal = 'stopped', 0.
        self.events.append(dict(t_s=round(t, 3), event='stop', release=rel['release'], why=why))

    # ---- per step ------------------------------------------------------------------------------
    def drive_target(self, dt):
        """Ramped speed factor for the feed belt drive (1 = feeder_speed), advanced one physics step."""
        r = dt / self.cfg['feeder_ramp_s']
        g = self.goal
        self.target = min(g, self.target + r) if g > self.target else max(g, self.target - r)
        return self.target

    # ---- every camera sample --------------------------------------------------------------------
    def _no_progress(self, k, t):
        h, W = self.hist.get(k, []), self.cfg['feeder_stall_s']
        if not h or h[0][0] > t - W + 1e-9:
            return False
        past = next(x for tt, x in reversed(h) if tt <= t - W + 1e-9)
        return h[-1][1] - past < .02 * W

    def in_lane(self, k, view):
        """Through the funnel: rear edge past the lane entry, or no longer on the belt (passed / discharged /
        dropped: not in the view)."""
        return k not in view or view[k]['x0'] > self.lane_in

    def observe(self, t, view, beam_cut, face_home):
        """view: {id: object} -- what the camera reports of every lump or blob still on the machine: 'cx'
        (centroid x), 'x0' / 'x1' (rear / front edge), 'y1' (outer edge), and with --sensing vision 'vx',
        'conf' and 'coasting'. The oracle view (--sensing oracle) is one object per lump, its true centroid
        and hull. beam_cut: the drop beam is blocked; face_home: the plough face is back home."""
        if self.vision:
            return self._observe_vision(t, view, beam_cut, face_home)
        W, x1 = self.cfg['feeder_stall_s'], self.g['x1']
        for k, o in view.items():
            h = self.hist.setdefault(k, [])
            h.append((t, o['cx']))
            while len(h) > 2 and h[1][0] < t - W - .02:
                h.pop(0)
        went = [k for k, o in view.items() if k not in self.released and o['cx'] > x1]
        if went and self.releases:
            rel = self.releases[-1]
            rel['members'] += went
            self.released += went
            late = rel['t_stop_s'] is not None          # after the stop: while braking, waiting or jogging
            if late:
                rel['after_stop'] += went
                self.counts['after_stop'] += len(went)
            self.events.append(dict(t_s=round(t, 3), event='released', release=rel['release'], blocks=went,
                                    after_stop=late))
            rel.setdefault('t_first_s', round(t, 3))
        q = [k for k in view if k not in self.released]
        if self.releases:
            # the beam's own check, as with the cameras: a lump of this release wholly beyond the beam, and the beam
            # never blocked since the release started; twice in a row it is dead (simulate raises the alarm). Until
            # 2026-10-05 every release that ended on the beam-miss fallback counted instead. The fallback runs out
            # BEAM_MISS_S after the true centroid passed the edge, and one release in twelve takes longer to tip
            # into the beam (S7): two slow lumps in a row stopped the line over a beam that then saw both
            rel, bx = self.releases[-1], self.g['beam']['x']
            if beam_cut and not rel.get('beam_seen'):
                rel['beam_seen'] = True
                self.miss_streak = 0
            if not rel.get('beam_seen') and 'passed_unseen' not in rel \
                    and any(k in view and view[k]['x0'] > bx + CLEAR for k in rel['members']):
                rel['passed_unseen'] = round(t, 3)
                self.miss_streak += 1
        if self.phase == 'feeding':
            rel = self.releases[-1]
            if self.jog is not None:
                j = self.jog
                if j not in view or view[j]['x0'] > x1 + CLEAR:
                    self.jog = None
                    self._stop(t, 'jog_done')
            elif beam_cut:
                self._stop(t, 'beam')
            elif rel['members'] and t - rel['t_first_s'] - self.miss_shift > self.beam_miss_s:
                self.counts['beam_missed'] += 1
                self._stop(t, 'beam_missed')
            elif not q and not rel['members']:
                self._stop(t, 'feed_belt_empty')
            elif q and self.goal == 1. and max(view[k]['cx'] for k in q) > x1 - CREEP_ZONE:
                self.goal = self.creep
                rel['t_creep_s'] = round(t, 3)
        elif self.phase in ('stopped', 'empty'):
            # a released lump the main belt did not take off the edge: jog it across at creep speed
            hanging = [k for k in self.released if k in view and view[k]['x0'] < x1 < view[k]['x1']
                       and self._no_progress(k, t)]
            if hanging:
                self.jog, self.phase, self.goal, self.staging = hanging[0], 'feeding', self.creep, False
                self.releases[-1]['jogs'] += 1
                self.counts['jogs'] += 1
                self.events.append(dict(t_s=round(t, 3), event='jog', release=self.releases[-1]['release'],
                                        block=hanging[0]))
            elif not q:
                if self.phase != 'empty':
                    self.phase = 'empty'
                    self.events.append(dict(t_s=round(t, 3), event='feed_belt_empty'))
            elif self.predict:
                if face_home and not self.staging and self._staged(view, q) and self._clear_to_release(view, q):
                    self.start(t)
                else:
                    self._stage(t, view, q)
            elif all(self.in_lane(k, view) for k in self.released) and face_home:
                self.start(t)

    # ---- vision: by region, not by track id --------------------------------------------------------
    def _observe_vision(self, t, view, beam_cut, face_home):
        """The same control on what the cameras report. Track ids are no use at the head edge: a lump tipping
        against the one behind it merges with it and splits off again every few frames (seed 4001: 90 new ids
        in 2 s), and a controller keyed on ids takes the lump hanging on the edge for the next one to go. So:
          gone      an object whose outline centroid is WENT_MARGIN past the edge (the camera's centroid error);
          queued    every other object: still on the feed belt, possibly overhanging;
          lead      how far forward the next lump to go can be: an object's outline centroid, or for a blob the
                    point nose_min behind its front edge (_lead_x). Creep, staging and the predicted arrival at
                    the funnel all go by it;
          just gone a gone object within JUST_WENT of the edge: a release has put something over (starts the
                    beam-miss clock);
          across    a gone object still across the edge: jogged if the most advanced one has not moved 2 %% of
                    feeder_stall_s in feeder_stall_s (it hangs);
          release   lane rule: every gone object's rear past the lane entry; predictive rule: every gone object
                    short of the lane entry has a speed from its track and is predicted out in time (as 5'),
                    none lies across the beam; and in both, every object from STAGE_REACH behind the edge to the
                    lane entry is seen with confidence -- an object it cannot vouch for holds the release;
                    and in both, the drop beam is clear. A lump tipping over the edge cuts the beam before its
                    outline centroid is WENT_MARGIN past it: to the cameras it is still queued, and the lead. Without
                    the beam a release started on it at once and the same beam stopped it one sample later, again and
                    again, the feed belt kept near creep speed meanwhile (seed 392: four empty releases in 0.07 s;
                    2026-10-04);
          hang      a release that ended on the beam-miss fallback left a lump hanging on the edge (it went but
                    did not drop far enough to cut the beam). The camera may not tell it from the next lump
                    touching it (one blob). So no release until the beam has seen it: jog at creep until the
                    beam is cut, at most JOG_MAX_S (seed 4001: the next release started under a merged blob);
          dead beam the camera sees an object pass wholly beyond the beam after a release that never cut it;
                    twice in a row: the beam is dead (simulate raises the alarm)."""
        W, x1, m, bx = self.cfg['feeder_stall_s'], self.g['x1'], WENT_MARGIN, self.g['beam']['x']
        self._mark_blobs(view)
        gone = [o for o in view.values() if o['cx'] > x1 + m]
        q = [o for o in view.values() if o['cx'] <= x1 + m]
        just = [o for o in gone if o['cx'] < x1 + JUST_WENT]
        across = [o for o in gone if o['x0'] < x1 + CLEAR]
        h = self.across_hist
        h.append((t, max((o['cx'] for o in across), default=None)))
        while len(h) > 2 and h[1][0] < t - W - .02:
            h.pop(0)
        if self.releases:
            rel = self.releases[-1]
            if beam_cut and not rel.get('beam_seen'):
                rel['beam_seen'] = True
                self.miss_streak = 0
            # an object wholly past the beam after a release that never cut it: the beam missed it
            if 't_first_s' in rel and not rel.get('beam_seen') and not rel.get('passed_unseen') \
                    and any(bx + .02 < o['x0'] < bx + .30 for o in gone):
                rel['passed_unseen'] = round(t, 3)
                self.miss_streak += 1
        if self.phase == 'feeding':
            rel = self.releases[-1]
            if just and 't_first_s' not in rel:
                rel['t_first_s'] = round(t, 3)
                self.events.append(dict(t_s=round(t, 3), event='released', release=rel['release']))
            if self.jog == 'hang':
                if beam_cut or t - self.jog_t0 > JOG_MAX_S:
                    self.jog, self.hang = None, False
                    self._stop(t, 'jog_done' if beam_cut else 'jog_timeout')
            elif self.jog is not None:
                if not across:
                    self.jog = None
                    self._stop(t, 'jog_done')
            elif beam_cut:
                self._stop(t, 'beam')
            elif 't_first_s' in rel and t - rel['t_first_s'] - self.miss_shift > self.beam_miss_s:
                self.counts['beam_missed'] += 1
                self.hang = True                        # it went and the beam has not seen it: it hangs
                self._stop(t, 'beam_missed')
            elif not q and 't_first_s' not in rel:
                self._stop(t, 'feed_belt_empty')
            elif q and self.goal == 1. and max(self._lead_x(o) for o in q) > x1 - CREEP_ZONE:
                self.goal = self.creep
                rel['t_creep_s'] = round(t, 3)
        elif self.phase in ('stopped', 'empty'):
            stuck = (across and len(h) > 1 and h[0][0] <= t - W + .02 and None not in (h[0][1], h[-1][1])
                     and h[-1][1] - h[0][1] < .02 * W)
            if self.hang:
                self.jog, self.jog_t0, self.phase, self.goal, self.staging = 'hang', t, 'feeding', self.creep, False
                self.releases[-1]['jogs'] += 1
                self.counts['jogs'] += 1
                self.events.append(dict(t_s=round(t, 3), event='jog', release=self.releases[-1]['release'], why='hang'))
            elif stuck:
                self.jog, self.phase, self.goal, self.staging = 'across', 'feeding', self.creep, False
                self.releases[-1]['jogs'] += 1
                self.counts['jogs'] += 1
                self.events.append(dict(t_s=round(t, 3), event='jog', release=self.releases[-1]['release']))
            elif not q:
                if self.phase != 'empty':
                    self.phase = 'empty'
                    self.events.append(dict(t_s=round(t, 3), event='feed_belt_empty'))
            elif not self._sure(view):
                if self.staging:                        # an object it cannot vouch for: do not move on a guess
                    self.staging, self.goal = False, 0.
                self.counts['unsure_waits'] = self.counts.get('unsure_waits', 0) + 1
            elif self.predict:
                lead = max(self._lead_x(o) for o in q)
                if face_home and not beam_cut and not self.staging \
                        and lead >= x1 - CREEP_ZONE - STAGE_STOP - STAGE_SLACK and self._clear_v(gone, q):
                    self.start(t)
                else:
                    self._stage_v(t, lead, bool(across))
            elif all(o['x0'] > self.lane_in for o in gone) and face_home and not beam_cut:
                self.start(t)

    def _clear_v(self, gone, q):
        """5' on the cameras' objects: every gone object short of the lane entry out in time."""
        t_clear = 0.
        for o in gone:
            if o['x0'] > self.lane_in:
                continue
            v = o.get('vx')
            if v is None or v < V_STALL or o['x0'] < self.g['beam']['x'] + CLEAR:
                return False
            rear = float(o['x0'])
            face_ahead = rear < self.bend and float(o['y1']) > self.lane_top
            t_clear = max(t_clear, max(0., self.funnel_in - rear) / self.cfg['v_belt']
                          + (self.lane_in - max(rear, self.funnel_in)) / (min(v, V_FACE) if face_ahead else v))
        if t_clear == 0.:
            return True
        k = max(q, key=self._lead_x)                    # the next to go; a blob: the earliest its front lump can
        x1, g, lead = self.g['x1'], self.g, self._lead_x(k)
        d_c = max(0., x1 - lead)
        t_tip = max(0., d_c - CREEP_ZONE) / g['speed_m_s'] + min(d_c, CREEP_ZONE) / g['creep_speed_m_s'] + TIP_S
        t_arrive = t_tip + max(0., self.funnel_in - (x1 + float(k['x1']) - lead)) / self.cfg['v_belt']
        return t_clear + RELEASE_MARGIN_S <= t_arrive

    def _mark_blobs(self, view):
        """Vision: which objects may hold more than one lump. The cameras count several (n_est: tracks that ran
        together, or an outline clearly not convex), or the outline has not looked like one convex lump (solid
        below SOLID_SURE) for BLOB_FRAMES frames running. It stays a blob until it has looked like one lump for
        UNBLOB_FRAMES: the shape measure crosses its thresholds from frame to frame, and a rule read off single
        frames would stage on the frames that say 'one lump'."""
        for tid, o in view.items():
            several = o.get('n_est', 1) > 1
            n = self.blob_run.get(tid, 0)
            n = max(n, 0) + 1 if several or o.get('solid', 1.) < vision.SOLID_SURE else min(n, 0) - 1
            self.blob_run[tid] = n
            if several or n >= BLOB_FRAMES:
                self.blobs.add(tid)
            elif n <= -UNBLOB_FRAMES:
                self.blobs.discard(tid)
        self.blobs &= set(view)
        self.blob_run = {tid: n for tid, n in self.blob_run.items() if tid in view}

    def _lead_x(self, o):
        """Vision: how far forward the centroid of the leading lump in object o can be. One lump: its outline
        centroid. A blob (_mark_blobs): the cameras cannot see which lump is in front or how long it is, only
        that its centroid lies at least nose_min behind the blob's front edge. Until 2026-10-04 a blob was led by
        its own centroid: staging it then ran the lump in front over the edge with no release decided."""
        if o.get('tid') in self.blobs:
            return max(o['cx'], o['x1'] - self.nose_min)
        return o['cx']

    def _stage_v(self, t, lead, across):
        x1 = self.g['x1']
        if self.staging:
            if lead >= x1 - CREEP_ZONE - STAGE_STOP or across:
                self.staging, self.goal = False, 0.
                self.events.append(dict(t_s=round(t, 3), event='staged', lead_to_edge_m=round(x1 - lead, 3)))
        elif not across and self.phase == 'stopped' and lead < x1 - CREEP_ZONE - STAGE_STOP - STAGE_SLACK:
            self.staging, self.goal, self.staged = True, 1., True
            self.counts['stagings'] += 1
            self.events.append(dict(t_s=round(t, 3), event='staging'))

    def _sure(self, view):
        """Vision only: every object between STAGE_REACH behind the head edge and the lane entry is seen now
        with CONF_OK or better. One the cameras are unsure of holds the release and the staging. A blob still on
        the feed belt is the exception: what is in doubt about it is how many lumps it holds,
        not where it is, and it is led by its front edge whatever the count. It only has to be SEEN (conf_seen,
        the confidence without the shape term); otherwise the feed belt would wait for as long as the blob looks
        neither like one lump nor like two."""
        x1 = self.g['x1']
        for o in view.values():
            if not (o['x1'] > x1 - STAGE_REACH and o['x0'] < self.lane_in):
                continue
            blob = o.get('tid') in self.blobs and o['cx'] <= x1 + WENT_MARGIN
            if o.get('coasting') or (o.get('conf_seen', o['conf']) if blob else o['conf']) < vision.CONF_OK:
                return False
        return True

    def audit(self, t, com_x):
        """Vision only, for the verification: which true lumps went over the edge in which release (true
        centroid past the edge), which of them after its stop, and which of those only once the belt had been
        run again for staging ('lumps_in_staging': carried over with no release decided, not by the release's
        own stopping distance). com_x: {lump: true centroid x} of the lumps still on the machine."""
        for k, x in com_x.items():
            if k not in self.lumps_went and x > self.g['x1'] and self.releases:
                rel = self.releases[-1]
                self.lumps_went.add(k)
                rel.setdefault('lumps', []).append(k)
                if rel['t_stop_s'] is not None:
                    rel.setdefault('lumps_after_stop', []).append(k)
                    if self.staged:
                        rel.setdefault('lumps_in_staging', []).append(k)

    # ---- predictive release ----------------------------------------------------------------------
    def _speed(self, k, view):
        """Centroid x-speed: with vision the track's own estimate, else from the camera history (None until
        it spans 0.3 s)."""
        if self.vision:
            return view[k].get('vx')
        h = self.hist[k]
        return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0]) if h and h[-1][0] - h[0][0] >= .3 - 1e-9 else None

    def _clear_to_release(self, view, q):
        """Can the next lump go without meeting one released before it in the funnel (see 5')?"""
        t_clear = 0.
        for k in self.released:
            if k not in view or view[k]['x0'] > self.lane_in:
                continue
            v = self._speed(k, view)
            if v is None or v < V_STALL or view[k]['x0'] < self.g['beam']['x'] + CLEAR:
                return False
            rear = float(view[k]['x0'])
            face_ahead = rear < self.bend and float(view[k]['y1']) > self.lane_top
            t_clear = max(t_clear, max(0., self.funnel_in - rear) / self.cfg['v_belt']
                          + (self.lane_in - max(rear, self.funnel_in)) / (min(v, V_FACE) if face_ahead else v))
        if t_clear == 0.:
            return True
        k = max(q, key=lambda j: view[j]['cx'])         # the next to go, by centroid
        x1, g = self.g['x1'], self.g
        d_c = max(0., x1 - view[k]['cx'])
        t_tip = max(0., d_c - CREEP_ZONE) / g['speed_m_s'] + min(d_c, CREEP_ZONE) / g['creep_speed_m_s'] + TIP_S
        nose = float(view[k]['x1']) - view[k]['cx']     # its front ahead of its centroid
        t_arrive = t_tip + max(0., self.funnel_in - (x1 + nose)) / self.cfg['v_belt']
        return t_clear + RELEASE_MARGIN_S <= t_arrive

    def _staged(self, view, q):
        return max(view[k]['cx'] for k in q) >= self.g['x1'] - CREEP_ZONE - STAGE_STOP - STAGE_SLACK

    def _stage(self, t, view, q):
        """Between releases: bring the next lump up to the creep zone, so the next release only creeps."""
        x1 = self.g['x1']
        lead = max(view[k]['cx'] for k in q)
        off_edge = all(k not in view or view[k]['x0'] > x1 + CLEAR for k in self.released)
        if self.staging:
            if lead >= x1 - CREEP_ZONE - STAGE_STOP or not off_edge:
                self.staging, self.goal = False, 0.
                self.events.append(dict(t_s=round(t, 3), event='staged', lead_to_edge_m=round(x1 - lead, 3)))
        elif off_edge and self.phase == 'stopped' and lead < x1 - CREEP_ZONE - STAGE_STOP - STAGE_SLACK:
            self.staging, self.goal = True, 1.
            self.counts['stagings'] += 1
            self.events.append(dict(t_s=round(t, 3), event='staging'))

    def geometry(self):
        """The feed belt for result.json: the hardware with the creep zone this controller adds."""
        return dict(self.g, creep_zone_m=CREEP_ZONE)

    def report(self):
        sizes = [len(r['lumps' if self.vision else 'members']) for r in self.releases
                 if r.get('lumps' if self.vision else 'members')]
        # Which lumps went, and which of them after the stop. With vision the controller has no lump identities:
        # self.released and counts['after_stop'] belong to the oracle path and stay empty, so the report takes
        # both from the verification record (audit). Until 2026-10-04 it read the oracle's, and a vision run
        # listed every lump as never released
        went = [k for r in self.releases for k in r.get('lumps', [])] if self.vision else list(self.released)
        after_stop = (sum(len(r.get('lumps_after_stop', [])) for r in self.releases) if self.vision
                      else self.counts['after_stop'])
        in_staging = sum(len(r.get('lumps_in_staging', [])) for r in self.releases)
        return dict(geometry=self.geometry(), releases=self.releases, events=self.events, final_phase=self.phase,
                    next_release_rear_past_x_m=round(self.lane_in, 4), released_order=went,
                    never_released=[k for k in range(self.n) if k not in went],
                    counts=dict(self.counts, after_stop=after_stop, in_staging=in_staging, releases=len(sizes),
                                single=sum(s == 1 for s in sizes),
                                lumps_per_release={str(s): sizes.count(s) for s in sorted(set(sizes))}),
                    held_by_station_s=round(self.held_s, 2),
                    **(dict(release='predict', funnel_mouth_x_m=round(self.funnel_in, 4),
                            release_margin_s=RELEASE_MARGIN_S, v_face_m_s=V_FACE) if self.predict else {}),
                    control='run at feeder_speed, creep once the leading centroid is within the creep zone, stop '
                            'when the drop beam is cut; a lump is released once its centroid is past the head edge '
                            '(it tips); ' + ('between releases the next lump is staged at the creep zone; the next '
                                             'release starts once every lump released before is predicted to be '
                                             'out of the funnel (rear past the lane entry) %.1f s before the next '
                                             'one reaches the funnel mouth, and the face is home' % RELEASE_MARGIN_S
                                             if self.predict else
                                             'next release once every released lump is through the funnel (rear '
                                             'past the lane entry) and the face is home'))
