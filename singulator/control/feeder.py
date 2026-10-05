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
     CREEP_ZONE of the edge, and stops as soon as the beam is cut. With --feeder-stop centroid (the default since
     2026-10-06, true centroids only) it stops instead once the centroid that went is STOP_PAST past the edge, and
     the lump tips on its own: creeping on until it had tipped into the beam (half a second, 1.6 cm at the median)
     carried a second lump over in one release of eleven (S8). A beam cut before any centroid is over -- a lump
     sagging over the edge -- then only means creep on (EARLY_BEAM_CREEP);
  3. a lump is released once its centroid is past the edge. Every lump released between the start of one
     release and the start of the next belongs to it; one that goes after the stop is counted
     ('after_stop'): its centroid was level with the one that went, within the stopping distance;
  4. the next release starts once every released lump is through the funnel -- its rear edge past the lane
     entry, where the lane's outer wall starts (lane_out_start), or off the belt -- and the plough face is
     back home. Until 2026-09-24 it waited for the tail to pass the lane EXIT, 0.82 m further on (user: go as
     soon as the lump is in the straight lane). The next lump cannot catch up: it still has the creep zone,
     the drop and feeder_gap + the face to cover, over 1.5 m at no more than the main belt speed;
  5. fallbacks, counted: a released lump left hanging on the edge (rear still on the feed belt) with no
     progress for feeder_stall_s gets a creep jog until its rear is CLEAR past the edge (--feeder-stop centroid:
     JOG_STEP at a time, feeder_stall_s apart, so that a jog does not push the next lump over); a release whose
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

What it reads: the drop beam, and a view of the lumps that one of two readers interprets for the one phase machine
(Feeder.observe):
  * --feeder-sensing oracle (the default since 2026-10-05, user: 给料直接读质心; and --sensing oracle, which makes
    the whole line ideal): one object per lump with its true centroid and hull, known by its id (_Ideal);
  * --feeder-sensing vision (the default from 2026-09-30 to 2026-10-04): the cameras' objects -- an outline centroid,
    not the centre of mass -- and the debounced beam, judged by region instead of by track id (_Camera, which says
    why). What is still open about feeding by camera is listed in TODO.md.

A blob -- lumps lying against each other, one object to the cameras -- is led by its FRONT EDGE, not by its outline
centroid (since 2026-10-04; see _Camera.lead_x). The lump in front of a blob has its centroid
0.2-0.5 m ahead of the blob's. Staging the blob's centroid at the creep zone therefore ran the lump in front over
the edge at full feed speed while no release was on: nothing had judged whether the funnel was clear for it, and it
was booked to the release before (EXPERIMENTS.md S6). Led by the front edge a blob is staged with its front at the
edge and crept from there until the beam is cut.

The feed belt's own drive is force-limited with a finite brake (physics/drives.py): it runs at feeder_speed on
approach and at feeder_creep_speed once the leading lump's centroid is within CREEP_ZONE of the edge. Nothing here
is calibrated. Which true lump went in which release, when the cameras cannot say, is the witness's record
(verify/feeder.py).
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
STOP_PAST = .003     # m: --feeder-stop centroid: a release stops once its leading centroid is this far past the head
                     # edge, and the lump tips on its own (TODO.md section 1, 改法二; S8)
JOG_STEP = .005      # m: --feeder-stop centroid: a hang jog runs only until the jogged lump's centroid is this much
                     # further on, then waits feeder_stall_s for it to tip before the next (None: until its rear is
                     # clear, as with --feeder-stop beam)
EARLY_BEAM_CREEP = 1  # 1: (the true centroid only) the beam cut before any centroid of the release is over the edge
                     # -- a lump sagging over the edge while still lying on the feed belt -- does not stop the release:
                     # it creeps on and stops once a centroid is over. 0: it stops at once, and the next release
                     # starts and is stopped again every sample (touching 7125: 53 releases, TODO.md section 1)


class Feeder:
    """Lump-by-lump release. The drive itself is a Conveyors motor; this sets its target speed factor."""

    def __init__(self, cfg, d):
        self.cfg, self.g = cfg, d['feeder']
        self.lane_in = float(d['lane_out_start'])       # a released lump whose rear is past this is in the lane
        self.creep = self.g['creep_speed_m_s'] / self.g['speed_m_s']
        self.released, self.releases, self.events = [], [], []     # released: lump ids, the ideal view only
        self.phase, self.goal, self.target = 'stopped', 0., 0.
        self.jog = None                                 # what is being jogged off the edge (the reader's token)
        self.counts = dict(after_stop=0, jogs=0, beam_missed=0)
        self.paused_at, self.held_s, self.miss_shift = None, 0., 0.   # station hold (see pause)
        self.predict = cfg['feeder_release'] == 'predict'
        self.staging = False
        self.staged = False                             # the belt was run for staging since the last release stopped
        self.miss_streak = 0                            # released lumps in a row that passed the beam unseen
        # a lump that went has to cross the hand-over gap before it reaches the beam (dragged at about half the
        # main belt speed while its rear still rests on the head drum): the fallback waits that much longer
        hand = self.g.get('x_lower', self.g['x1']) - self.g['x1']
        self.beam_miss_s = BEAM_MISS_S + (hand / (.5 * cfg['v_belt']) if hand > 0. else 0.)
        if self.predict:
            self.funnel_in, self.bend, self.lane_top = float(d['P0'][0]), float(d['P1'][0]), float(d['lane_top'])
            self.counts['stagings'] = 0
        # the cameras' objects, or the ideal view: one object per lump with its true centroid (--feeder-sensing
        # oracle, the default since 2026-10-05; --sensing oracle makes the whole line ideal)
        self.vision = cfg['sensing'] == 'vision' and cfg['feeder_sensing'] == 'vision'
        # --feeder-stop centroid needs the true centroid: the cameras' outline centroid is off it by centimetres, so
        # on the camera path the beam ends every release
        self.stop_past = STOP_PAST if cfg['feeder_stop'] == 'centroid' and not self.vision else None
        self.reader = _Camera(self) if self.vision else _Ideal(self)

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
        self.reader.forget()

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
    def observe(self, t, view, beam_cut, face_home):
        """One 10 ms sample of the phase machine. view: {id: object} -- every lump or blob still on the machine as
        this controller's view reports it: 'cx' (centroid x), 'x0' / 'x1' (rear / front edge), 'y1' (outer edge),
        and from the cameras 'vx', 'conf', 'coasting', 'solid', 'n_est'. beam_cut: the drop beam is blocked;
        face_home: the plough face is back home. What 'gone over the edge', 'queued', 'hanging' and 'clear to
        go' mean on that view is the reader's part (_Ideal, _Camera)."""
        r, x1 = self.reader, self.g['x1']
        r.look(t, view)
        if self.releases:
            # the beam's own check: something of this release wholly beyond the beam, and the beam never blocked
            # since the release started; twice in a row it is dead (the run raises the alarm). Until 2026-10-05
            # every release that ended on the beam-miss fallback counted instead. The fallback runs out BEAM_MISS_S
            # after the centroid passed the edge, and one release in twelve takes longer to tip into the beam (S7):
            # two slow lumps in a row stopped the line over a beam that then saw both
            rel = self.releases[-1]
            if beam_cut and not rel.get('beam_seen'):
                rel['beam_seen'] = True
                self.miss_streak = 0
            if not rel.get('beam_seen') and r.passed_unseen(rel):
                rel['passed_unseen'] = round(t, 3)
                self.miss_streak += 1
        if self.phase == 'feeding':
            rel = self.releases[-1]
            r.feeding(t, rel)
            if self.jog is not None:
                why = r.jog_over(t, self.jog, beam_cut)
                if why:
                    self.jog = None
                    self._stop(t, why)
            elif beam_cut and EARLY_BEAM_CREEP and not self.vision and 't_first_s' not in rel:
                if 't_early_beam_s' not in rel:         # nothing over yet: creep on until a centroid is
                    rel['t_early_beam_s'] = round(t, 3)
                    self.counts['early_beam'] = self.counts.get('early_beam', 0) + 1
                self.goal = self.creep
            elif beam_cut:
                self._stop(t, 'beam')
            elif self.stop_past is not None and r.went_lead(rel) > x1 + self.stop_past:
                self._stop(t, 'went')
                r.forget()                              # it has feeder_stall_s from here to tip before a jog
            elif 't_first_s' in rel and t - rel['t_first_s'] - self.miss_shift > self.beam_miss_s:
                self.counts['beam_missed'] += 1
                r.missed()
                self._stop(t, 'beam_missed')
            elif not r.queued and 't_first_s' not in rel:
                self._stop(t, 'feed_belt_empty')
            elif r.queued and self.goal == 1. and r.lead() > x1 - CREEP_ZONE:
                self.goal = self.creep
                rel['t_creep_s'] = round(t, 3)
        elif self.phase in ('stopped', 'empty'):
            jog = r.jog_due(t)
            if jog:
                # something that went and was not taken off the edge: jog it across at creep speed
                self.jog, info = jog
                self.phase, self.goal, self.staging = 'feeding', self.creep, False
                self.releases[-1]['jogs'] += 1
                self.counts['jogs'] += 1
                self.events.append(dict(t_s=round(t, 3), event='jog', release=self.releases[-1]['release'], **info))
            elif not r.queued:
                if self.phase != 'empty':
                    self.phase = 'empty'
                    self.events.append(dict(t_s=round(t, 3), event='feed_belt_empty'))
            elif not r.sure():
                if self.staging:                        # an object it cannot vouch for: do not move on a guess
                    self.staging, self.goal = False, 0.
                self.counts['unsure_waits'] = self.counts.get('unsure_waits', 0) + 1
            elif self.predict:
                lead = r.lead()
                if face_home and r.beam_ok(beam_cut) and not self.staging \
                        and lead >= x1 - CREEP_ZONE - STAGE_STOP - STAGE_SLACK and r.funnel_clear():
                    self.start(t)
                else:
                    self._stage(t, lead, r.off_edge())
            elif r.lane_clear() and face_home and r.beam_ok(beam_cut):
                self.start(t)

    def _stage(self, t, lead, off_edge):
        """Between releases: bring the next lump up to the creep zone, so the next release only creeps. lead: the
        leading queued centroid; off_edge: nothing that went still lies across the edge."""
        x1 = self.g['x1']
        if self.staging:
            if lead >= x1 - CREEP_ZONE - STAGE_STOP or not off_edge:
                self.staging, self.goal = False, 0.
                self.events.append(dict(t_s=round(t, 3), event='staged', lead_to_edge_m=round(x1 - lead, 3)))
        elif off_edge and self.phase == 'stopped' and lead < x1 - CREEP_ZONE - STAGE_STOP - STAGE_SLACK:
            self.staging, self.goal, self.staged = True, 1., True
            self.counts['stagings'] += 1
            self.events.append(dict(t_s=round(t, 3), event='staging'))

    def _funnel_clear(self, ahead, lead, reach):
        """Can the next lump go without meeting one released before it in the funnel (see 5')? ahead: (object,
        its x-speed or None) of everything that went and is still seen; lead: the leading centroid of the next to
        go; reach: where its front is when that centroid is at the edge."""
        t_clear = 0.
        for o, v in ahead:
            if o['x0'] > self.lane_in:
                continue
            if v is None or v < V_STALL or o['x0'] < self.g['beam']['x'] + CLEAR:
                return False
            rear = float(o['x0'])
            face_ahead = rear < self.bend and float(o['y1']) > self.lane_top
            t_clear = max(t_clear, max(0., self.funnel_in - rear) / self.cfg['v_belt']
                          + (self.lane_in - max(rear, self.funnel_in)) / (min(v, V_FACE) if face_ahead else v))
        if t_clear == 0.:
            return True
        x1, g = self.g['x1'], self.g
        d_c = max(0., x1 - lead)
        t_tip = max(0., d_c - CREEP_ZONE) / g['speed_m_s'] + min(d_c, CREEP_ZONE) / g['creep_speed_m_s'] + TIP_S
        t_arrive = t_tip + max(0., self.funnel_in - reach) / self.cfg['v_belt']
        return t_clear + RELEASE_MARGIN_S <= t_arrive

    # ---- result.json -----------------------------------------------------------------------------
    def geometry(self):
        """The feed belt for result.json: the hardware with the creep zone this controller adds."""
        return dict(self.g, creep_zone_m=CREEP_ZONE, **({} if self.stop_past is None else dict(
            stop_rule='centroid', stop_past_m=self.stop_past, jog_step_m=JOG_STEP)))

    def report(self):
        """The controller's record. (Which lumps went, release by release, is added by verify.feeder.FeederWitness:
        with the cameras the controller has no lump identities.)"""
        return dict(geometry=self.geometry(), releases=self.releases, events=self.events, final_phase=self.phase,
                    next_release_rear_past_x_m=round(self.lane_in, 4), counts=dict(self.counts),
                    held_by_station_s=round(self.held_s, 2),
                    **(dict(release='predict', funnel_mouth_x_m=round(self.funnel_in, 4),
                            release_margin_s=RELEASE_MARGIN_S, v_face_m_s=V_FACE) if self.predict else {}),
                    control='run at feeder_speed, creep once the leading centroid is within the creep zone, stop '
                            + ('when the drop beam is cut' if self.stop_past is None else
                               'once the centroid that went is %.3f m past the head edge (the beam: before any did)'
                               % self.stop_past)
                            + '; a lump is released once its centroid is past the head edge (it tips); '
                            + ('between releases the next lump is staged at the creep zone; the next '
                                             'release starts once every lump released before is predicted to be '
                                             'out of the funnel (rear past the lane entry) %.1f s before the next '
                                             'one reaches the funnel mouth, and the face is home' % RELEASE_MARGIN_S
                                             if self.predict else
                                             'next release once every released lump is through the funnel (rear '
                                             'past the lane entry) and the face is home'))


class _Ideal:
    """The ideal view: one object per lump, its true centroid and hull, under the lump's own id. A lump has gone
    once its centroid is past the head edge, and the controller knows which one it was (Feeder.released, the
    'members' of a release)."""

    def __init__(self, f):
        self.f = f
        self.hist = {}                                  # (t, centroid x) per object, feeder_stall_s long
        self.view, self.q = {}, []
        self.jog_from = None                            # centroid x of the jogged lump when its jog started

    def forget(self):
        self.hist = {}

    def look(self, t, view):
        f = self.f
        W, x1 = f.cfg['feeder_stall_s'], f.g['x1']
        self.view = view
        for k, o in view.items():
            h = self.hist.setdefault(k, [])
            h.append((t, o['cx']))
            while len(h) > 2 and h[1][0] < t - W - .02:
                h.pop(0)
        went = [k for k, o in view.items() if k not in f.released and o['cx'] > x1]
        if went and f.releases:
            rel = f.releases[-1]
            rel['members'] += went
            f.released += went
            late = rel['t_stop_s'] is not None          # after the stop: while braking, waiting or jogging
            if late:
                rel['after_stop'] += went
                f.counts['after_stop'] += len(went)
            f.events.append(dict(t_s=round(t, 3), event='released', release=rel['release'], blocks=went,
                                 after_stop=late))
            rel.setdefault('t_first_s', round(t, 3))
        self.q = [k for k in view if k not in f.released]   # still on the feed belt

    queued = property(lambda self: bool(self.q))

    def lead(self):
        return max(self.view[k]['cx'] for k in self.q)

    def went_lead(self, rel):
        """The leading centroid of the lumps this release has put over the edge (-inf before any)."""
        view = self.view
        return max((view[k]['cx'] for k in rel['members'] if k in view), default=float('-inf'))

    def passed_unseen(self, rel):
        """A lump of this release lies wholly beyond the beam."""
        view, bx = self.view, self.f.g['beam']['x']
        return 'passed_unseen' not in rel and any(k in view and view[k]['x0'] > bx + CLEAR for k in rel['members'])

    def feeding(self, t, rel):
        pass

    def missed(self):
        pass

    def jog_over(self, t, jog, beam_cut):
        """The jogged lump's rear is CLEAR past the edge (or it is gone from the view); with JOG_STEP, also once its
        centroid has gone JOG_STEP further: it is then given feeder_stall_s to tip before the next jog."""
        view = self.view
        if jog not in view or view[jog]['x0'] > self.f.g['x1'] + CLEAR:
            return 'jog_done'
        if JOG_STEP is not None and self.f.stop_past is not None and view[jog]['cx'] > self.jog_from + JOG_STEP:
            self.forget()
            return 'jog_step'
        return None

    def jog_due(self, t):
        """A released lump the main belt did not take off the edge: across it, and no progress."""
        f, view, x1 = self.f, self.view, self.f.g['x1']
        hanging = [k for k in f.released if k in view and view[k]['x0'] < x1 < view[k]['x1']
                   and self._no_progress(k, t)]
        if hanging:
            self.jog_from = view[hanging[0]]['cx']
        return (hanging[0], dict(block=hanging[0])) if hanging else None

    def _no_progress(self, k, t):
        h, W = self.hist.get(k, []), self.f.cfg['feeder_stall_s']
        if not h or h[0][0] > t - W + 1e-9:
            return False
        past = next(x for tt, x in reversed(h) if tt <= t - W + 1e-9)
        return h[-1][1] - past < .02 * W

    def sure(self):
        return True

    def beam_ok(self, beam_cut):
        return True

    def off_edge(self):
        view, x1 = self.view, self.f.g['x1']
        return all(k not in view or view[k]['x0'] > x1 + CLEAR for k in self.f.released)

    def lane_clear(self):
        """Every released lump is through the funnel: rear edge past the lane entry, or no longer on the belt
        (passed / discharged / dropped: not in the view)."""
        view, lane_in = self.view, self.f.lane_in
        return all(k not in view or view[k]['x0'] > lane_in for k in self.f.released)

    def _speed(self, k):
        """Centroid x-speed from the history (None until it spans 0.3 s)."""
        h = self.hist[k]
        return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0]) if h and h[-1][0] - h[0][0] >= .3 - 1e-9 else None

    def funnel_clear(self):
        f, view = self.f, self.view
        k = max(self.q, key=lambda j: view[j]['cx'])    # the next to go, by centroid
        nose = float(view[k]['x1']) - view[k]['cx']     # its front ahead of its centroid
        return f._funnel_clear(((view[j], self._speed(j)) for j in f.released if j in view),
                               view[k]['cx'], f.g['x1'] + nose)


class _Camera:
    """The cameras' objects. Track ids are no use at the head edge: a lump tipping against the one behind it
    merges with it and splits off again every few frames (seed 4001: 90 new ids in 2 s), and a controller keyed
    on ids takes the lump hanging on the edge for the next one to go. So everything is judged by region:
      gone      an object whose outline centroid is WENT_MARGIN past the edge (the camera's centroid error);
      queued    every other object: still on the feed belt, possibly overhanging;
      lead      how far forward the next lump to go can be: an object's outline centroid, or for a blob the
                point nose_min behind its front edge (lead_x). Creep, staging and the predicted arrival at
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
                twice in a row: the beam is dead (the run raises the alarm)."""

    def __init__(self, f):
        self.f = f
        self.nose_min = NOSE_SHARE * f.cfg['size_min']
        self.blobs, self.blob_run = set(), {}           # track ids that may hold several lumps (_mark_blobs)
        self.across_hist = []                           # (t, centroid x of the most advanced object across the edge)
        self.hang, self.jog_t0 = False, 0.              # a release ended on the beam-miss fallback
        self.view, self.q, self.gone, self.just, self.across = {}, [], [], [], []

    def forget(self):
        self.across_hist = []

    def look(self, t, view):
        f = self.f
        W, x1, m = f.cfg['feeder_stall_s'], f.g['x1'], WENT_MARGIN
        self.view = view
        self._mark_blobs(view)
        self.gone = [o for o in view.values() if o['cx'] > x1 + m]
        self.q = [o for o in view.values() if o['cx'] <= x1 + m]
        self.just = [o for o in self.gone if o['cx'] < x1 + JUST_WENT]
        self.across = [o for o in self.gone if o['x0'] < x1 + CLEAR]
        h = self.across_hist
        h.append((t, max((o['cx'] for o in self.across), default=None)))
        while len(h) > 2 and h[1][0] < t - W - .02:
            h.pop(0)

    queued = property(lambda self: bool(self.q))

    def lead(self):
        return max(self.lead_x(o) for o in self.q)

    def passed_unseen(self, rel):
        """An object wholly past the beam after a release that never cut it: the beam missed it."""
        bx = self.f.g['beam']['x']
        return 't_first_s' in rel and not rel.get('passed_unseen') and any(bx + .02 < o['x0'] < bx + .30
                                                                           for o in self.gone)

    def feeding(self, t, rel):
        """The first object over the edge dates the release."""
        if self.just and 't_first_s' not in rel:
            rel['t_first_s'] = round(t, 3)
            self.f.events.append(dict(t_s=round(t, 3), event='released', release=rel['release']))

    def missed(self):
        self.hang = True                                # it went and the beam has not seen it: it hangs

    def jog_over(self, t, jog, beam_cut):
        if jog == 'hang':
            if beam_cut or t - self.jog_t0 > JOG_MAX_S:
                self.hang = False
                return 'jog_done' if beam_cut else 'jog_timeout'
            return None
        return None if self.across else 'jog_done'

    def jog_due(self, t):
        h, W = self.across_hist, self.f.cfg['feeder_stall_s']
        stuck = (self.across and len(h) > 1 and h[0][0] <= t - W + .02 and None not in (h[0][1], h[-1][1])
                 and h[-1][1] - h[0][1] < .02 * W)
        if self.hang:
            self.jog_t0 = t
            return 'hang', dict(why='hang')
        return ('across', {}) if stuck else None

    def sure(self):
        """Every object between STAGE_REACH behind the head edge and the lane entry is seen now with CONF_OK or
        better. One the cameras are unsure of holds the release and the staging. A blob still on the feed belt is
        the exception: what is in doubt about it is how many lumps it holds, not where it is, and it is led by its
        front edge whatever the count. It only has to be SEEN (conf_seen, the confidence without the shape term);
        otherwise the feed belt would wait for as long as the blob looks neither like one lump nor like two."""
        x1 = self.f.g['x1']
        for o in self.view.values():
            if not (o['x1'] > x1 - STAGE_REACH and o['x0'] < self.f.lane_in):
                continue
            blob = o.get('tid') in self.blobs and o['cx'] <= x1 + WENT_MARGIN
            if o.get('coasting') or (o.get('conf_seen', o['conf']) if blob else o['conf']) < vision.CONF_OK:
                return False
        return True

    def beam_ok(self, beam_cut):
        return not beam_cut

    def off_edge(self):
        return not self.across

    def lane_clear(self):
        return all(o['x0'] > self.f.lane_in for o in self.gone)

    def funnel_clear(self):
        f = self.f
        k = max(self.q, key=self.lead_x)                # the next to go; a blob: the earliest its front lump can
        lead = self.lead_x(k)
        return f._funnel_clear(((o, o.get('vx')) for o in self.gone), lead, f.g['x1'] + float(k['x1']) - lead)

    def _mark_blobs(self, view):
        """Which objects may hold more than one lump. The cameras count several (n_est: tracks that ran
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

    def lead_x(self, o):
        """How far forward the centroid of the leading lump in object o can be. One lump: its outline
        centroid. A blob (_mark_blobs): the cameras cannot see which lump is in front or how long it is, only
        that its centroid lies at least nose_min behind the blob's front edge. Until 2026-10-04 a blob was led by
        its own centroid: staging it then ran the lump in front over the edge with no release decided."""
        if o.get('tid') in self.blobs:
            return max(o['cx'], o['x1'] - self.nose_min)
        return o['cx']
