"""The feed head's transfer record: what the bench trial measures, taken from the true state in every run.

User, 2026-09-30: 先做"给料机头＋下游接带"小试，别把 50 mm 高差定死。用真实滚筒、实际耐磨带和实际煤矸，验证挂边、
提前被下带拖走、两块同时翻落，以及带料停车距离；高差和交接位置做成可调。每批仍用实际场景的 3–5 块，反复试并齐、
相贴、扁平和斜放情况。如果同放两块经常导致下游无法独立计量，就需要改变机械分离方式。

The bench itself is hardware (designs/transfer_trial/DESIGN.md). The model runs the same questions to pick the
geometries worth building and to set up the instrumentation: the line's feed belt with the transfer as built
(--feeder-head-d, --feeder-tail-d, --feeder-handover, --feeder-step) and one of the bench's layouts on the feed belt
(--layout, sim/layouts.py; --bench ends the run once every lump lies on the main belt).

TransferWatch is that instrumentation, not a controller input. It records:
  tip        a lump's centroid passes the head tangent x1 (it goes);
  hang       挂边: a lump that went still rests on the feed belt or its head drum and has moved less than
             HANG_MOVE over HANG_S (hang_stuck_s: how long); hang_s is how long it touched the head after it went;
  early      提前被下带拖走: a lump touches the belt below while its centroid is still behind x1 (the belt below
             has hold of a lump the feed belt still carries), and 'dragged' when it also went within DRAG_S
             of that contact while the feed belt stood still;
  double     两块同时翻落: a release in which two or more lumps went, with the time between them; 'in_staging'
             lists the lumps that went only once the belt had been run again for staging (no release decided);
  stops      带料停车距离: every feed belt stop -- belt travel from the stop command to rest, and how far the
             lumps still on the feed belt slid meanwhile. A stop the feeder ends before the belt is at rest
             (staging, the next release or a jog runs it again) has no stop distance: it is listed apart
             (stops_interrupted) and kept out of the maxima;
  apart      下游能否独立计量: for each lump after the first, the clear gap to the lump before it on the belt
             below once it lies wholly on it (rear SETTLE_X past the belt start); under APART it would reach
             the measuring belt with the one before -- not measurable alone.
"""
import math

import mujoco

from ..machine import feed_belt, plough

HANG_S = .50          # s: a lump that went, still on the feed belt / drum, that moved less than ...
HANG_MOVE = .01       # m: ... this over the last HANG_S hangs on the head (挂边)
DRAG_S = 1.0          # s: went within this of first touching the belt below, feed belt standing: dragged off
SETTLE_X = .30        # m: 'wholly on the belt below' = rear this far past its flat start
APART = .10           # m: closer than this to the lump before = one item at the station (station.GROUP_GAP)




class TransferWatch:
    """The bench's instrumentation (see the module docstring), from the true state every 10 ms."""

    def __init__(self, cfg, d, model, lumps):
        fd = d['feeder']
        self.cfg, self.fd, self.lumps = cfg, fd, lumps
        self.x1, self.x_lower = fd['x1'], fd.get('x_lower', fd['x1'])
        name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ''
        self.feed_g = {g for g in range(model.ngeom) if name(g) in (feed_belt.BODY, feed_belt.DRUM)}
        self.lower_g = {g for g in range(model.ngeom) if name(g) in (plough.MAIN_BELT, plough.TAIL)}
        self.rec = {L.k: dict(lump=L.k, t_tip=None, t_lower=None, lower_before_tip=False, hang_s=0.,
                              t_last_feed=None, gap_before_m=None) for L in lumps}
        self.stops, self.stop_now = [], None
        self.order = []

    def observe(self, t, data, feeder, feed_f):
        """One sample. feeder: the feed belt control (its releases and stops); feed_f: the feed belt drive's speed
        factor now."""
        touch_feed, touch_lower = set(), set()
        gl = {L.geom: L.k for L in self.lumps}
        for i in range(data.ncon):
            c = data.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in gl:
                    if b in self.feed_g:
                        touch_feed.add(gl[a])
                    elif b in self.lower_g:
                        touch_lower.add(gl[a])
        for L in self.lumps:
            r = self.rec[L.k]
            if L.state not in ('on_belt', 'passed'):
                continue
            cx = float(data.xipos[L.body][0])
            if r['t_tip'] is None and cx > self.x1:
                r['t_tip'] = round(t, 3)
                self.order.append(L.k)
            if L.k in touch_lower and r['t_lower'] is None:
                r['t_lower'] = round(t, 3)
                r['lower_before_tip'] = r['t_tip'] is None
                r['feed_speed_at_contact'] = round(feed_f * self.fd['speed_m_s'], 4)
            if r['t_tip'] is not None and L.k in touch_feed:
                r['t_last_feed'] = round(t, 3)
                # hang: it went, still rests on the feed belt or its head drum, and has not moved HANG_MOVE over HANG_S
                h = r.setdefault('_h', [])
                h.append((t, cx))
                while len(h) > 2 and h[1][0] <= t - HANG_S + 1e-9:
                    h.pop(0)
                if t - r['t_tip'] >= HANG_S - 1e-9 and h[0][0] <= t - HANG_S + .02 and h[-1][1] - h[0][1] < HANG_MOVE:
                    r['hang_stuck_s'] = round(r.get('hang_stuck_s', 0.) + .01, 3)
            else:
                r.pop('_h', None)
            if r['t_tip'] is not None and r['gap_before_m'] is None and float(L.V[:, 0].min()) > self.x_lower + SETTLE_X:
                i = self.order.index(L.k)
                if i > 0:
                    P = self.lumps[self.order[i - 1]]
                    r['gap_before_m'] = round(float(P.V[:, 0].min()) - float(L.V[:, 0].max()), 4)
                    r['before'] = P.k
                else:
                    r['gap_before_m'] = float('inf')
        # stops of the feed belt: travel from the command to rest, and the slide of the lumps still on it
        if feeder.releases and feeder.releases[-1]['t_stop_s'] is not None and (
                not self.stops or self.stops[-1]['release'] != feeder.releases[-1]['release']):
            rel = feeder.releases[-1]
            on = [L.k for L in self.lumps if L.state == 'on_belt' and float(data.xipos[L.body][0]) < self.x1]
            self.stops.append(dict(release=rel['release'], why=rel['stop'], t_s=rel['t_stop_s'],
                                   f0=round(feed_f, 3), travel_m=0.,
                                   x0={k: float(data.xipos[self.lumps[k].body][0]) for k in on}, done=False))
        for s in self.stops:
            if not s['done']:
                if feeder.goal > 0.:
                    # the feeder runs the belt again (staging, the next release, a jog) before it came to rest: this
                    # stop has no stop distance. Until 2026-10-04 the record stayed open through the run that
                    # followed and booked all of it as the stop (seed 392: 0.756 m of 'stop travel' over 8.16 s)
                    s.update(done=round(t, 3), travel_m=round(s['travel_m'], 4), stop_time_s=round(t - s['t_s'], 3),
                             interrupted=('jog' if feeder.jog is not None else 'staging' if feeder.staging
                                          else 'release'))
                    del s['x0']
                    continue
                s['travel_m'] += feed_f * self.fd['speed_m_s'] * .01
                if feed_f < 1e-4:
                    s['done'] = round(t, 3)
                    s['lump_slide_m'] = round(max((float(data.xipos[self.lumps[k].body][0]) - x
                                                  for k, x in s['x0'].items()), default=0.), 4)
                    s['travel_m'] = round(s['travel_m'], 4)
                    s['stop_time_s'] = round(t - s['t_s'], 3)
                    del s['x0']

    def report(self, feeder):
        recs = list(self.rec.values())
        for r in recs:
            r['hang_s'] = round(max(0., (r['t_last_feed'] or 0.) - (r['t_tip'] or 0.)), 3) if r['t_tip'] else None
            r['hang'] = r.get('hang_stuck_s', 0.) > 0.
            r.pop('_h', None)
            r['dragged'] = bool(r['lower_before_tip'] and r['t_tip'] is not None
                                and r['t_tip'] - r['t_lower'] <= DRAG_S and r.get('feed_speed_at_contact', 1.) < 1e-3)
        who = 'lumps' if feeder.vision else 'members'       # vision: the true lumps that went (verification)
        late = 'lumps_after_stop' if feeder.vision else 'after_stop'   # ... and those that went after the stop
        rels = [dict(release=x['release'], lumps=list(x[who]), after_stop=list(x.get(late, [])),
                     in_staging=list(x.get('lumps_in_staging', [])), jogs=x['jogs'], stop=x['stop'],
                     tip_spread_s=(round(max(self.rec[k]['t_tip'] for k in x[who])
                                         - min(self.rec[k]['t_tip'] for k in x[who]), 3)
                                   if len(x[who]) > 1 and all(self.rec[k]['t_tip'] for k in x[who]) else None))
                for x in feeder.releases if x.get(who)]
        gaps = [r['gap_before_m'] for r in recs if r['gap_before_m'] is not None and math.isfinite(r['gap_before_m'])]
        close = [r for r in recs if r['gap_before_m'] is not None and r['gap_before_m'] < APART]
        stops = [s for s in self.stops if s.get('done') and not s.get('interrupted')]    # the belt came to rest
        cut = [s for s in self.stops if s.get('interrupted')]                             # run again before that
        fin = lambda v: None if v is None or not math.isfinite(v) else v
        return dict(case=self.cfg['layout'], head=self.fd.get('head'), step_m=self.fd['step_m'],
                    x1=self.x1, x_lower=self.x_lower,
                    lumps=[{k: fin(v) if isinstance(v, float) else v for k, v in r.items()} for r in recs],
                    releases=rels, stops=stops, stops_interrupted=cut,
                    summary=dict(lumps=len(recs), went=sum(r['t_tip'] is not None for r in recs),
                                 releases=len(rels), double_releases=sum(len(x['lumps']) > 1 for x in rels),
                                 lumps_in_doubles=sum(len(x['lumps']) for x in rels if len(x['lumps']) > 1),
                                 after_stop=sum(len(x['after_stop']) for x in rels),
                                 in_staging=sum(len(x['in_staging']) for x in rels),
                                 hangs=sum(r['hang'] for r in recs), jogs=sum(x['jogs'] for x in rels),
                                 early_contacts=sum(r['lower_before_tip'] for r in recs),
                                 dragged=sum(r['dragged'] for r in recs),
                                 not_apart=len(close), gap_min_m=round(min(gaps), 3) if gaps else None,
                                 stop_travel_max_m=max((s['travel_m'] for s in stops), default=None),
                                 lump_slide_max_m=max((s['lump_slide_m'] for s in stops), default=None),
                                 stops_interrupted=len(cut)),
                    rules=dict(hang_s=HANG_S, drag_s=DRAG_S, settle_x_m=SETTLE_X, apart_m=APART))
