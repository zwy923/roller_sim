"""Feed belt head + belt below: the bench layouts and the transfer record (--layout, --bench; 2026-09-30).

User, 2026-09-30: 先做"给料机头＋下游接带"小试，别把 50 mm 高差定死。用真实滚筒、实际耐磨带和实际煤矸，验证挂边、
提前被下带拖走、两块同时翻落，以及带料停车距离；高差和交接位置做成可调。每批仍用实际场景的 3–5 块，反复试并齐、
相贴、扁平和斜放情况。如果同放两块经常导致下游无法独立计量，就需要改变机械分离方式。

The bench itself is hardware (designs/transfer_trial/DESIGN.md). This module runs the same questions in the
model, to pick the geometries worth building and to set up the instrumentation: the line's feed belt with the
transfer as built (--feeder-head-d, --feeder-tail-d, --feeder-handover, --feeder-step) and one of the bench's
layouts on the feed belt (--layout; they work in a whole-line run too, and --bench ends the run once every
lump lies on the main belt):

  scatter   the default layout of every run;
  aligned   并齐: two or three lumps side by side, fronts level (within 5 mm), long axes along the belt,
            the rest behind them;
  touching  相贴: the batch pressed together -- two abreast and the rest nose to tail behind them, every
            lump 3 mm from a neighbour (3-D, measured on the compiled model);
  flat      扁平: every lump from the flat (tabular) family, scattered;
  oblique   斜放: every lump turned 30-60 deg (either way) to the belt, scattered.

TransferWatch records, in every run, from the true state (the bench's instrumentation, not a controller input):
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
import numpy as np

from .feeder import FRONT_MARGIN, REAR_MARGIN
from .lumps import block_world_vertices

HANG_S = .50          # s: a lump that went, still on the feed belt / drum, that moved less than ...
HANG_MOVE = .01       # m: ... this over the last HANG_S hangs on the head (挂边)
DRAG_S = 1.0          # s: went within this of first touching the belt below, feed belt standing: dragged off
SETTLE_X = .30        # m: 'wholly on the belt below' = rear this far past its flat start
APART = .10           # m: closer than this to the lump before = one item at the station (station.GROUP_GAP)
TOUCH = .003          # m: 3-D gap of 'touching' lumps in the touching layout


def _place(model, data, L, x, y, zlow, yaw):
    """Lump L flat (its generated frame) turned by yaw, plan box centred on (x, y), lowest point at zlow."""
    L.place(data, x, y, 0., yaw)
    data.qvel[L.v:L.v + 6] = 0.
    mujoco.mj_forward(model, data)
    V = block_world_vertices(data, dict(geom=L.geom, local=L.local))
    data.qpos[L.q] += x - (V[:, 0].min() + V[:, 0].max()) / 2
    data.qpos[L.q + 1] += y - (V[:, 1].min() + V[:, 1].max()) / 2
    data.qpos[L.q + 2] += zlow - V[:, 2].min()
    mujoco.mj_forward(model, data)
    return block_world_vertices(data, dict(geom=L.geom, local=L.local))


def _extent(model, data, L, yaw):
    """Plan extents (dx, dy) of lump L turned by yaw."""
    V = _place(model, data, L, 0., -5., 5., yaw)
    return float(np.ptp(V[:, 0])), float(np.ptp(V[:, 1]))


def arrange(cfg, d, model, data, lumps, rng):
    """Lay the batch on the feed belt in the bench layout cfg['layout'] (scatter / rows / flat: keep the layout
    already placed; flat lumps come from make_blocks). Returns a description."""
    case = cfg['layout']
    if case in ('scatter', 'rows', 'flat'):
        return dict(case=case)
    fd = d['feeder']
    h, x1 = fd['step_m'], fd['x1']
    zl = h + cfg.get('feed_drop_height', .006)
    y_lo, y_hi = max(.03, cfg['lane_y'] + .005), cfg['belt_w'] - .03
    front = x1 - FRONT_MARGIN
    info = dict(case=case)
    if case == 'oblique':
        # every lump 30-60 deg to the belt, in rows from the head backwards, 2 cm apart
        order = list(range(len(lumps)))
        rng.shuffle(order)
        yaws = {k: math.radians(rng.uniform(30., 60.)) * (1 if rng.random() < .5 else -1) for k in order}
        _rows(model, data, lumps, order, front, y_lo, y_hi, zl, yaws.get, .02)
        info['yaw_deg'] = {str(k): round(math.degrees(v), 1) for k, v in yaws.items()}
    elif case == 'aligned':
        # the widest set of two or three that fits across, long axes along the belt, fronts level
        widths = {L.k: _extent(model, data, L, 0.) for L in lumps}
        ks = sorted(widths, key=lambda k: widths[k][1])
        n = 3 if len(ks) >= 3 and sum(widths[k][1] for k in ks[:3]) + .06 <= y_hi - y_lo else 2
        row = ks[:n]
        y = y_lo + rng.uniform(0., max(0., (y_hi - y_lo) - sum(widths[k][1] for k in row) - .03 * (n - 1)))
        for k in row:
            dx, dy = widths[k]
            _place(model, data, lumps[k], front - dx / 2 - rng.uniform(0., .005), y + dy / 2, zl, 0.)
            y += dy + rng.uniform(0., .03)
        _rows(model, data, lumps, ks[n:], front - max(widths[k][0] for k in row) - .05, y_lo, y_hi, zl,
              lambda k: 0., .05)
        info.update(abreast=row)
    elif case == 'touching':
        # two abreast at the front, touching; the rest nose to tail behind the first, each touching the one ahead
        ks = [L.k for L in lumps]
        rng.shuffle(ks)
        w = {k: _extent(model, data, lumps[k], 0.) for k in ks}
        a, b = ks[0], ks[1]
        ya = y_lo + .10
        _place(model, data, lumps[a], front - w[a][0] / 2, ya + w[a][1] / 2, zl, 0.)
        _place(model, data, lumps[b], front - w[b][0] / 2, ya + w[a][1] + w[b][1] / 2 + .01, zl, 0.)
        _close(model, data, lumps[a], lumps[b], 1)
        prev = a
        for k in ks[2:]:
            V = block_world_vertices(data, dict(geom=lumps[prev].geom, local=lumps[prev].local))
            _place(model, data, lumps[k], float(V[:, 0].min()) - w[k][0] / 2 - .01,
                   float(V[:, 1].mean()), zl, 0.)
            _close(model, data, lumps[k], lumps[prev], 0)
            prev = k
        info.update(abreast=[a, b], nose_to_tail=[a] + ks[2:])
    x_rear = min(float(block_world_vertices(data, dict(geom=L.geom, local=L.local))[:, 0].min()) for L in lumps)
    if x_rear < fd['x0'] + REAR_MARGIN:
        raise RuntimeError('trial layout %s does not fit on the %.2f m feed belt' % (case, fd['length_m']))
    return info


def _rows(model, data, lumps, ks, front, y_lo, y_hi, zl, yaw, gap):
    """Lay lumps ks in rows across the belt from front backwards, gap apart."""
    x, y, depth = front, y_lo, 0.
    for k in ks:
        dx, dy = _extent(model, data, lumps[k], yaw(k))
        if y + dy > y_hi and y > y_lo:
            x -= depth + gap
            y, depth = y_lo, 0.
        _place(model, data, lumps[k], x - dx / 2, y + dy / 2, zl, yaw(k))
        y += dy + gap
        depth = max(depth, dx)


def _close(model, data, A, B, axis):
    """Slide A along axis (0 = x, 1 = y) toward B until their 3-D gap is TOUCH."""
    ft = np.zeros(6)
    for _ in range(8):
        gap = float(mujoco.mj_geomDistance(model, data, A.geom, B.geom, .5, ft))
        if abs(gap - TOUCH) < .0005:
            break
        VA = block_world_vertices(data, dict(geom=A.geom, local=A.local))
        VB = block_world_vertices(data, dict(geom=B.geom, local=B.local))
        sign = 1. if VA[:, axis].mean() < VB[:, axis].mean() else -1.
        data.qpos[A.q + axis] += sign * (gap - TOUCH)
        mujoco.mj_forward(model, data)


class TransferWatch:
    """The bench's instrumentation (see the module docstring), from the true state every 10 ms."""

    def __init__(self, cfg, d, model, lumps):
        fd = d['feeder']
        self.cfg, self.fd, self.lumps = cfg, fd, lumps
        self.x1, self.x_lower = fd['x1'], fd.get('x_lower', fd['x1'])
        name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ''
        self.feed_g = {g for g in range(model.ngeom) if name(g) in ('feeder', 'fdrum')}
        self.lower_g = {g for g in range(model.ngeom) if name(g) in ('mfloor', 'mtail')}
        self.rec = {L.k: dict(lump=L.k, t_tip=None, t_lower=None, lower_before_tip=False, hang_s=0.,
                              t_last_feed=None, gap_before_m=None) for L in lumps}
        self.stops, self.stop_now = [], None
        self.order = []

    def observe(self, t, data, feeder, belts):
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
                r['feed_speed_at_contact'] = round(belts.feed['f'] * self.fd['speed_m_s'], 4)
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
                                   f0=round(belts.feed['f'], 3), travel_m=0.,
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
                s['travel_m'] += belts.feed['f'] * self.fd['speed_m_s'] * .01
                if belts.feed['f'] < 1e-4:
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
