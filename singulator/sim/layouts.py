"""How the batch lies on the feed belt at t = 0 (--layout).

  scatter   the default: random x, y and yaw in one layer, lumps may touch (plan_scatter). Pure: it needs the
            lumps' shapes only, and it is what every run draws first, so that a seed gives the same batch whatever
            layout follows;
  aligned   并齐: two or three lumps side by side, fronts level (within 5 mm), long axes along the belt,
            the rest behind them;
  touching  相贴: the batch pressed together -- two abreast and the rest nose to tail behind them, every
            lump 3 mm from a neighbour (3-D, measured on the compiled model);
  flat      扁平: every lump from the flat (tabular) family (lumps.make_blocks), scattered;
  oblique   斜放: every lump turned 30-60 deg (either way) to the belt, in rows.

aligned, touching and oblique are the bench trial's layouts (designs/transfer_trial/); arrange() lays them on the
compiled model, because they are measured on the lumps as compiled.
"""
import math

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from ..geom2d import overlap

FRONT_MARGIN = .05   # m: at t = 0 the batch's front edge is this far behind the feed belt's head edge
REAR_MARGIN = .02    # m: ... and its rear edge at least this far in front of the feed belt's tail end
TOUCH = .003         # m: 3-D gap of 'touching' lumps in the touching layout


# ---- scatter: on the shapes alone ---------------------------------------------------------------------------
def footprint(vertices, yaw):
    """Plan-view AABB and lowest z of a block turned by yaw about z."""
    c, s = math.cos(yaw), math.sin(yaw)
    x = c * vertices[:, 0] - s * vertices[:, 1]
    y = s * vertices[:, 0] + c * vertices[:, 1]
    return x.min(), x.max(), y.min(), y.max(), vertices[:, 2].min()


def hull2d(vertices, yaw):
    """Plan-view convex hull of a block turned by yaw about z, as an (n, 2) array. (scipy's hull, not
    geom2d.hull2: the vertex order decides the last digits of the overlap test, and with them which layout a
    seed gives.)"""
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.stack([c * vertices[:, 0] - s * vertices[:, 1], s * vertices[:, 0] + c * vertices[:, 1]], 1)
    return p[ConvexHull(p).vertices]


def _shift_to_start(cfg, blocks, d, placed, what):
    """Move the laid-out batch onto the feed belt: front edge FRONT_MARGIN short of the head edge, the whole batch
    on the feed belt, one step up."""
    front = max(x + footprint(blocks[k]['vertices'], yaw)[1] for k, x, y, z, yaw in placed)
    fd = d['feeder']
    shift = fd['x1'] - FRONT_MARGIN - front
    rear = min(x + footprint(blocks[k]['vertices'], yaw)[0] for k, x, y, z, yaw in placed) + shift
    if rear < fd['x0'] + REAR_MARGIN:
        raise RuntimeError('%s is %.2f m deep and does not fit on the %.2f m feed belt'
                           % (what, front - rear + shift, fd['length_m']))
    lift = fd['step_m']                          # the feed belt top is one step above the main belt
    return [(k, x + shift, y, z + lift, yaw) for k, x, y, z, yaw in placed], rear, front + shift


def plan_scatter(cfg, blocks, rng, d):
    """De-stacked layer with an ARBITRARY distribution (user, 2026-09-20): the batch arrives from a
    chute already broken into one layer, but nothing controls where the lumps sit or which way they
    point. Lumps are dropped at random (x, y, yaw) inside a patch and kept only if their plan-view
    hulls do not overlap, so the layer is single-height by construction and lumps may touch.

    The batch is spread over the whole belt width: the lumps that start below lane_top ride into the lane
    without meeting the face. The patch grows until the batch fits, so a dense batch stays dense rather than
    being silently spread out.
    """
    y_lo, y_hi = max(.03, cfg['lane_y'] + .005), cfg['belt_w'] - .03
    length = cfg['feed_len']
    order = list(range(len(blocks)))
    rng.shuffle(order)
    for _ in range(40):                       # grow the patch until the whole batch fits in one layer
        placed, hulls = [], []
        for k in order:
            for _try in range(400):
                yaw = rng.uniform(-math.pi, math.pi)
                h = hull2d(blocks[k]['vertices'], yaw)
                lo, hi = h.min(0), h.max(0)
                if hi[1] - lo[1] > y_hi - y_lo:
                    continue
                x = rng.uniform(0., length)
                y = rng.uniform(y_lo - lo[1], y_hi - hi[1])
                hh = h + (x, y)
                if any(overlap(hh, o) for o in hulls):
                    continue
                hulls.append(hh)
                fp = footprint(blocks[k]['vertices'], yaw)
                placed.append((k, x, y, cfg['feed_drop_height'] - fp[4], yaw))
                break
            else:
                break
        if len(placed) == len(blocks):
            break
        length += .20
    else:
        raise RuntimeError('could not lay %d lumps out in one layer' % len(blocks))
    return _shift_to_start(cfg, blocks, d, placed, 'feed patch')


# ---- the bench layouts: on the compiled model ---------------------------------------------------------------
def lay_flat(model, data, L, x, y, zlow, yaw=0.):
    """Lump L flat (its generated frame) turned by yaw, plan box centred on (x, y), lowest point at zlow."""
    L.place(data, x, y, 0., yaw)
    data.qvel[L.v:L.v + 6] = 0.
    mujoco.mj_forward(model, data)
    V = L.world(data)
    data.qpos[L.q] += x - (V[:, 0].min() + V[:, 0].max()) / 2
    data.qpos[L.q + 1] += y - (V[:, 1].min() + V[:, 1].max()) / 2
    data.qpos[L.q + 2] += zlow - V[:, 2].min()
    mujoco.mj_forward(model, data)
    return L.world(data)


def _extent(model, data, L, yaw):
    """Plan extents (dx, dy) of lump L turned by yaw."""
    V = lay_flat(model, data, L, 0., -5., 5., yaw)
    return float(np.ptp(V[:, 0])), float(np.ptp(V[:, 1]))


def arrange(cfg, d, model, data, lumps, rng):
    """Lay the batch on the feed belt in the bench layout cfg['layout'] (scatter / flat: keep the layout
    already placed; flat lumps come from make_blocks). Returns a description."""
    case = cfg['layout']
    if case in ('scatter', 'flat'):
        return dict(case=case)
    fd = d['feeder']
    h, x1 = fd['step_m'], fd['x1']
    zl = h + cfg['feed_drop_height']
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
            lay_flat(model, data, lumps[k], front - dx / 2 - rng.uniform(0., .005), y + dy / 2, zl, 0.)
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
        lay_flat(model, data, lumps[a], front - w[a][0] / 2, ya + w[a][1] / 2, zl, 0.)
        lay_flat(model, data, lumps[b], front - w[b][0] / 2, ya + w[a][1] + w[b][1] / 2 + .01, zl, 0.)
        _close(model, data, lumps[a], lumps[b], 1)
        prev, laid, refit = a, [lumps[a], lumps[b]], []
        for k in ks[2:]:
            V = lumps[prev].world(data)
            lay_flat(model, data, lumps[k], float(V[:, 0].min()) - w[k][0] / 2 - .01,
                   float(V[:, 1].mean()), zl, 0.)
            moved = _inside(model, data, lumps[k], y_lo, y_hi)
            _close(model, data, lumps[k], lumps[prev], 0)
            if _clear(model, data, lumps[k], laid) or moved:
                refit.append(k)
            laid.append(lumps[k])
            prev = k
        info.update(abreast=[a, b], nose_to_tail=[a] + ks[2:], **(dict(refit=refit) if refit else {}))
    x_rear = min(float(L.world(data)[:, 0].min()) for L in lumps)
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
        lay_flat(model, data, lumps[k], x - dx / 2, y + dy / 2, zl, yaw(k))
        y += dy + gap
        depth = max(depth, dx)


def _inside(model, data, A, y_lo, y_hi):
    """Shift lump A across the belt until its plan lies within y_lo .. y_hi. A lump laid nose to tail behind a
    narrower one, centred on it, could reach past the low-side skirt (touching 7277: 8.5 cm into it). True if it
    moved; a lump already inside is not touched."""
    V = A.world(data)
    lo, hi = float(V[:, 1].min()), float(V[:, 1].max())
    dy = y_lo - lo if lo < y_lo else y_hi - hi if hi > y_hi else 0.
    if dy:
        data.qpos[A.q + 1] += dy
        mujoco.mj_forward(model, data)
    return bool(dy)


def _clear(model, data, A, laid):
    """Lump A, just closed onto the lump ahead of it (_close), may reach into another one already laid: the one
    abreast when that is the longer and A the wider (touching 7134: 11.5 cm into it; 55 seeds in 2000 laid lumps
    more than 1 mm into each other, which the line refused to run). Back A off along the belt until it is clear of
    every lump laid, then slide it forward until it is TOUCH from the first it meets. Each step forward is the gap
    left less TOUCH, and no lump is nearer than that gap, so A never reaches into one. True if it moved; a lump
    clear of all of them is not touched, so a layout that did not overlap is laid as before."""
    ft = np.zeros(6)
    gap = lambda: min(float(mujoco.mj_geomDistance(model, data, A.geom, B.geom, .5, ft)) for B in laid)
    if gap() >= 0.:
        return False
    behind = min(float(B.world(data)[:, 0].min()) for B in laid)
    data.qpos[A.q] -= float(A.world(data)[:, 0].max()) - behind + .01
    mujoco.mj_forward(model, data)
    for _ in range(60):
        step = gap() - TOUCH
        if step < .0005:
            break
        data.qpos[A.q] += step
        mujoco.mj_forward(model, data)
    return True


def _close(model, data, A, B, axis):
    """Slide A along axis (0 = x, 1 = y) toward B until their 3-D gap is TOUCH."""
    ft = np.zeros(6)
    for _ in range(8):
        gap = float(mujoco.mj_geomDistance(model, data, A.geom, B.geom, .5, ft))
        if abs(gap - TOUCH) < .0005:
            break
        VA = A.world(data)
        VB = B.world(data)
        sign = 1. if VA[:, axis].mean() < VB[:, axis].mean() else -1.
        data.qpos[A.q + axis] += sign * (gap - TOUCH)
        mujoco.mj_forward(model, data)
