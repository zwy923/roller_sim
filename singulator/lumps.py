"""Lumps: shape families, material draws, the feed layer, and outline geometry of a placed lump."""
import hashlib
import math

import numpy as np
from scipy.spatial import ConvexHull

from . import feeder

# Shape families. 'size' is the SCREEN SIZE of a screened size class -- the feed is "300-500 mm 粒级原煤"
# (user, 2026-09-20). A square aperture passes whatever fits through the hole, so it bounds the lump's
# INTERMEDIATE axis; an elongated lump goes through lengthwise and its LONG axis follows from the family
# ratio, long = size / b, up to 2.2 x size. The long axis is then capped at `long_max` (--size-long-max,
# 0.50 m, user 2026-09-20) to stand for the upstream picking / breaking that
# keeps the longest slivers out of the feed; pass None for an uncapped screen class.
#
# History, because it decides which archived runs are comparable: 2026-09-17 screen-aperture convention
# (what this is again); 2026-09-18 briefly "size is the MAXIMUM dimension, 长宽高都不该超过 500 mm", which
# deleted exactly the 0.6-1.1 m tail that makes a third of the stalled lumps in runs/n60s* -- every
# 100-seed archive predates that change and is valid for the screen class.
#
# Synthetic shape mixture for screening; the weights and ratios have no verified source or plant
# calibration. Axis spans are a size proxy, not a simulation of passage through a square sieve.
FAMILIES = {
    'round':     dict(b=(.80, .95), c=(.70, .90), points=48, jitter=.04, weight=.30),   # equant
    'angular':   dict(b=(.70, .90), c=(.55, .80), points=12, jitter=.12, weight=.35),   # blocky, fresh breaks
    'flat':      dict(b=(.75, .95), c=(.25, .45), points=40, jitter=.05, weight=.20),   # tabular / platy
    'elongated': dict(b=(.45, .65), c=(.35, .60), points=40, jitter=.05, weight=.10),   # prismatic
    'bladed':    dict(b=(.45, .65), c=(.20, .35), points=40, jitter=.06, weight=.05),   # flat AND long
}
# Lump (particle) density, not bulk: ROM bituminous coal 1250-1450, coal-measure gangue (carbonaceous
# mudstone / shale / sandstone) 2250-2700 kg/m3. Sampled per lump, because real feed is not one value.
# These are inherited screening assumptions, not a verified reference range or an assay of this feed.
DENSITY_RANGE = {'coal': (1250., 1450.), 'gangue': (2250., 2700.)}

# Dry two-endmember surrogate; see MATERIAL_MODEL.md for evidence and assumptions.
# Fractions below are gangue-endmember MASS fractions within a particle, not ash yields.
COMPOSITION_MODEL = 'coal_gangue_dry_v1'
GANGUE_MASS_RANGE = {'coal': (0., .10), 'middlings': (.10, .80), 'gangue': (.80, 1.)}


def validate_material(cfg):
    model = cfg.get('material_model', 'legacy_binary')
    if model not in ('composition', 'legacy_binary'):
        raise ValueError('unknown material_model: %s' % model)
    if model == 'composition':
        gangue, middlings = cfg['gangue_fraction'], cfg.get('middlings_fraction', .25)
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (gangue, middlings)) or gangue + middlings > 1:
            raise ValueError('gangue_fraction and middlings_fraction must be in [0, 1] and sum <= 1')
        for name in ('coal', 'gangue'):
            lo = cfg.get(name + '_density_min', DENSITY_RANGE[name][0])
            hi = cfg.get(name + '_density_max', DENSITY_RANGE[name][1])
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                raise ValueError('%s density bounds must be finite, positive and ordered' % name)


def sample_composition(cfg, rng):
    """Composition first, density second. Category probabilities are by particle count.

    Endmembers are coal-rich matrix and mineral-rich gangue rock, not pure carbon/minerals.
    Their internal porosity/impurities are implicit in their effective particle densities.
    No extra pore space, water, chemical assay or resolved internal layers are claimed.
    """
    g, mid = cfg['gangue_fraction'], cfg.get('middlings_fraction', .25)
    material = str(rng.choice(['coal', 'middlings', 'gangue'], p=[max(0., 1 - g - mid), mid, g]))
    wg = float(rng.uniform(*GANGUE_MASS_RANGE[material]))
    fractions = dict(coal=1 - wg, gangue=wg)
    densities = {name: float(rng.uniform(cfg.get(name + '_density_min', limits[0]),
                                        cfg.get(name + '_density_max', limits[1])))
                 for name, limits in DENSITY_RANGE.items()}
    # Additive component volumes: MASS fractions require a harmonic, not arithmetic mean.
    specific_volume = {name: fractions[name] / densities[name] for name in fractions}
    density = 1 / sum(specific_volume.values())
    volume_fractions = {name: v * density for name, v in specific_volume.items()}
    vg = volume_fractions['gangue']
    rgba = [(1 - vg) * c + vg * g for c, g in zip((.20, .20, .23), (.66, .58, .46))] + [1.]
    return dict(material=material, material_model=COMPOSITION_MODEL, density=density, rgba=rgba,
                composition=dict(basis='dry_endmember_mass', calibrated=False,
                                 mass_fractions=fractions, volume_fractions=volume_fractions,
                                 endmember_density_kg_m3=densities,
                                 spatial_model='homogeneous_equivalent'))


def set_mass_properties(block, mass_kg=None):
    """Use hull volume before compilation, compiled body mass afterwards (same physical volume)."""
    mass = float(mass_kg) if mass_kg is not None else float(ConvexHull(block['vertices']).volume * block['density'])
    block.update(mass_kg=mass, volume_m3=mass / block['density'], weight_N=mass * 9.81)
    if 'composition' in block:
        block['composition']['component_mass_kg'] = {
            name: mass * fraction for name, fraction in block['composition']['mass_fractions'].items()}


def batch_material_summary(blocks):
    mass = sum(b['mass_kg'] for b in blocks)
    volume = sum(b['volume_m3'] for b in blocks)
    counts = {name: sum(b['material'] == name for b in blocks) for name in ('coal', 'middlings', 'gangue')}
    out = dict(total_mass_kg=mass, total_weight_N=mass * 9.81, particle_volume_m3=volume,
               effective_particle_density_kg_m3=mass / volume,
               category_counts=counts, category_count_fractions={k: v / len(blocks) for k, v in counts.items()},
               calibrated=False)
    if all('composition' in b for b in blocks):
        components = {name: sum(b['composition']['component_mass_kg'][name] for b in blocks)
                      for name in ('coal', 'gangue')}
        out.update(material_model=COMPOSITION_MODEL, basis='dry_endmember_mass',
                   component_mass_kg=components, mass_fractions={k: v / mass for k, v in components.items()})
    else:
        out['material_model'] = 'legacy_binary'
    return out

# Sliding friction by contact pair: inherited, uncalibrated assumptions. No material-dependent
# sampling or measured static/kinetic distinction is currently justified.
FRICTION = dict(block_belt=.55,     # coal or rock on a rubber belt cover, 0.45-0.70
                block_steel=.45,    # coal on mild steel, 0.35-0.60 (wall friction angle 20-31 deg)
                block_block=.60,    # coal on coal / rock on rock sliding, 0.50-0.80
                lined_steel=.20)    # the same plate lined with UHMW-PE, 0.15-0.25


def sample_family(rng):
    """Draw a shape family by the uncalibrated screening weights."""
    names = list(FAMILIES)
    w = np.array([FAMILIES[n]['weight'] for n in names])
    return str(rng.choice(names, p=w / w.sum()))


def sample_density(rng, material):
    lo, hi = DENSITY_RANGE[material]
    return float(rng.uniform(lo, hi))


def make_block(rng, family, size, long_max=None):
    """Convex point cloud with long/mid/short axes along body x/y/z (lying flat).

    `size` is the SCREEN SIZE, i.e. the square aperture the lump passed, so it fixes the INTERMEDIATE
    axis. The family sets the ratios (mid = b x long, short = c x long), so the long axis is size / b --
    1.05 x size for an equant lump, up to 2.2 x for a bladed one. `long_max` (m, None = uncapped) then
    trims the long axis for upstream picking/breaking; the short axis follows the trimmed long, so a
    capped lump is a less elongated lump of the same size class, not a smaller one.
    """
    f = FAMILIES[family]
    rb, rc = rng.uniform(*f['b']), rng.uniform(*f['c'])
    long = size / rb if long_max is None else min(size / rb, float(long_max))
    axes = np.sort(np.array([long, size, rc * long]))[::-1]   # long, mid, short: the lump lies flat
    u = rng.normal(size=(f['points'], 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    pts = u * axes / 2 * (1 + f['jitter'] * rng.uniform(-1, 1, size=(f['points'], 1)))
    pts *= axes / (pts.max(0) - pts.min(0))      # affine rescale keeps the hull convex
    pts -= (pts.max(0) + pts.min(0)) / 2
    return dict(family=family, size=float(size), size_definition='screen_intermediate_axis',
                long_axis_cap_m=(float(long_max) if long_max is not None else None),
                axes=axes.tolist(), vertices=pts)


def footprint(vertices, yaw):
    """Plan-view AABB and lowest z of a block turned by yaw about z."""
    c, s = math.cos(yaw), math.sin(yaw)
    x = c * vertices[:, 0] - s * vertices[:, 1]
    y = s * vertices[:, 0] + c * vertices[:, 1]
    return x.min(), x.max(), y.min(), y.max(), vertices[:, 2].min()


def exit_cut(V, edges, x):
    """y-interval of a convex block's cross-section with the vertical plane at x, or None if it does not reach it."""
    P, Q = V[edges[:, 0]], V[edges[:, 1]]
    dp, dq = P[:, 0] - x, Q[:, 0] - x
    m = (dp * dq <= 0) & (dp != dq)
    if not m.any():
        return None
    y = P[m, 1] + dp[m] / (dp[m] - dq[m]) * (Q[m, 1] - P[m, 1])
    return float(y.min()), float(y.max())


def block_world_vertices(data, b):
    """World vertices of a lump. b['local'] are the COMPILED mesh vertices, which MuJoCo re-centres on the
    centroid and turns onto the principal axes, compensating in the geom's own pos/quat -- so they must
    be placed with the geom frame. Placing them with the body frame (as before 2026-09-22) was off by up
    to 0.18 m on seed 81 and corrupted every observation built on it."""
    return data.geom_xpos[b['geom']] + b['local'] @ data.geom_xmat[b['geom']].reshape(3, 3).T


def make_blocks(cfg, rng):
    validate_material(cfg)
    blocks = []
    for _ in range(cfg['count']):
        fam = sample_family(rng)
        if cfg.get('layout') == 'flat':
            fam = 'flat'                        # bench layout 扁平: the same draws, every lump tabular
        mat = 'gangue' if rng.random() < cfg['gangue_fraction'] else 'coal'
        size = rng.uniform(cfg['size_min'], cfg['size_max'])
        blocks.append(dict(make_block(rng, fam, size, cfg['size_long_max']), material=mat,
                           density=sample_density(rng, mat)))
    if cfg.get('material_model', 'legacy_binary') == 'composition':
        # Derive a separate stream without consuming the layout stream (or reusing its draws).
        # Preserve old vertices and subsequent layout for the same seed and gangue_fraction.
        seed = hashlib.sha256(repr(rng.bit_generator.state).encode('utf-8')).digest()
        material_rng = np.random.default_rng(np.frombuffer(seed, dtype='<u4'))
        for block in blocks:
            block.update(sample_composition(cfg, material_rng))
    for block in blocks:
        set_mass_properties(block)
    return blocks


def hull2d(vertices, yaw):
    """Plan-view convex hull of a block turned by yaw about z, as an (n, 2) array."""
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.stack([c * vertices[:, 0] - s * vertices[:, 1], s * vertices[:, 0] + c * vertices[:, 1]], 1)
    return p[ConvexHull(p).vertices]


def overlap2d(a, b, margin=.005):
    """Separating-axis test on two convex polygons, inflated by margin. True = they would overlap."""
    for poly in (a, b):
        e = np.roll(poly, -1, 0) - poly
        n = np.stack([-e[:, 1], e[:, 0]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        pa, pb = a @ n.T, b @ n.T
        if (pa.max(0) + margin < pb.min(0)).any() or (pb.max(0) + margin < pa.min(0)).any():
            return False
    return True


def _shift_to_start(cfg, blocks, d, placed, what):
    """Move the laid-out batch onto the feed belt: front edge FRONT_MARGIN short of the head edge, the whole batch
    on the feed belt, one step up."""
    front = max(x + footprint(blocks[k]['vertices'], yaw)[1] for k, x, y, z, yaw in placed)
    fd = d['feeder']
    shift = fd['x1'] - feeder.FRONT_MARGIN - front
    rear = min(x + footprint(blocks[k]['vertices'], yaw)[0] for k, x, y, z, yaw in placed) + shift
    if rear < fd['x0'] + feeder.REAR_MARGIN:
        raise RuntimeError('%s is %.2f m deep and does not fit on the %.2f m feed belt'
                           % (what, front - rear + shift, fd['length_m']))
    lift = fd['step_m']                          # the feed belt top is one step above the main belt
    return [(k, x + shift, y, z + lift, yaw) for k, x, y, z, yaw in placed], rear, front + shift


def plan_scatter(cfg, blocks, rng, d):
    """De-stacked layer with an ARBITRARY distribution (user, 2026-09-20): the batch arrives from a
    chute already broken into one layer, but nothing controls where the lumps sit or which way they
    point. Lumps are dropped at random (x, y, yaw) inside a patch and kept only if their plan-view
    hulls do not overlap, so the layer is single-height by construction and lumps may touch.

    plan_rows instead lays the batch out in tidy rows with 0.02 m side gaps and 0.10 m row gaps, which
    hands the machine a regular input it will not see in service. The patch grows until the batch fits,
    so a dense batch stays dense rather than being silently spread out.
    """
    y_lo = d['lane_top'] + .01 if cfg['feed_band'] == 'plough' else max(.03, cfg['lane_y'] + .005)
    y_hi = cfg['belt_w'] - .01 if cfg['feed_band'] == 'plough' else cfg['belt_w'] - .03
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
                if any(overlap2d(hh, o) for o in hulls):
                    continue
                hulls.append(hh)
                fp = footprint(blocks[k]['vertices'], yaw)
                placed.append((k, x, y, cfg.get('feed_drop_height', .006) - fp[4], yaw))
                break
            else:
                break
        if len(placed) == len(blocks):
            break
        length += .20
    else:
        raise RuntimeError('could not lay %d lumps out in one layer' % len(blocks))
    return _shift_to_start(cfg, blocks, d, placed, 'feed patch')


def plan_rows(cfg, blocks, rng, d):
    """De-stacked layer of side-by-side rows across the belt: the abreast input the S tests used.

    Rows are built back from the plough, so the whole layer sits clear of the face at t = 0 whatever
    its depth turns out to be; the belt then carries it in.

    --feed-band plough confines the layer to the band the face can actually reach (y above lane_top).
    The face only ever pushes material DOWN, so a lump placed above lane_top is guaranteed to meet it;
    a lump placed below is already in the lane band and rides through untouched -- which is what the
    old full-belt layer did to 318 of 600 lumps in the 100-condition screening (53 %). Do NOT try to
    enforce the band with a guide wall on the belt instead: a wall along y = lane_top closes a wedge
    between itself and the face (0.55 m -> 0) whose apex has no outlet, and the pilot arches solid.
    """
    y_lo = d['lane_top'] + .01 if cfg['feed_band'] == 'plough' else max(.03, cfg['lane_y'] + .005)
    y_hi = cfg['belt_w'] - .01 if cfg['feed_band'] == 'plough' else cfg['belt_w'] - .03
    band = y_hi - y_lo
    placed, x_rear, order = [], 0., list(range(len(blocks)))
    rng.shuffle(order)
    i = 0
    while i < len(order):
        row, used = [], 0.
        while i < len(order) and len(row) < 3:
            k = order[i]
            for attempt in range(60):
                yaw = rng.uniform(-math.pi, math.pi) if attempt < 40 else rng.uniform(-.08, .08)
                fp = footprint(blocks[k]['vertices'], yaw)
                w = fp[3] - fp[2]
                if used + w + .02 * len(row) <= band:
                    row.append((k, yaw, w, fp))
                    used += w
                    break
            else:
                break
            i += 1
        if not row:
            raise RuntimeError('nothing fits across the belt')
        free = band - used - .02 * (len(row) - 1)
        y = y_lo + rng.uniform(0., max(free, 0.))
        depth = 0.
        for j, (k, yaw, w, fp) in enumerate(row):
            dx = rng.uniform(0., cfg['row_stagger'])
            placed.append((k, x_rear + dx - fp[0], y - fp[2], cfg.get('feed_drop_height', .006) - fp[4], yaw))
            depth = max(depth, fp[1] - fp[0] + dx)
            y += w + .02
        x_rear += depth + cfg['row_gap']
    return _shift_to_start(cfg, blocks, d, placed, 'feed layer')
