"""Lumps: shape families, material composition, mass -- what a batch is made of.

Where the batch lies on the feed belt is sim/layouts.py; a lump's true pose in the running model is
physics/lumps.py.
"""
import hashlib

import numpy as np
from scipy.spatial import ConvexHull

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
# mudstone / shale / sandstone) 2250-2700 kg/m3 -- the defaults of --coal-density-min/max and
# --gangue-density-min/max (config.py); inherited screening assumptions, not a verified reference range or an
# assay of this feed. The table itself only fixes the order of the endmembers and feeds the two discarded draws
# of make_blocks().
DENSITY_RANGE = {'coal': (1250., 1450.), 'gangue': (2250., 2700.)}

# Dry two-endmember surrogate; see docs/MATERIAL_MODEL.md for evidence and assumptions.
# Fractions below are gangue-endmember MASS fractions within a particle, not ash yields.
COMPOSITION_MODEL = 'coal_gangue_dry_v1'
GANGUE_MASS_RANGE = {'coal': (0., .10), 'middlings': (.10, .80), 'gangue': (.80, 1.)}


def sample_composition(cfg, rng):
    """Composition first, density second. Category probabilities are by particle count.

    Endmembers are coal-rich matrix and mineral-rich gangue rock, not pure carbon/minerals.
    Their internal porosity/impurities are implicit in their effective particle densities.
    No extra pore space, water, chemical assay or resolved internal layers are claimed.
    """
    g, mid = cfg['gangue_fraction'], cfg['middlings_fraction']
    material = str(rng.choice(['coal', 'middlings', 'gangue'], p=[max(0., 1 - g - mid), mid, g]))
    wg = float(rng.uniform(*GANGUE_MASS_RANGE[material]))
    fractions = dict(coal=1 - wg, gangue=wg)
    densities = {name: float(rng.uniform(cfg[name + '_density_min'], cfg[name + '_density_max']))
                 for name in DENSITY_RANGE}
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
    components = {name: sum(b['composition']['component_mass_kg'][name] for b in blocks)
                  for name in ('coal', 'gangue')}
    return dict(total_mass_kg=mass, total_weight_N=mass * 9.81, particle_volume_m3=volume,
                effective_particle_density_kg_m3=mass / volume,
                category_counts=counts, category_count_fractions={k: v / len(blocks) for k, v in counts.items()},
                calibrated=False, material_model=COMPOSITION_MODEL, basis='dry_endmember_mass',
                component_mass_kg=components, mass_fractions={k: v / mass for k, v in components.items()})

def sample_family(rng):
    """Draw a shape family by the uncalibrated screening weights."""
    names = list(FAMILIES)
    w = np.array([FAMILIES[n]['weight'] for n in names])
    return str(rng.choice(names, p=w / w.sum()))


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


def make_blocks(cfg, rng):
    """The batch: shapes from the main stream `rng` (which the layout goes on drawing from), materials from a
    stream of their own.

    Two draws per lump are made and thrown away -- the category and the density of the two-category sampler
    this model replaced on 2026-09-24. They keep the main stream where it was: the same seed gives the same
    shapes and the same layout as every batch run since, which is what makes seeds comparable across
    EXPERIMENTS.md (checks/material_checks.py pins it).
    """
    blocks = []
    for _ in range(cfg['count']):
        fam = sample_family(rng)
        if cfg['layout'] == 'flat':
            fam = 'flat'                        # bench layout 扁平: the same draws, every lump tabular
        binary = 'gangue' if rng.random() < cfg['gangue_fraction'] else 'coal'    # stream compatibility, unused
        size = rng.uniform(cfg['size_min'], cfg['size_max'])
        block = make_block(rng, fam, size, cfg['size_long_max'])
        rng.uniform(*DENSITY_RANGE[binary])                                       # stream compatibility, unused
        blocks.append(block)
    # A separate stream, derived from the main one without consuming it (or reusing its draws).
    seed = hashlib.sha256(repr(rng.bit_generator.state).encode('utf-8')).digest()
    material_rng = np.random.default_rng(np.frombuffer(seed, dtype='<u4'))
    for block in blocks:
        block.update(sample_composition(cfg, material_rng))
        set_mass_properties(block)
    return blocks
