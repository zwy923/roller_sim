"""Buffer belt, measuring belt and the docking of the flip separator: the second half of the line as hardware.

    lane -> main belt head edge -> buffer belt B (one step down) -> measuring belt M (weigh + volume) -> separator

  * B: station_w wide, buffer_len long, its top station_step BELOW the main belt, from the main belt's head edge,
    running at buffer_speed; a lump tips onto it once its centroid passes that edge. Staging beam S3 far enough
    before B's far end for B to stop a lead that cuts it (STOP_BACK at
    least), STOP_Z above it;
  * M: same width and height as B, measure_len long, a flat joint with B. Load cells carry M and its skirts; the
    volume scanner (three depth heads, machine/sensors.py) over it;
  * the flip separator (machine/separator.py), as wide as M, with side walls (separator_wall high) riding on the
    plate, behind it; the plate-zone camera confirms that its path is empty.

Geometry is pure (no MuJoCo). The control of all three is control/station.py; what weighs and scans is sensing/.
"""
import math

import numpy as np

from ..config import SAMPLE_S
from . import sensors, separator
from .parts import SKIRT, plate_belt, skirt

BUFFER, MEASURE = 'bbelt', 'mbelt'       # the belt plates (joints 'bbeltj', 'mbeltj'); skirts 'bskirt_*', 'mskirt_*'
RAMP_S = .30          # s: start / stop ramp of the B and M drives and of the held section upstream
FORCE_MAX = 2000.     # N: B and M drive limits, assumed
CLEAR = .02           # m: a lump is wholly on a belt once its rear is this far past the belt's start
FRONT_MARGIN = .05    # m: a lump at rest on M keeps its front this far short of M's head edge
STOP_BACK = .15       # m: the staging beam is at least this far before B's far end (more when B runs fast:
                      # its stop distance plus the camera latency plus STAGE_ROOM) ...
STOP_Z = .05          # ... this high above B (a flat lump is at least 6 cm thick)
STAGE_ROOM = .05      # m: a staged lead stops at least this far short of B's far end
PLATE_ZONE_DEPTH = .45  # m: the camera's plate zone extends at least this far below the inlet, plus 5 cm
GAP = .02             # m: M's head edge -> plate inlet
DROP = .02            # m: M's top -> rib tops at the plate inlet


def geometry(cfg, x1, y_c):
    """B from the main belt head edge x1, M after it, the separator after that, all centred on the lane
    centre line y_c."""
    for k in ('station_w', 'buffer_len', 'measure_len', 'station_step', 'station_speed', 'buffer_speed', 'scan_s',
              'sort_density', 'separator_swing_s', 'separator_friction', 'separator_wall'):
        if not math.isfinite(cfg[k]) or cfg[k] < 0 or (cfg[k] == 0 and k != 'separator_wall'):
            raise ValueError('--%s must be finite and positive' % k.replace('_', '-'))
    if cfg['station_w'] <= cfg['lane_w']:
        raise ValueError('--station-w %.3f m must exceed the lane width %.3f m: the lane hands over into the buffer'
                         % (cfg['station_w'], cfg['lane_w']))
    span = math.hypot(cfg['size_long_max'], cfg['size_max'])       # plan diagonal of a flat-lying lump, bound
    stop = cfg['station_speed'] * (RAMP_S / 2 + SAMPLE_S)
    stop_back = max(STOP_BACK, cfg['buffer_speed'] * (RAMP_S / 2 + SAMPLE_S + sensors.LATENCY_S) + STAGE_ROOM)
    need = dict(buffer_len=CLEAR + span + stop_back, measure_len=CLEAR + stop + span + FRONT_MARGIN)
    for k, v in need.items():
        if cfg[k] < v:
            raise ValueError('--%s %.3f m cannot hold one lump (%.3f m plan extent) wholly: needs %.3f m'
                             % (k.replace('_', '-'), cfg[k], span, v))
    top = -cfg['station_step']
    b0, b1 = x1, x1 + cfg['buffer_len']
    m0, m1 = b1, b1 + cfg['measure_len']
    sep = separator.for_width(cfg['station_w'], side_wall_height=cfg['separator_wall'])
    origin = np.array([m1 + GAP, y_c, top - DROP - sep.rib_height - sep.inlet_height])
    world = lambda P: [round(float(a + b), 4) for a, b in zip(origin, P)]
    pivot = world(separator.layout(sep)[0])
    closed, opened = separator.state(sep, sep.closed_deg), separator.state(sep, sep.open_deg)
    mu = cfg['separator_friction']
    inlet = world(closed['inlet'])
    out_c, in_o = world(closed['outlet']), world(opened['inlet'])
    # the plate's path: where a lump sent over the plate may still be while the plate must not move
    zone = ((m1, y_c - cfg['station_w'] / 2 - .10, min(inlet[2] - PLATE_ZONE_DEPTH, out_c[2]) - .05),
            (out_c[0] + .10, y_c + cfg['station_w'] / 2 + .10, inlet[2] + .50))
    return dict(sep=sep, width_m=cfg['station_w'], y_c=y_c, top_z=top, step_m=cfg['station_step'],
                speed_m_s=cfg['station_speed'], buffer_speed_m_s=cfg['buffer_speed'], ramp_s=RAMP_S,
                buffer=dict(x0=b0, x1=b1, length_m=cfg['buffer_len'], speed_m_s=cfg['buffer_speed']),
                measure=dict(x0=m0, x1=m1, length_m=cfg['measure_len'], scan_s=cfg['scan_s']),
                beam_stop=dict(x=b1 - stop_back, z=top + STOP_Z),
                plate_zone=[[round(float(v), 4) for v in p] for p in zone],
                longest_plan_extent_m=round(span, 4), stop_distance_m=round(stop, 4),
                separator=dict(width_m=sep.width, rib_count=sep.rib_count, inlet_height_m=sep.inlet_height,
                               origin=[round(float(v), 4) for v in origin], floor_z=round(float(origin[2]), 4),
                               inlet=inlet, pivot=pivot, outlet_closed=out_c,
                               inlet_open=in_o, outlet_open=world(opened['outlet']),
                               closed_deg=sep.closed_deg, open_deg=sep.open_deg, swing_s=cfg['separator_swing_s'],
                               deck_friction=mu,
                               coal_slides_when_closed=bool(math.tan(math.radians(sep.closed_deg)) > mu)),
                sort_density_kg_m3=cfg['sort_density'],
                # a lump whose lowest point is below this has left the machine (the closed plate's outlet - 0.2 m)
                drop_z=round(float(origin[2] + closed['outlet'][2] - .2), 4),
                end_x=out_c[0])


def report(st):
    """The JSON-safe part of geometry()."""
    return {k: v for k, v in st.items() if k != 'sep'}


def landing(st, com):
    """Where a lump that fell below drop_z went, by its centroid: through the gap in front of the pivot
    ('sorted', 'gangue'), over the plate's far end ('sorted', 'coal'), else off a side or upstream
    ('dropped', None)."""
    if abs(com[1] - st['y_c']) > st['width_m'] / 2 + .10 or com[0] < st['measure']['x1'] - .05:
        return 'dropped', None
    return 'sorted', ('gangue' if com[0] < st['separator']['pivot'][0] else 'coal')


# ---- MJCF -------------------------------------------------------------------------------------------------
def belt_bodies(st):
    """B and M, their tops station_step below the main belt; B starts at the main belt's head edge (a lump tips
    onto it), M butts B flat."""
    return [plate_belt(name, st[key]['x0'], st[key]['x1'], st['y_c'], st['top_z'], st['width_m'] / 2, rgba)
            for name, key, rgba in ((BUFFER, 'buffer', '.20 .38 .50 1'), (MEASURE, 'measure', '.34 .30 .48 1'))]


def skirts(st):
    """M's skirts stand on the weigh frame (the load cells carry them); B's on the fixed frame."""
    return [skirt('%s_%s' % (pre, side), st['y_c'] + sign * (st['width_m'] + SKIRT['thickness']) / 2,
                  st[key]['x0'], st[key]['x1'], st['top_z'] + .004)
            for pre, key in (('bskirt', 'buffer'), ('mskirt', 'measure')) for side, sign in (('in', -1), ('out', 1))]
