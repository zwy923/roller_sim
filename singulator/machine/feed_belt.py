"""The feed belt: a short stop-start belt with a step-down transfer onto the main belt.

    feed belt (whole batch laid on it, own drive), its top feeder_step ABOVE the main belt
    -> main belt, straight run of feeder_gap -> plough face -> lane

Why the step. On a feed belt butted flat to the main belt a lump straddling the joint is held or dragged by
friction on two surfaces. With the feed belt a step above the main belt, a lump overhanging the head edge still
lies wholly on the feed belt -- the overhang is in the air, clear of the main belt -- until its CENTROID passes
the edge; then it tips onto the main belt, which takes it away. Hold-back is gravity, not friction.

Hardware as modelled:
  * a belt as wide as the main belt (belt_w, same skirts), feeder_len long, its top feeder_step above the main belt
    top, with a sharp head edge at x1 (the default). The transfer as built can be modelled instead (user
    2026-09-30: 高差和交接位置做成可调): a driven head drum (feeder_head_d) whose top is tangent to the flat run
    at x1, the main belt starting over its own tail pulley (feeder_tail_d) at x_lower = x1 + feeder_handover, which
    must leave the two clear of each other (min_handover). With a real drum and a small step the belts stand
    0.1-0.25 m apart and a lump spans both; the bench trial (designs/transfer_trial/) is where that is measured;
  * a drop beam S1 across the belt BEAM_X past the edge at half the step height. A lump still lying on the feed
    belt overhangs above it; one that tips cuts it. The beam is the feed belt's stop signal;
  * feeder_gap of main belt between the edge and the plough start: at least the longest plan extent of a
    flat-lying lump plus the creep stopping distance, so a lump held up at the plough never still lies on the feed
    belt.

The release control is control/feeder.py. Geometry here is pure (no MuJoCo); nothing is calibrated.
"""
import math

from ..config import SAMPLE_S
from .parts import plate_belt, pulley

BODY, DRUM = 'feeder', 'fdrum'       # the belt plate (joint 'feederj') and its head drum (joint 'fdrumj')
EQUIP_CLEAR = .01    # m: the head drum and the tail pulley below keep at least this apart
BEAM_X = .10         # m: the drop beam lies this far past the head edge, at half the step height


def geometry(cfg, first_x):
    """Feed belt from x0 to the head edge x1, feeder_step above the main belt. first_x: the plough start, the
    first thing downstream that can hold a lump up."""
    for k in ('feeder_speed', 'feeder_creep_speed', 'feeder_step', 'feeder_len', 'feeder_gap', 'feeder_ramp_s',
              'feeder_stall_s', 'feeder_force_max'):
        if not math.isfinite(cfg[k]) or cfg[k] <= 0:
            raise ValueError('--%s must be finite and positive' % k.replace('_', '-'))
    for k in ('feeder_head_d', 'feeder_tail_d', 'feeder_handover'):
        if not math.isfinite(cfg[k]) or cfg[k] < 0:
            raise ValueError('--%s must be finite and not negative' % k.replace('_', '-'))
    v, vc, h, gap = cfg['feeder_speed'], cfg['feeder_creep_speed'], cfg['feeder_step'], cfg['feeder_gap']
    R, r, hand = cfg['feeder_head_d'] / 2, cfg['feeder_tail_d'] / 2, cfg['feeder_handover']
    need = min_handover(h, R, r)
    if hand < need - 1e-9:
        raise ValueError('--feeder-handover %.3f m: the head drum (D %.3f m) and the belt below (tail pulley D %.3f m, '
                         '%.3f m lower) would collide; it needs at least %.3f m' % (hand, 2 * R, 2 * r, h, need))
    if v >= cfg['v_belt']:
        raise ValueError('--feeder-speed %.3g must be below the main belt speed %.3g: the faster main belt is '
                         'what takes a released lump away from the rest' % (v, cfg['v_belt']))
    if vc > v:
        raise ValueError('--feeder-creep-speed %.3g must not exceed --feeder-speed %.3g' % (vc, v))
    span = math.hypot(cfg['size_long_max'], cfg['size_max'])   # plan diagonal of a flat-lying lump, bound
    stop = vc * cfg['feeder_ramp_s'] * (vc / v) / 2 + vc * SAMPLE_S   # ramp-down from creep + one camera sample
    if gap < hand + span + stop:
        raise ValueError('--feeder-gap %.3f m is shorter than the hand-over (%.3f m) + one lump (%.3f m plan extent) + '
                         'the feed belt stopping distance (%.3f m): a lump held up at the plough could still lie on '
                         'the feed belt' % (gap, hand, span, stop))
    x1 = first_x - gap
    head = {} if not (R or r or hand) else dict(
        head=dict(drum_d_m=2 * R, tail_d_m=2 * r, handover_m=hand, min_handover_m=round(need, 4),
                  drum_centre=[x1, None, round(h - R, 4)] if R else None,
                  tail_centre=[x1 + hand, None, round(-r, 4)] if r else None))
    return dict(x0=x1 - cfg['feeder_len'], x1=x1, length_m=cfg['feeder_len'], width_m=cfg['belt_w'],
                **({} if not hand else dict(x_lower=x1 + hand)), **head,
                gap_to_plough_m=gap, step_m=h, speed_m_s=v, creep_speed_m_s=vc,
                main_to_feeder_speed_ratio=cfg['v_belt'] / v,
                longest_plan_extent_m=round(span, 4), stop_distance_m=round(stop, 4),
                stop_distance_basis='nominal ramp-down from creep plus one sample; not a bound under brake overload',
                beam=dict(x=x1 + hand + BEAM_X, z=h / 2), release_rule='centroid past the head edge (the lump tips)')


def min_handover(h, R, r, clear=None):
    """Smallest horizontal distance from the head drum's top tangent (radius R, 0 = a sharp edge) to the
    flat start of the belt below (tail pulley radius r, 0 = a square plate end) with the drum and the belt
    below clear of each other (by EQUIP_CLEAR unless `clear` is given). The feed belt top is h above the belt
    below."""
    clear = EQUIP_CLEAR if clear is None else clear
    if R == 0. and r == 0.:
        return 0.
    if r == 0.:                     # the drum against the square end of the plate below (its top at 0)
        return math.sqrt(max(0., R * R - (h - R) ** 2)) + clear if h < 2 * R else 0.
    # the drum against the tail pulley: centres (0, h - R) and (x, -r)
    dz = h - R + r
    return math.sqrt(max(0., (R + r + clear) ** 2 - dz ** 2))


def bodies(fd, y, half_w):
    """MJCF: the feed belt, its top feeder_step above the main belt and ending in a sharp head edge at x1 over the
    main belt's start. A lump overhanging the edge is still carried by the feed belt alone until its centroid
    passes the edge and it tips down. With --feeder-head-d the belt wraps a driven head drum whose top is tangent
    to its flat run at x1."""
    out = [plate_belt(BODY, fd['x0'], fd['x1'], y, fd['step_m'], half_w, '.22 .36 .30 1')]
    hd = fd.get('head')
    if hd and hd['drum_d_m']:
        out.append(pulley(DRUM, hd['drum_d_m'], fd['x1'], y, hd['drum_centre'][2], half_w, '.22 .36 .30 1'))
    return out
