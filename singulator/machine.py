"""Machine dimensions and derived geometry: feed belt, plough face, lane, main belt, station, motion windows.

Pure geometry, no MuJoCo. Coordinates: x along the belt, y from the low-side datum, z up, belt top z = 0.
"""
import math

import numpy as np

from . import devices, feeder, station

SLAT = dict(pitch=.10, thickness=.04, gap=.0005)

PLOUGH = dict(thickness=.03, height=.50, bottom_gap=.005)

# Free-spinning vertical rollers in place of the plough plate (--face-kind rollers). A lump then meets
# a ROLLING contact instead of a sliding one: the only tangential force left is what accelerates the
# roller's own inertia (plus bearing drag), so the face stops both queueing lumps and spinning them,
# and an arch loses the tangential reaction it needs at that abutment. Pitch 0.11 m leaves a 20 mm
# clear gap between Ø90 rollers, against a smallest lump dimension of 0.13 m -- nothing gets in.
# Height 0.50 m = the tallest a lump of this size class can lie (its short axis <= the 0.5 m aperture).
FACE_ROLLER = dict(d=.09, pitch=.11, height=.50, bottom_gap=.005, mass=8., damping=.02)

FACE_SEGS = 24                  # boxes in the curved face polyline (--face-shape curve)

SKIRT = dict(height=.50, thickness=.02)

SIDE_BELT = dict(pitch=.10, thickness=.02, height=.50, bottom_gap=.005)

# Equipment never collides with equipment in the simulation (contype 2 / conaffinity 0), so a moving part
# swinging through a fixed wall passes silently. The fixed walls next to the swinging face therefore stop
# this far short of everything the face sweeps (motion_windows), and motion_clearance() checks the result.
MOTION_CLEARANCE = .01


def derive(cfg):
    """Feed belt, plough-face frame, lane, exit plane, main belt extent, side belt and the station.

    The plough sweeps material toward LOW y: the incoming stream sits on the -n side of the face, so the
    face pushes it that way. The diagonal therefore has to stop exactly one lane width above the low
    skirt -- if it runs closer, the face and the skirt close into a wedge and the pilot arches solid.
    Past that point the same plate continues straight as the lane's outer wall, so the whole plough is
    one bent plate and the lane is a parallel channel of width lane_w.
    """
    # Reject combinations that used to build a silently nonsensical machine (negative diagonal, a lane
    # wider than the belt, a plough that runs backwards).
    if cfg['lane_w'] <= 0.:
        raise ValueError('--lane-w must be > 0, got %g' % cfg['lane_w'])
    if cfg['lane_y'] < 0. or cfg['lane_y'] + cfg['lane_w'] >= cfg['belt_w']:
        raise ValueError('lane must sit inside the belt: need 0 <= lane_y (%g) and lane_y + lane_w (%g) < belt_w (%g);'
                         ' otherwise there is nothing for the face to sweep and the diagonal runs backwards'
                         % (cfg['lane_y'], cfg['lane_y'] + cfg['lane_w'], cfg['belt_w']))
    lane_top = cfg['lane_y'] + cfg['lane_w']
    P0 = np.array([cfg['plough_x'], cfg['belt_w'], 0.])       # high end of the face
    # The face turns linearly (in angle) from curve_top at the belt edge to curve_exit at the lane: abreast
    # pairs are sheared apart on the steep top, and the lane is handed lumps at curve_exit, leaving at crawl
    # v*cos^2(curve_exit) -- 0.30 m/s at 30 deg against 0.132 at 55. The price is the stagger budget, integral
    # tan(beta) dy over the sweep: 0.48 m at 55->25 and 0.58 m at 60->30, against 0.785 m for a straight
    # 55 deg face (the straight face was an option until 2026-10-05).
    if not 0. < cfg['curve_exit_deg'] < cfg['curve_top_deg'] < 90.:
        raise ValueError('curved face needs 0 < curve-exit-deg < curve-top-deg < 90, got top %g exit %g'
                         % (cfg['curve_top_deg'], cfg['curve_exit_deg']))
    # Polyline of FACE_SEGS boxes: segment angles run exactly top -> exit, each dropping an equal share of the
    # y sweep, so the last box really is at curve_exit_deg and the face ends on y = lane_top (the lane's outer
    # wall continues from there).
    b_seg = np.radians(np.linspace(cfg['curve_top_deg'], cfg['curve_exit_deg'], FACE_SEGS))
    dy = (cfg['belt_w'] - lane_top) / FACE_SEGS
    xs = [cfg['plough_x']]
    ys = [cfg['belt_w']]
    for b in b_seg:
        xs.append(xs[-1] + dy / math.tan(b))
        ys.append(ys[-1] - dy)
    face_pts = [np.array([x, y, 0.]) for x, y in zip(xs, ys)]
    P1 = face_pts[-1]                                         # the bend: face -> lane outer wall; the hinge
    s_end = float(np.sum(dy / np.sin(b_seg)))
    stagger = float(np.sum(dy * np.tan(b_seg)))
    # a lump on the face moves along it at v*cos(beta), so dy/dt = -v*sin(beta)*cos(beta)
    transit = float(np.sum(dy / (cfg['v_belt'] * np.sin(b_seg) * np.cos(b_seg))))
    beta = math.radians(cfg['curve_top_deg'])
    beta_exit = math.radians(cfg['curve_exit_deg'])
    exit_x = P1[0] + cfg['lane_len']
    # the batch starts on the feed belt, which ends feeder_gap before the plough start; the conveying surface
    # begins at its tail end
    fd = feeder.geometry(cfg, cfg['plough_x'])
    x0, x1 = fd['x0'], exit_x + .05
    # behind the main belt's head edge x1: buffer belt, measuring belt and separator. The singulator's
    # measuring plane is that edge, where a lump tips onto the buffer belt
    st = station.geometry(cfg, x1, cfg['lane_y'] + cfg['lane_w'] / 2)
    final_x = x1
    sweep = cfg['belt_w'] - lane_top
    sb_x0 = P1[0] - cfg['side_belt_back'] if cfg['side_belt'] else None
    sb_chain = n_sbslat = 0
    if cfg['side_belt']:
        sb_chain = math.ceil((x1 - sb_x0) / SIDE_BELT['pitch']) * SIDE_BELT['pitch']
        n_sbslat = int(round(sb_chain / SIDE_BELT['pitch']))
    v_side = cfg['v_belt'] * cfg['side_belt_ratio']
    # beta budget: a lump slides along the face only while tan(beta) < 1/mu_face (belt drive resolved along
    # the face beats the friction opposing it); above that it locks and the belt runs under it. Lining the
    # face with UHMW-PE (0.20) moves the ceiling from 65.8 to 78.7 deg.
    lock = math.tan(beta) * cfg['friction_steel']
    d = dict(face_pts=face_pts, P0=P0, P1=P1,
                lane_top=lane_top, exit_x=exit_x, final_x=final_x,
                belt_x0=x0, belt_x1=x1, feeder=fd, station=st,
                # a lump whose lowest point falls below drop_z has left the machine (tracking.landing says where)
                drop_z=st['drop_z'], end_states=('sorted', 'dropped', 'taken_off'), view_x1=st['end_x'] + .3,
                sb_x0=sb_x0, sb_chain=sb_chain, n_sbslat=n_sbslat, v_side=v_side,
                report=dict(face_shape='curve', face_kind=cfg['face_kind'],
                            curve_top_deg=cfg['curve_top_deg'], curve_exit_deg=cfg['curve_exit_deg'],
                            face_transit_s=round(transit, 2),
                            belt_width_m=cfg['belt_w'],
                            lane_y_m=cfg['lane_y'], lane_width_m=cfg['lane_w'], lane_length_m=cfg['lane_len'],
                            diagonal_length_m=s_end, diagonal_from=[float(P0[0]), float(P0[1])],
                            bend_at=[float(P1[0]), float(P1[1])], exit_plane_x_m=exit_x,
                            sweep_m=sweep, belt_speed_m_s=cfg['v_belt'],
                            # the four speeds below are at the lane handoff (curve_exit)
                            along_face_speed_m_s=cfg['v_belt'] * math.cos(beta_exit),
                            face_crawl_speed_m_s=cfg['v_belt'] * math.cos(beta_exit) ** 2,
                            face_lateral_speed_m_s=cfg['v_belt'] * math.sin(beta_exit) * math.cos(beta_exit),
                            deabreast_rate_m_s=cfg['v_belt'] * math.sin(beta_exit) ** 2,
                            functional_length_m=float(exit_x - P0[0]),
                            predicted_stagger_of_outermost_lump_m=stagger,
                            face_locking=dict(mu_face=cfg['friction_steel'], tan_beta=math.tan(beta),
                                              ratio_mu_tan_beta=lock, margin=1. - lock,
                                              slides_along_face=bool(lock < 1.),
                                              lock_ceiling_deg=math.degrees(math.atan(1. / cfg['friction_steel']))
                                              if cfg['friction_steel'] > 0 else None),
                            friction=dict(block_belt=cfg['friction_belt'], block_steel=cfg['friction_steel'],
                                          block_block=cfg['friction_block']),
                            belt_push_normal_to_face_per_100kg_N=cfg['friction_belt'] * 100. * 9.81 * math.sin(beta),
                            final_plane_x_m=final_x,
                            side_belt=dict(on=bool(cfg['side_belt']), from_x_m=sb_x0,
                                           to_x_m=x1 if cfg['side_belt'] else None,
                                           length_m=(x1 - sb_x0) if cfg['side_belt'] else 0.,
                                           speed_m_s=v_side if cfg['side_belt'] else None,
                                           ratio_to_main=cfg['side_belt_ratio'], slats=n_sbslat),
                            feed_band='full', layout=cfg['layout'],
                            lumps_per_batch=cfg['count'],
                            unjam=dict(on=True, face_hinge='bend',
                                       face_swing_deg=cfg['face_swing_deg'],
                                       face_force_max_N=cfg['face_force_max'],
                                       hold_timeout_action='stop the batch with the face still retracted; '
                                                           'it never closes onto material'),
                            moving_parts='feed belt, main belt, side belt, buffer belt, measuring belt, separator '
                                         'plate; the face swings when unjam is on'))
    d.update(motion_windows(cfg, d))
    d['report']['motion_windows'] = dict(skirt_out_end_x_m=d['skirt_out_end'], lane_out_start_x_m=d['lane_out_start'],
                                         clearance_m=MOTION_CLEARANCE)
    d['report']['feeder'] = dict(fd, lumps_start_on_it=True)
    d['report']['station'] = station.report(st)
    d['devices'] = devices.layout(cfg, d)               # the sensors as hardware
    d['report']['devices'] = devices.report(d['devices'])
    return d


def face_roller_centres(d, pitch, r):
    """Centres of the vertical face rollers: spaced by arc length along the face line, one radius behind
    it, so the roller SURFACES are tangent to the same material face the plate presented.

    The bottom roller is laid first, tangent to both the face line and the lane's outer-wall line
    (y = lane_top) -- the fillet of the bend, r * tan(b/2) either side of P1 -- so its low tangent is
    flush with the wall and the wall's inlet corner stays behind it. The rest follow upstream at pitch.
    Pre-2026-09-22 builds counted from the top end instead: the bottom roller then ended 48 mm above the
    wall line and the bare wall corner at P1 stood in the lane mouth.
    """
    pts = np.array(d['face_pts'])[:, :2]
    seg = np.diff(pts, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.], np.cumsum(ln)])
    b = math.atan2(-seg[-1][1], seg[-1][0])                 # last segment against the wall line (+x)
    s_bot = s[-1] - r * math.tan(b / 2)
    svals = s_bot - pitch * np.arange(int((s_bot - r) / pitch + 1e-9) + 1)[::-1]
    out = []
    for si in svals:
        i = max(0, min(int(np.searchsorted(s, si)) - 1, len(seg) - 1))
        u = seg[i] / ln[i]
        p = pts[i] + (si - s[i]) * u
        out.append(p + r * np.array([-u[1], u[0]]))       # +n, away from the material
    return out


def face_plate_segments(d):
    """The plate face (--face-kind plate) as plan-view quadrilaterals, one per segment of the face polyline:
    the material face on the polyline, the plate on its far (+n) side."""
    out = []
    for A, Bp in zip(d['face_pts'][:-1], d['face_pts'][1:]):
        u = (Bp - A)[:2] / np.linalg.norm((Bp - A)[:2])
        off = np.array([-u[1], u[0]]) * PLOUGH['thickness']
        out.append(np.array([A[:2], Bp[:2], Bp[:2] + off, A[:2] + off]))
    return out


def face_swing_outline(cfg, d):
    """Plan-view outline points of the face (its rollers, or its plate) at every sampled swing angle about the
    hinge at the bend."""
    if cfg['face_kind'] == 'rollers':
        R = FACE_ROLLER['d'] / 2
        a = np.linspace(0., 2 * math.pi, 48, endpoint=False)
        ring = R * np.stack([np.cos(a), np.sin(a)], 1)
        pts = [c + ring for c in face_roller_centres(d, FACE_ROLLER['pitch'], R)]
    else:
        pts = face_plate_segments(d)
    P = np.concatenate(pts)
    piv = d['P1'][:2]
    out = []
    for th in np.radians(np.linspace(0., cfg['face_swing_deg'], 51)):
        c, s = math.cos(th), math.sin(th)
        out.append(piv + (P - piv) @ np.array([[c, s], [-s, c]]))
    return np.concatenate(out)


def face_parts(cfg, d, sides=16):
    """The face's moving parts at rest as convex plan-view polygons, and the hinge point (the bend). Rollers are
    circumscribed polygons (never smaller than the roller)."""
    if cfg['face_kind'] == 'rollers':
        R = FACE_ROLLER['d'] / 2 / math.cos(math.pi / sides)
        a = np.linspace(0., 2 * math.pi, sides, endpoint=False)
        ring = R * np.stack([np.cos(a), np.sin(a)], 1)
        parts = [c + ring for c in face_roller_centres(d, FACE_ROLLER['pitch'], FACE_ROLLER['d'] / 2)]
    else:
        parts = face_plate_segments(d)
    return np.array(parts), np.array(d['P1'][:2], float)


def motion_windows(cfg, d):
    """Where the outer skirt must end and the lane's outer wall must start so the face clears both.

    Pre-2026-09-22 builds ran both walls through the face: the first roller sat 35 mm inside skirt_out at
    rest and the swing kept rollers inside it all the way to -25 deg; the last roller sat 6 mm inside
    lane_out. With equipment-equipment contact filtered out nothing showed. The walls now stop short;
    the belt edge between the skirt end and the swung face is open while the face is retracted.
    """
    P = face_swing_outline(cfg, d)
    cl = MOTION_CLEARANCE
    near_skirt = P[P[:, 1] > cfg['belt_w'] - cl] if len(P) else P
    skirt_end = float(near_skirt[:, 0].min() - cl) if len(near_skirt) else float(d['belt_x1'])
    near_lane = P[P[:, 1] < d['lane_top'] + PLOUGH['thickness'] + cl] if len(P) else P
    lane_start = float(max(d['P1'][0], near_lane[:, 0].max() + cl)) if len(near_lane) else float(d['P1'][0])
    if skirt_end <= d['belt_x0'] + .5:
        raise ValueError('the face swing reaches the outer skirt at x = %.3f, too close to the belt start' % skirt_end)
    return dict(skirt_out_end=skirt_end, lane_out_start=lane_start)
