"""The line's geometry: every section placed one after the other.

    feed belt -> main belt, plough face, lane -> buffer belt -> measuring belt -> flip separator, and the sensors

derive(cfg) is pure geometry (numpy only); the same dict `d` goes to the model (assembly.py), the drives, the
sensors and the controllers. What it holds:

    feeder, station, devices     the sections' own geometry (feed_belt.py, station.py, sensors.py)
    face_pts, P0, P1             the face polyline, its high end at the belt edge, the bend (and hinge)
    lane_top, lane_out_start     y of the lane's outer wall; x where that wall starts (the lane entry)
    exit_x, final_x              the lane's exit plane; the measuring plane = the main belt's head edge
    belt_x0, belt_x1             the conveying surface: the feed belt's tail end, the main belt's head edge
    skirt_out_end                x where the outer skirt ends (the window the swinging face needs)
    sb_x0, sb_chain, n_sbslat, v_side    the side belt: start, chain length, slats, surface speed
    drop_z                       a lump whose lowest point is below it has left the machine
    view_x1                      downstream end of what the plan view shows
    report                       the JSON-safe description that goes to result.json ('geometry')

Coordinates: x along the belt, y from the low-side datum, z up, main belt top z = 0.
"""
import math

from . import feed_belt, plough, sensors, station


def derive(cfg):
    g = plough.geometry(cfg)
    P0, P1, lane_top, exit_x = g['P0'], g['P1'], g['lane_top'], g['exit_x']
    beta, beta_exit = g['beta'], g['beta_exit']
    # the batch starts on the feed belt, which ends feeder_gap before the plough start; the conveying surface
    # begins at its tail end
    fd = feed_belt.geometry(cfg, cfg['plough_x'])
    x0, x1 = fd['x0'], exit_x + .05
    # behind the main belt's head edge x1: buffer belt, measuring belt and separator. The singulator's
    # measuring plane is that edge, where a lump tips onto the buffer belt
    st = station.geometry(cfg, x1, cfg['lane_y'] + cfg['lane_w'] / 2)
    final_x = x1
    sweep = cfg['belt_w'] - lane_top
    sb = plough.side_belt(cfg, P1[0], x1)
    v_side = cfg['v_belt'] * cfg['side_belt_ratio']
    # beta budget: a lump slides along the face only while tan(beta) < 1/mu_face (belt drive resolved along
    # the face beats the friction opposing it); above that it locks and the belt runs under it. Lining the
    # face with UHMW-PE (0.20) moves the ceiling from 65.8 to 78.7 deg.
    lock = math.tan(beta) * cfg['friction_steel']
    d = dict(face_pts=g['face_pts'], P0=P0, P1=P1,
             lane_top=lane_top, exit_x=exit_x, final_x=final_x,
             belt_x0=x0, belt_x1=x1, feeder=fd, station=st,
             # a lump whose lowest point falls below drop_z has left the machine (station.landing says where)
             drop_z=st['drop_z'], view_x1=st['end_x'] + .3,
             v_side=v_side, **sb,
             report=dict(face_shape='curve', face_kind=cfg['face_kind'],
                         curve_top_deg=cfg['curve_top_deg'], curve_exit_deg=cfg['curve_exit_deg'],
                         face_transit_s=round(g['transit'], 2),
                         belt_width_m=cfg['belt_w'],
                         lane_y_m=cfg['lane_y'], lane_width_m=cfg['lane_w'], lane_length_m=cfg['lane_len'],
                         diagonal_length_m=g['length'], diagonal_from=[float(P0[0]), float(P0[1])],
                         bend_at=[float(P1[0]), float(P1[1])], exit_plane_x_m=exit_x,
                         sweep_m=sweep, belt_speed_m_s=cfg['v_belt'],
                         # the four speeds below are at the lane handoff (curve_exit)
                         along_face_speed_m_s=cfg['v_belt'] * math.cos(beta_exit),
                         face_crawl_speed_m_s=cfg['v_belt'] * math.cos(beta_exit) ** 2,
                         face_lateral_speed_m_s=cfg['v_belt'] * math.sin(beta_exit) * math.cos(beta_exit),
                         deabreast_rate_m_s=cfg['v_belt'] * math.sin(beta_exit) ** 2,
                         functional_length_m=float(exit_x - P0[0]),
                         predicted_stagger_of_outermost_lump_m=g['stagger'],
                         face_locking=dict(mu_face=cfg['friction_steel'], tan_beta=math.tan(beta),
                                           ratio_mu_tan_beta=lock, margin=1. - lock,
                                           slides_along_face=bool(lock < 1.),
                                           lock_ceiling_deg=math.degrees(math.atan(1. / cfg['friction_steel']))
                                           if cfg['friction_steel'] > 0 else None),
                         friction=dict(block_belt=cfg['friction_belt'], block_steel=cfg['friction_steel'],
                                       block_block=cfg['friction_block']),
                         belt_push_normal_to_face_per_100kg_N=cfg['friction_belt'] * 100. * 9.81 * math.sin(beta),
                         final_plane_x_m=final_x,
                         side_belt=dict(on=bool(cfg['side_belt']), from_x_m=sb['sb_x0'],
                                        to_x_m=x1 if cfg['side_belt'] else None,
                                        length_m=(x1 - sb['sb_x0']) if cfg['side_belt'] else 0.,
                                        speed_m_s=v_side if cfg['side_belt'] else None,
                                        ratio_to_main=cfg['side_belt_ratio'], slats=sb['n_sbslat']),
                         feed_band='full', layout=cfg['layout'],
                         lumps_per_batch=cfg['count'],
                         unjam=dict(on=True, face_hinge='bend',
                                    face_swing_deg=cfg['face_swing_deg'],
                                    face_force_max_N=cfg['face_force_max'],
                                    hold_timeout_action='stop the batch with the face still retracted; '
                                                        'it never closes onto material'),
                         moving_parts='feed belt, main belt, side belt, buffer belt, measuring belt, separator '
                                      'plate; the face swings when unjam is on'))
    d.update(plough.motion_windows(cfg, d))
    d['report']['motion_windows'] = dict(skirt_out_end_x_m=d['skirt_out_end'], lane_out_start_x_m=d['lane_out_start'],
                                         clearance_m=plough.MOTION_CLEARANCE)
    d['report']['feeder'] = dict(fd, lumps_start_on_it=True)
    d['report']['station'] = station.report(st)
    d['devices'] = sensors.layout(cfg, d)               # the sensors as hardware
    d['report']['devices'] = sensors.report(d['devices'])
    return d
