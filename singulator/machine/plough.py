"""The singulation section: main belt, plough face, lane, side belt and their skirts.

    main belt (from under the feed belt head to its own head edge) -> plough face -> lane (lane_w wide)

The plough sweeps material toward LOW y: the incoming stream sits on the -n side of the face, so the face pushes
it that way. The face therefore has to stop exactly one lane width above the low skirt -- if it runs closer, the
face and the skirt close into a wedge and the pilot arches solid. Past that point (the bend, where the face is
hinged) the lane's outer wall continues straight, so the lane is a parallel channel of width lane_w.

Geometry is pure (numpy, no MuJoCo). Coordinates: x along the belt, y from the low-side datum, z up, main belt
top z = 0.
"""
import math

import numpy as np

from .parts import SKIRT, plate_belt, pulley, skirt

MAIN_BELT, TAIL = 'mfloor', 'mtail'       # the main belt plate (joint 'mfloorj'), its tail pulley ('mtailj')
FACE, FACE_JOINT, FACE_ACTUATOR = 'face', 'facej', 'face_drive'

PLOUGH = dict(thickness=.03, height=.50, bottom_gap=.005)

# Free-spinning vertical rollers in place of the plough plate (--face-kind rollers). A lump then meets
# a ROLLING contact instead of a sliding one: the only tangential force left is what accelerates the
# roller's own inertia (plus bearing drag), so the face stops both queueing lumps and spinning them,
# and an arch loses the tangential reaction it needs at that abutment. Pitch 0.11 m leaves a 20 mm
# clear gap between Ø90 rollers, against a smallest lump dimension of 0.13 m -- nothing gets in.
# Height 0.50 m = the tallest a lump of this size class can lie (its short axis <= the 0.5 m aperture).
FACE_ROLLER = dict(d=.09, pitch=.11, height=.50, bottom_gap=.005, mass=8., damping=.02)

FACE_SEGS = 24                  # boxes in the face polyline

SIDE_BELT = dict(pitch=.10, thickness=.02, height=.50, bottom_gap=.005)

# Equipment never collides with equipment in the simulation (contype 2 / conaffinity 0), so a moving part
# swinging through a fixed wall passes silently. The fixed walls next to the swinging face therefore stop
# this far short of everything the face sweeps (motion_windows), and assembly.motion_clearance() checks the result.
MOTION_CLEARANCE = .01


def geometry(cfg):
    """The face polyline from P0 at the belt edge to the bend P1 on the lane's outer wall line, the lane, and the
    numbers the report gives for the face (its length, stagger budget, transit time, end angles)."""
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
    return dict(face_pts=face_pts, P0=P0, P1=P1, lane_top=lane_top, exit_x=P1[0] + cfg['lane_len'],
                length=float(np.sum(dy / np.sin(b_seg))), stagger=float(np.sum(dy * np.tan(b_seg))),
                # a lump on the face moves along it at v*cos(beta), so dy/dt = -v*sin(beta)*cos(beta)
                transit=float(np.sum(dy / (cfg['v_belt'] * np.sin(b_seg) * np.cos(b_seg)))),
                beta=math.radians(cfg['curve_top_deg']), beta_exit=math.radians(cfg['curve_exit_deg']))


def side_belt(cfg, bend_x, belt_x1):
    """The driven side belt on the low side of the funnel and lane: a chain of slats from side_belt_back upstream
    of the bend to the main belt's head edge. Without it (--no-side-belt): no slats."""
    if not cfg['side_belt']:
        return dict(sb_x0=None, sb_chain=0, n_sbslat=0)
    x0 = bend_x - cfg['side_belt_back']
    chain = math.ceil((belt_x1 - x0) / SIDE_BELT['pitch']) * SIDE_BELT['pitch']
    return dict(sb_x0=x0, sb_chain=chain, n_sbslat=int(round(chain / SIDE_BELT['pitch'])))


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


# ---- MJCF -------------------------------------------------------------------------------------------------
def main_belt_body(d, y, half_w):
    """The main belt, from under the feed belt head (or its own tail pulley) to the head edge."""
    fd = d['feeder']
    return [plate_belt(MAIN_BELT, fd.get('x_lower', fd['x1']), d['belt_x1'], y, 0., half_w, '.31 .33 .36 1')]


def tail_pulley_body(d, y, half_w):
    """With --feeder-tail-d the main belt starts over its own tail pulley under the feed belt head."""
    fd = d['feeder']
    hd = fd.get('head')
    if hd and hd['tail_d_m']:
        return [pulley(TAIL, hd['tail_d_m'], fd.get('x_lower', fd['x1']), y, hd['tail_centre'][2], half_w,
                       '.31 .33 .36 1')]
    return []


def side_belt_bodies(cfg, d):
    """The forward-driven side belt on the low side of the lane, in place of the static skirt there: one body per
    slat, 'vslat%d' on the slide joint 'vj%d' (the drive places them along their chain)."""
    out = []
    for i in range(d['n_sbslat']):
        shade = (.16, .55, .36, 1) if i % 2 else (.22, .66, .45, 1)
        out.append('<body name="vslat%d" pos="0 %.4f %.4f"><joint name="vj%d" type="slide" axis="1 0 0" armature="10000"/>'
                   '<inertial pos="0 0 0" mass="2" diaginertia=".01 .01 .01"/>'
                   '<geom class="belt" type="box" size="%.5f %.4f %.4f" rgba="%s"/></body>'
                   % (i, cfg['lane_y'] - SIDE_BELT['thickness'] / 2,
                      SIDE_BELT['bottom_gap'] + SIDE_BELT['height'] / 2, i,
                      SIDE_BELT['pitch'] / 2 - .0005, SIDE_BELT['thickness'] / 2,
                      SIDE_BELT['height'] / 2, ' '.join('%.10g' % v for v in shade)))
    return out


def face_body(cfg, d):
    """The face -- a row of free vertical rollers ('fr%d'), or a polyline of plate boxes ('plough%d') -- as one
    body hinged at the bend, with its position servo. Returns (body XML, actuator XML)."""
    piv = d['P1']
    if cfg['face_kind'] == 'rollers':         # rollers ride on the swinging face: nested bodies
        R, m, h = FACE_ROLLER['d'] / 2, FACE_ROLLER['mass'], FACE_ROLLER['height']
        zc = FACE_ROLLER['bottom_gap'] + FACE_ROLLER['height'] / 2
        fgeoms = []
        for i, c in enumerate(face_roller_centres(d, FACE_ROLLER['pitch'], R)):
            name = 'fr%d' % i
            fgeoms.append('<body name="%s" pos="%.4f %.4f %.4f">'
                          '<joint name="%sj" type="hinge" axis="0 0 1" armature=".001" damping="%.4g" frictionloss="%.6g"/>'
                          '<inertial pos="0 0 0" mass="%.3f" diaginertia="%.5f %.5f %.5f"/>'
                          '<geom class="wall" type="cylinder" size="%.4f %.4f" rgba="%s"/></body>'
                          % (name, c[0] - piv[0], c[1] - piv[1], zc, name, FACE_ROLLER['damping'],
                             cfg['face_bearing_drag'], m,
                             m * (3 * R * R + h * h) / 12, m * (3 * R * R + h * h) / 12, m * R * R / 2,
                             R, h / 2, '.80 .82 .86 1' if i % 2 else '.62 .66 .72 1'))
    else:                                     # --face-kind plate: one box per segment of the polyline
        fgeoms = []
        for i, (A, Bp) in enumerate(zip(d['face_pts'][:-1], d['face_pts'][1:])):
            mid = (A + Bp) / 2
            ln = float(np.linalg.norm(Bp - A))
            ang = math.atan2(Bp[1] - A[1], Bp[0] - A[0])
            off = np.array([-math.sin(ang), math.cos(ang), 0.]) * PLOUGH['thickness'] / 2
            c = mid + off
            fgeoms.append('<geom name="plough%d" class="wall" type="box" size="%.4f %.4f %.4f" pos="%.4f %.4f %.4f" euler="0 0 %.6f" rgba=".78 .45 .25 1"/>'
                          % (i, ln / 2, PLOUGH['thickness'] / 2, PLOUGH['height'] / 2,
                             c[0] - piv[0], c[1] - piv[1], PLOUGH['bottom_gap'] + PLOUGH['height'] / 2, ang))
    # the hinge: a finite-torque position servo against soft joint stops
    delta = math.radians(cfg['face_swing_deg'])
    lo, hi = min(0., delta), max(0., delta)
    joint_xml = (f'<joint name="{FACE_JOINT}" type="hinge" axis="0 0 1" limited="true" '
                 f'range="{lo} {hi}" damping="10" solreflimit="{cfg["solref"]} {cfg["dampratio"]}"/>')
    # Approximate carrier as a uniform rectangular beam along the chord. Rollers retain their
    # own masses/inertias. Neither the 20 kg carrier nor servo gains are equipment measurements.
    chord = d['P1'] - d['P0']
    length = float(np.linalg.norm(chord))
    mid = (d['P0'] + d['P1']) / 2 - piv
    mass, thick, height = cfg['face_carrier_mass'], PLOUGH['thickness'], PLOUGH['height']
    iz = mass * (length ** 2 + thick ** 2) / 12
    inertial_xml = (f'<inertial pos="{mid[0]} {mid[1]} {height/2}" mass="{mass}" '
                    f'quat="{math.cos(math.atan2(chord[1], chord[0])/2)} 0 0 '
                    f'{math.sin(math.atan2(chord[1], chord[0])/2)}" '
                    f'diaginertia="{mass*(thick**2+height**2)/12} '
                    f'{mass*(length**2+height**2)/12} {iz}"/>')
    torque = cfg['face_force_max'] * length
    actuator = (f'<position name="{FACE_ACTUATOR}" joint="{FACE_JOINT}" kp="{cfg["face_kp"]}" '
                f'kv="{cfg["face_kv"]}" ctrllimited="true" ctrlrange="{lo} {hi}" '
                f'forcelimited="true" forcerange="{-torque} {torque}"/>')
    # child bodies (the rollers) come after the geoms, which is the order MJCF expects
    body = ('<body name="%s" pos="%.4f %.4f 0">%s%s'
            '<geom type="cylinder" size=".03 .22" pos="0 0 .22" contype="0" conaffinity="0" rgba=".2 .2 .2 1"/>'
            '%s</body>'
            % (FACE, piv[0], piv[1], joint_xml, inertial_xml, ''.join(fgeoms)))
    return body, actuator


def lane_wall(cfg, d):
    """The lane's outer wall ('lane_out'), from where the swinging face leaves room for it to the head edge; a
    wall of zero length (the bend at the belt end) is dropped."""
    a = (np.array(d['P1'], float) if d['lane_out_start'] <= d['P1'][0] + 1e-12   # a window only if the face needs one
         else np.array([d['lane_out_start'], d['lane_top'], 0.]))
    b = np.array([d['belt_x1'], d['lane_top'], 0.])
    ln = float(np.linalg.norm(b - a))
    if ln < .001:
        return []
    mid = (a + b) / 2
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    off = np.array([-math.sin(ang), math.cos(ang), 0.]) * PLOUGH['thickness'] / 2
    c = mid + off
    return ['<geom name="lane_out" class="wall" type="box" size="%.4f %.4f %.4f" pos="%.4f %.4f %.4f" euler="0 0 %.6f" rgba=".62 .5 .38 1"/>'
            % (ln / 2, PLOUGH['thickness'] / 2, PLOUGH['height'] / 2,
               c[0], c[1], PLOUGH['bottom_gap'] + PLOUGH['height'] / 2, ang)]


def skirts(cfg, d):
    """The two skirts of the main belt, from the feed belt's tail end: the outer one ('skirt_out') up to the
    window the swinging face needs, the low one ('skirt_in') up to the side belt (or the head edge)."""
    return [skirt('skirt_out', cfg['belt_w'] + SKIRT['thickness'] / 2, d['belt_x0'], d['skirt_out_end'], .004),
            skirt('skirt_in', cfg['lane_y'] - SKIRT['thickness'] / 2, d['belt_x0'],
                  d['sb_x0'] if cfg['side_belt'] else d['belt_x1'], .004)]


def markers(cfg, d):
    """Drawn only: the lane's exit plane, the measuring plane (the head edge), and with the 'lane' release rule
    the feed belt's next-release line (every released lump's rear must be past it)."""
    y = cfg['lane_y'] + cfg['lane_w'] / 2
    out = ['<geom name="exit_line" type="box" size=".006 %.4f .004" pos="%.4f %.4f .006" '
           'contype="0" conaffinity="0" rgba="1 .6 .15 .85"/>' % (cfg['lane_w'] / 2, d['exit_x'], y),
           '<geom name="final_line" type="box" size=".006 %.4f .004" pos="%.4f %.4f %.4f" '
           'contype="0" conaffinity="0" rgba="1 .15 .15 .9"/>' % (cfg['lane_w'] / 2, d['final_x'], y, .006)]
    if cfg['feeder_release'] == 'lane':
        out.append('<geom name="release_line" type="box" size=".006 %.4f .004" pos="%.4f %.4f .006" '
                   'contype="0" conaffinity="0" rgba=".15 .45 1 .9"/>' % (cfg['lane_w'] / 2, d['lane_out_start'], y))
    return out
