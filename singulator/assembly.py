"""MuJoCo model of the machine and the moving-part clearance sweep."""
import math

import mujoco
import numpy as np

from . import devices, separator
from .machine import FACE_ROLLER, PLOUGH, SIDE_BELT, SKIRT, SLAT, face_roller_centres
from .physics import DEFAULTS


def f3(values):
    return ' '.join('%.10g' % v for v in values)


def build_xml(cfg, d, blocks):
    # Configs archived before this revision explicitly retain the kinematic assembly when built alone.
    dynamic_face = cfg.get('face_drive_model', 'kinematic') == 'dynamic'
    cfg = dict(DEFAULTS, **cfg)
    assets, body = [], []
    face_actuator = ''
    for k, b in enumerate(blocks):
        assets.append('<mesh name="blk%d" vertex="%s"/>' % (k, f3(b['vertices'].ravel())))

    half_y = (cfg['belt_w'] + 2 * SKIRT['thickness'] + .06) / 2
    y_floor_mid = cfg['belt_w'] / 2
    # the main belt: a plate whose position is held and whose velocity is prescribed, which for contact is
    # exactly a belt surface; from under the feed belt head (or its own tail pulley) to the head edge
    fd = d['feeder']
    m0 = fd.get('x_lower', fd['x1'])
    body.append('<body name="mfloor" pos="%.4f %.4f %.4f"><joint name="mfloorj" type="slide" axis="1 0 0" armature="10000"/>'
                '<inertial pos="0 0 0" mass="5" diaginertia=".1 .1 .1"/>'
                '<geom class="belt" type="box" size="%.4f %.4f %.4f" rgba=".31 .33 .36 1"/></body>'
                % ((m0 + d['belt_x1']) / 2, y_floor_mid, -SLAT['thickness'] / 2,
                   (d['belt_x1'] - m0) / 2, half_y, SLAT['thickness'] / 2))
    # the short stop-start feed belt, its top feeder_step above the main belt and ending in a sharp head
    # edge at x1 over the main belt's start. A lump overhanging the edge is still carried by the feed belt
    # alone (the overhang is in the air) until its centroid passes the edge and it tips down. Same
    # held-plate model as the main belt; a real head pulley would round the edge
    body.append('<body name="feeder" pos="%.4f %.4f %.4f"><joint name="feederj" type="slide" axis="1 0 0" armature="10000"/>'
                '<inertial pos="0 0 0" mass="5" diaginertia=".1 .1 .1"/>'
                '<geom class="belt" type="box" size="%.4f %.4f %.4f" rgba=".22 .36 .30 1"/></body>'
                % ((fd['x0'] + fd['x1']) / 2, y_floor_mid, fd['step_m'] - SLAT['thickness'] / 2,
                   (fd['x1'] - fd['x0']) / 2, half_y, SLAT['thickness'] / 2))
    hd = fd.get('head')
    # the transfer as built (--feeder-head-d / --feeder-tail-d / --feeder-handover): the feed belt wraps a
    # driven head drum whose top is tangent to its flat run at x1; the belt below starts over its own tail
    # pulley at x_lower. Both turn with their belts (surface speed = belt speed)
    pulleys = []
    if hd and hd['drum_d_m']:
        pulleys.append(('fdrum', hd['drum_d_m'], fd['x1'], hd['drum_centre'][2], '.22 .36 .30 1'))
    if hd and hd['tail_d_m']:
        pulleys.append(('mtail', hd['tail_d_m'], fd.get('x_lower', fd['x1']), hd['tail_centre'][2], '.31 .33 .36 1'))
    for name, dia, cx, cz, rgba in pulleys:
        body.append('<body name="%s" pos="%.4f %.4f %.4f"><joint name="%sj" type="hinge" axis="0 1 0" armature="1000"/>'
                    '<inertial pos="0 0 0" mass="20" diaginertia=".2 .2 .2"/>'
                    '<geom class="belt" type="cylinder" fromto="0 %.4f 0 0 %.4f 0" size="%.4f" rgba="%s"/></body>'
                    % (name, cx, y_floor_mid, cz, name, -half_y, half_y, dia / 2, rgba))
    st = d['station']
    # buffer belt B and measuring belt M, held plates like the main belt, their tops station_step below
    # it; B starts at the main belt's head edge (a lump tips onto it), M butts B flat
    for name, key, rgba in (('bbelt', 'buffer', '.20 .38 .50 1'), ('mbelt', 'measure', '.34 .30 .48 1')):
        b = st[key]
        body.append('<body name="%s" pos="%.4f %.4f %.4f"><joint name="%sj" type="slide" axis="1 0 0" armature="10000"/>'
                    '<inertial pos="0 0 0" mass="5" diaginertia=".1 .1 .1"/>'
                    '<geom class="belt" type="box" size="%.4f %.4f %.4f" rgba="%s"/></body>'
                    % (name, (b['x0'] + b['x1']) / 2, st['y_c'], st['top_z'] - SLAT['thickness'] / 2, name,
                       (b['x1'] - b['x0']) / 2, st['width_m'] / 2, SLAT['thickness'] / 2, rgba))
    static = []

    # forward-driven side belt on the low side of the lane, in place of the static skirt there
    for i in range(d['n_sbslat']):
        shade = (.16, .55, .36, 1) if i % 2 else (.22, .66, .45, 1)
        body.append('<body name="vslat%d" pos="0 %.4f %.4f"><joint name="vj%d" type="slide" axis="1 0 0" armature="10000"/>'
                    '<inertial pos="0 0 0" mass="2" diaginertia=".01 .01 .01"/>'
                    '<geom class="belt" type="box" size="%.5f %.4f %.4f" rgba="%s"/></body>'
                    % (i, cfg['lane_y'] - SIDE_BELT['thickness'] / 2,
                       SIDE_BELT['bottom_gap'] + SIDE_BELT['height'] / 2, i,
                       SIDE_BELT['pitch'] / 2 - .0005, SIDE_BELT['thickness'] / 2,
                       SIDE_BELT['height'] / 2, f3(shade)))

    # One bent plate: diagonal face, then straight as the lane's outer wall. Two flat skirts.
    # With --face-shape curve the diagonal is a polyline of boxes; with the swing unjam the whole
    # face (the single box or the whole polyline) is one hinged body instead of static geoms.
    lane_a = (np.array(d['P1'], float) if d['lane_out_start'] <= d['P1'][0] + 1e-12   # a window only if the face needs one
              else np.array([d['lane_out_start'], d['lane_top'], 0.]))
    lane_b = np.array([d['belt_x1'], d['lane_top'], 0.])
    fr = []                                   # vertical face rollers, as (name, centre, z) in world x-y
    if cfg['face_kind'] == 'rollers':
        R = FACE_ROLLER['d'] / 2
        zc = FACE_ROLLER['bottom_gap'] + FACE_ROLLER['height'] / 2
        fr = [('fr%d' % i, c, zc)
              for i, c in enumerate(face_roller_centres(d, FACE_ROLLER['pitch'], R))]

    def roller_xml(name, cx, cy, zc, i):
        R, m, h = FACE_ROLLER['d'] / 2, FACE_ROLLER['mass'], FACE_ROLLER['height']
        return ('<body name="%s" pos="%.4f %.4f %.4f">'
                '<joint name="%sj" type="hinge" axis="0 0 1" armature=".001" damping="%.4g" frictionloss="%.6g"/>'
                '<inertial pos="0 0 0" mass="%.3f" diaginertia="%.5f %.5f %.5f"/>'
                '<geom class="wall" type="cylinder" size="%.4f %.4f" rgba="%s"/></body>'
                % (name, cx, cy, zc, name, FACE_ROLLER['damping'], cfg['face_bearing_drag'], m,
                   m * (3 * R * R + h * h) / 12, m * (3 * R * R + h * h) / 12, m * R * R / 2,
                   R, h / 2, '.80 .82 .86 1' if i % 2 else '.62 .66 .72 1'))

    if cfg['unjam']:
        piv = d['P0'] if cfg['face_hinge'] == 'upstream' else d['P1']
        segs = (list(zip(d['face_pts'][:-1], d['face_pts'][1:])) if d['face_pts'] is not None
                else [(d['P0'], d['P1'])])
        fgeoms = []
        if fr:                                # rollers ride on the swinging face: nested bodies
            segs = []
            fgeoms = [roller_xml(nm, c[0] - piv[0], c[1] - piv[1], zc, i) for i, (nm, c, zc) in enumerate(fr)]
        for i, (A, Bp) in enumerate(segs):
            mid = (A + Bp) / 2
            ln = float(np.linalg.norm(Bp - A))
            ang = math.atan2(Bp[1] - A[1], Bp[0] - A[0])
            off = np.array([-math.sin(ang), math.cos(ang), 0.]) * PLOUGH['thickness'] / 2
            c = mid + off
            fgeoms.append('<geom name="plough%d" class="wall" type="box" size="%.4f %.4f %.4f" pos="%.4f %.4f %.4f" euler="0 0 %.6f" rgba=".78 .45 .25 1"/>'
                          % (i, ln / 2, PLOUGH['thickness'] / 2, PLOUGH['height'] / 2,
                             c[0] - piv[0], c[1] - piv[1], PLOUGH['bottom_gap'] + PLOUGH['height'] / 2, ang))
        # child bodies (the rollers) come after the geoms, which is the order MJCF expects
        joint_xml = '<joint name="facej" type="hinge" axis="0 0 1" armature="100000"/>'
        inertial_xml = '<inertial pos="0 0 0" mass="20" diaginertia="1 1 1"/>'
        if dynamic_face:
            delta = math.radians(cfg['face_swing_deg'])
            lo, hi = min(0., delta), max(0., delta)
            joint_xml = (f'<joint name="facej" type="hinge" axis="0 0 1" limited="true" '
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
            face_actuator = (f'<position name="face_drive" joint="facej" kp="{cfg["face_kp"]}" '
                             f'kv="{cfg["face_kv"]}" ctrllimited="true" ctrlrange="{lo} {hi}" '
                             f'forcelimited="true" forcerange="{-torque} {torque}"/>')
        body.append('<body name="face" pos="%.4f %.4f 0">%s%s'
                    '<geom type="cylinder" size=".03 .22" pos="0 0 .22" contype="0" conaffinity="0" rgba=".2 .2 .2 1"/>'
                    '%s</body>'
                    % (piv[0], piv[1], joint_xml, inertial_xml, ''.join(fgeoms)))
        wall_edges = [('lane_out', lane_a, lane_b)]
    else:
        if fr:                                # static roller face: free bodies at fixed positions
            body += [roller_xml(nm, c[0], c[1], zc, i) for i, (nm, c, zc) in enumerate(fr)]
            wall_edges = []
        else:
            wall_edges = ([('plough%d' % i, A, Bp) for i, (A, Bp) in enumerate(zip(d['face_pts'][:-1], d['face_pts'][1:]))]
                          if d['face_pts'] is not None else [('plough', d['P0'], d['P1'])])
        wall_edges.append(('lane_out', lane_a, lane_b))
    for tag, A, Bp in wall_edges:
        mid = (A + Bp) / 2
        ln = float(np.linalg.norm(Bp - A))
        if ln < .001:
            # only a degenerate edge (a lane_out wall of zero length when the bend sits at the belt end)
            # is dropped. The threshold used to be 0.05 m, which silently deleted 19 of the 24 boxes of a
            # STATIC curved face -- every segment steeper than ~27 deg -- leaving a 0.8 m hole in the
            # plough for any curve run that is not the hinged swing body (--no-unjam / --unjam-mode pulse)
            continue
        ang = math.atan2(Bp[1] - A[1], Bp[0] - A[0])
        off = np.array([-math.sin(ang), math.cos(ang), 0.]) * PLOUGH['thickness'] / 2
        c = mid + off
        static.append('<geom name="%s" class="wall" type="box" size="%.4f %.4f %.4f" pos="%.4f %.4f %.4f" euler="0 0 %.6f" rgba=".62 .5 .38 1"/>'
                      % (tag, ln / 2, PLOUGH['thickness'] / 2, PLOUGH['height'] / 2,
                         c[0], c[1], PLOUGH['bottom_gap'] + PLOUGH['height'] / 2, ang))
    skirts = [('skirt_out', cfg['belt_w'] + SKIRT['thickness'] / 2, d['belt_x0'], d['skirt_out_end'], .004),
              ('skirt_in', cfg['lane_y'] - SKIRT['thickness'] / 2, d['belt_x0'],
               d['sb_x0'] if cfg['side_belt'] else d['belt_x1'], .004)]
    # M's skirts stand on the weigh frame (the load cells carry them); B's on the fixed frame
    for pre, key in (('bskirt', 'buffer'), ('mskirt', 'measure')):
        for side, sign in (('in', -1), ('out', 1)):
            skirts.append(('%s_%s' % (pre, side), st['y_c'] + sign * (st['width_m'] + SKIRT['thickness']) / 2,
                           st[key]['x0'], st[key]['x1'], st['top_z'] + .004))
    for name, y, xa, xb, z0 in skirts:
        static.append('<geom name="%s" class="wall" type="box" size="%.4f %.4f %.4f" pos="%.4f %.4f %.4f" rgba=".55 .55 .6 .45"/>'
                      % (name, (xb - xa) / 2, SKIRT['thickness'] / 2, SKIRT['height'] / 2,
                         (xa + xb) / 2, y, z0 + SKIRT['height'] / 2))
    # the exit plane and the measuring plane, drawn only
    lines = ['<geom name="exit_line" type="box" size=".006 %.4f .004" pos="%.4f %.4f .006" '
             'contype="0" conaffinity="0" rgba="1 .6 .15 .85"/>'
             % (cfg['lane_w'] / 2, d['exit_x'], cfg['lane_y'] + cfg['lane_w'] / 2),
             '<geom name="final_line" type="box" size=".006 %.4f .004" pos="%.4f %.4f %.4f" '
             'contype="0" conaffinity="0" rgba="1 .15 .15 .9"/>'
             % (cfg['lane_w'] / 2, d['final_x'], cfg['lane_y'] + cfg['lane_w'] / 2, .006)]
    # the floor the separator stands on catches everything (no bins modelled); every beam, camera and the
    # volume scanner as hardware (singulator/devices.py)
    static.append('<geom name="floor" class="equip" type="plane" size="30 30 .1" pos="1 .5 %.4f" material="grid"/>'
                  % st['separator']['floor_z'])
    static += devices.xml(d['devices'])
    static += lines
    if cfg.get('feeder_release', 'predict') == 'lane':
        # the feed belt's next-release line, drawn only: every released lump's rear must be past it
        static.append('<geom name="release_line" type="box" size=".006 %.4f .004" pos="%.4f %.4f .006" '
                      'contype="0" conaffinity="0" rgba=".15 .45 1 .9"/>'
                      % (cfg['lane_w'] / 2, d['lane_out_start'], cfg['lane_y'] + cfg['lane_w'] / 2))

    sep_body, sep_equality, sep_actuator = separator.embed(st['sep'], st['separator']['origin'],
                                                           cfg['separator_friction'])
    body.append(sep_body)
    # the standalone model's defaults (steel, density 7850, joint damping 2), as equipment touching lumps only
    sep_default = ('<default class="separator"><geom contype="2" conaffinity="0" priority="1" density="7850" '
                   'rgba=".40 .43 .49 1" friction="%s .005 .0001"/><joint damping="2"/></default>'
                   % cfg['friction_steel'])
    for k, b in enumerate(blocks):
        rgba = b.get('rgba', (.2, .2, .23, 1) if b['material'] == 'coal' else (.66, .58, .46, 1))
        body.append('<body name="b%d" pos="%.2f -4 2"><freejoint name="bj%d"/>'
                    '<geom name="bg%d" class="rock" type="mesh" mesh="blk%d" density="%.12g" rgba="%s"/></body>'
                    % (k, -6 - 1.2 * k, k, k, k, b['density'], f3(rgba)))

    extent = d['view_x1'] - d['belt_x0'] + .6
    top_h = max(1.8, extent * cfg['top_h'] / cfg['width'])
    camera = ('<camera name="top" pos="%.3f %.3f 6" xyaxes="1 0 0 0 1 0" orthographic="true" fovy="%.3f"/>'
              % ((d['belt_x0'] + d['view_x1']) / 2, cfg['belt_w'] / 2, top_h))
    xml = f"""<mujoco model="plough_singulator">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="{cfg['dt']}" gravity="0 0 -9.81" cone="elliptic" impratio="10" integrator="implicitfast" iterations="{cfg['solver_iterations']}" tolerance="{cfg['solver_tolerance']}"><flag multiccd="enable"/></option>
  <size memory="300M"/>
  <visual><global offwidth="{cfg['width']}" offheight="{cfg['height']}"/><quality shadowsize="4096"/>
    <headlight ambient=".45 .45 .45" diffuse=".55 .55 .55" specular=".1 .1 .1"/></visual>
  <default>
    <geom condim="{cfg['contact_condim']}" solref="{cfg['solref']} {cfg['dampratio']}" solimp=".95 .99 .001" friction="{cfg['friction_block']} {cfg['torsional_friction']} {cfg['rolling_friction']}"/>
    <default class="equip"><geom contype="2" conaffinity="0"/></default>
    <default class="belt"><geom contype="2" conaffinity="0" priority="1" friction="{cfg['friction_belt']} .005 .0001"/></default>
    <default class="wall"><geom contype="2" conaffinity="0" priority="1" friction="{cfg['friction_steel']} .005 .0001"/></default>
    <default class="rock"><geom contype="1" conaffinity="3"/></default>{sep_default}
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".92 .95 .98" rgb2=".62 .7 .8" width="64" height="64"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".86 .87 .88" rgb2=".78 .79 .8" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="30 30"/>
    {chr(10).join(assets)}
  </asset>
  <worldbody>
    <light pos="1 -2 5" dir="0 .3 -1" diffuse=".5 .5 .5" castshadow="true"/>
    {camera}
    {chr(10).join(static)}
    {chr(10).join(body)}
  </worldbody>
</mujoco>"""
    actuators = face_actuator + sep_actuator
    xml = xml.replace('.005 .0001"', f'{cfg["torsional_friction"]} {cfg["rolling_friction"]}"')
    if sep_equality:
        xml = xml.replace('</mujoco>', '<equality>' + sep_equality + '</equality></mujoco>')
    return xml.replace('</mujoco>', '<actuator>' + actuators + '</actuator></mujoco>') if actuators else xml


def motion_clearance(cfg, d, steps=26):
    """Sweep each moving mechanism through its stroke; smallest distance to any other collidable equipment.

    Returns {mechanism: dict(min_distance_m, pair, at)}; negative distance = the parts intersect.
    """
    model = mujoco.MjModel.from_xml_string(build_xml(cfg, d, []))
    data = mujoco.MjData(model)
    # Slat chains start with all slide joints at zero in model.qpos0. Put each slat at its actual
    # initial chain position before sweeping: otherwise the whole side belt sits at x = 0 and a face
    # collision farther downstream is silently absent from this equipment-only check.
    chain_q = []
    chain_x = []
    for i in range(d['n_sbslat']):
        chain_q.append(model.jnt_qposadr[model.joint('vj%d' % i).id])
        chain_x.append(d['sb_x0'] + i * SIDE_BELT['pitch'])
    name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or (
        'geom%d@%s' % (g, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])))
    collidable = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]

    def subtree(root):
        bodies = {root}
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p == root:
                    bodies.add(b)
                    break
                p = model.body_parentid[p]
        return [g for g in collidable if model.geom_bodyid[g] in bodies]

    moves = []                                  # (mechanism, its geoms, its joint, stroke positions)
    if cfg['unjam']:
        moves.append(('face', subtree(model.body('face').id), model.jnt_qposadr[model.joint('facej').id],
                      np.radians(np.linspace(0., cfg['face_swing_deg'], steps))))
    elif cfg['face_kind'] == 'rollers':
        rollers = [g for g in collidable if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                               model.geom_bodyid[g]) or '').startswith('fr')]
        moves.append(('face_rollers', rollers, None, [0.]))
    # the separator plate and linkage over the whole stroke (degrees), posed through the closed loop
    c = d['station']['sep']
    moves.append(('separator', subtree(model.body(separator.PREFIX + 'plate').id)
                      + subtree(model.body(separator.PREFIX + 'barrel').id),
                      lambda deg: separator.place(model, data, c, deg), np.linspace(c.closed_deg, c.open_deg, steps)))

    def pose(q, p):
        data.qpos[:] = model.qpos0
        data.qpos[chain_q] = chain_x
        if callable(q):
            q(p)
        elif q is not None:
            data.qpos[q] = p
        mujoco.mj_forward(model, data)

    ft = np.zeros(6)

    def distance(ga, gb):
        """Signed distance, or None when the routine's own witness points contradict it (it returns 0
        for some separated box pairs with parallel faces, with witness points half a metre apart). A clear
        pair whose witness points are within 10 % of the distance counts, at the smaller of the two (box-box
        distances run ~1 mm off at 45 mm: the separator's side wall against the measuring belt)."""
        dist = float(mujoco.mj_geomDistance(model, data, ga, gb, .05, ft))
        w = float(np.linalg.norm(ft[:3] - ft[3:]))
        if dist >= .05 or abs(w - abs(dist)) < 1e-6:
            return dist
        if dist > 0. and abs(w - dist) < .1 * dist:
            return min(dist, w)
        return None

    out = {}
    for mech, mine, q, positions in moves:
        others = [g for g in collidable if g not in mine]
        best, unresolved = (math.inf, None, None), []
        for p in positions:
            pose(q, p)
            retry = []
            for ga in mine:
                for gb in others:
                    if model.geom_type[gb] != mujoco.mjtGeom.mjGEOM_PLANE and (
                            np.linalg.norm(data.geom_xpos[ga] - data.geom_xpos[gb])
                            > model.geom_rbound[ga] + model.geom_rbound[gb] + .05):
                        continue
                    dist = distance(ga, gb)
                    if dist is None:
                        retry.append((ga, gb))
                    elif dist < best[0]:
                        best = (dist, (name(ga), name(gb)), float(p))
            for ga, gb in retry:        # re-measure just either side of the degenerate pose (finer than the sweep step)
                vals = []
                for e in ((1e-3, 5e-3, 1e-2) if q is not None else ()):
                    for sgn in (-1, 1):
                        pose(q, p + sgn * e)
                        vals.append(distance(ga, gb))
                    if any(v is not None for v in vals):
                        break
                vals = [v for v in vals if v is not None]
                if not vals:
                    unresolved.append((name(ga), name(gb), float(p)))
                elif min(vals) < best[0]:
                    best = (min(vals), (name(ga), name(gb)), float(p))
            pose(q, p)
        out[mech] = dict(min_distance_m=round(best[0], 5), pair=best[1], at=best[2], unresolved=unresolved)
    return out
