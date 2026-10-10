"""The MuJoCo model of the line, put together from its sections; what is what in the compiled model; and the
moving-part clearance sweep.

build_xml() asks each section for its parts and places them in one fixed order. The order is part of the
model: bodies, joints and geoms are numbered by it, MuJoCo's contact and constraint lists follow the numbers,
and a result depends on them in its last digits -- which a contact-rich run amplifies. The order below is the one
every batch since 2026-09-30 was run with; keep it unless you mean to give up comparing with those runs
bit for bit.
"""
import math

import numpy as np

from . import feed_belt, plough, sensors, separator, side_gate, station
from .parts import SKIRT, f3

LUMP_BODY, LUMP_JOINT, LUMP_GEOM, LUMP_MESH = 'b%d', 'bj%d', 'bg%d', 'blk%d'

# contact categories, by the equipment a lump touches (bit i of trajectory.npz 'touch' = CATS[i]): 'belt' is the
# feed and main belts, 'buffer' the buffer belt with its skirts, 'weigher' the measuring belt with its (the weigh
# frame), 'device' the sensor hardware (machine/sensors.py), 'pusher' the side gate's plates (machine/side_gate.py)
CATS = ('other', 'belt', 'plough', 'block', 'skirt', 'lane_wall', 'side_belt', 'buffer', 'weigher',
        'separator', 'device', 'pusher')


def build_xml(cfg, d, blocks):
    st = d['station']
    secondary = f'{cfg["torsional_friction"]} {cfg["rolling_friction"]}'    # torsional and rolling coefficients
    half_y = (cfg['belt_w'] + 2 * SKIRT['thickness'] + .06) / 2
    y_mid = cfg['belt_w'] / 2

    # ---- bodies, in the fixed order (see the module docstring) -----------------------------------------
    face, face_actuator = plough.face_body(cfg, d)
    sep_body, sep_equality, sep_actuator = separator.embed(st['sep'], st['separator']['origin'],
                                                           cfg['separator_friction'], secondary)
    body = (plough.main_belt_body(d, y_mid, half_y)
            + feed_belt.bodies(d['feeder'], y_mid, half_y)
            + plough.tail_pulley_body(d, y_mid, half_y)
            + station.belt_bodies(st)
            + plough.side_belt_bodies(cfg, d)
            + [face, sep_body])
    for k, b in enumerate(blocks):
        rgba = b.get('rgba', (.2, .2, .23, 1) if b['material'] == 'coal' else (.66, .58, .46, 1))
        body.append('<body name="b%d" pos="%.2f -4 2"><freejoint name="bj%d"/>'
                    '<geom name="bg%d" class="rock" type="mesh" mesh="blk%d" density="%.12g" rgba="%s"/></body>'
                    % (k, -6 - 1.2 * k, k, k, k, b['density'], f3(rgba)))
    assets = ['<mesh name="blk%d" vertex="%s"/>' % (k, f3(b['vertices'].ravel())) for k, b in enumerate(blocks)]
    gate_actuators = ''
    if cfg['side_pusher'] != 'off':
        gate_bodies, gate_actuators = side_gate.hardware(cfg, d)
        body += gate_bodies

    # ---- fixed geoms ---------------------------------------------------------------------------------------
    side_skirts = plough.skirts(cfg, d) if cfg['side_pusher'] == 'off' else side_gate.skirts(cfg, d)
    static = plough.lane_wall(cfg, d) + side_skirts + station.skirts(st)
    # the floor the separator stands on catches everything (no bins modelled)
    static.append('<geom name="floor" class="equip" type="plane" size="30 30 .1" pos="1 .5 %.4f" material="grid"/>'
                  % st['separator']['floor_z'])
    static += sensors.xml(d['devices'])           # every beam, camera and the volume scanner as hardware
    static += plough.markers(cfg, d)

    # the standalone separator's defaults (steel, density 7850, joint damping 2), as equipment touching lumps only
    sep_default = ('<default class="separator"><geom contype="2" conaffinity="0" priority="1" density="7850" '
                   'rgba=".40 .43 .49 1" friction="%s %s"/><joint damping="2"/></default>'
                   % (cfg['friction_steel'], secondary))
    extent = d['view_x1'] - d['belt_x0'] + .6
    top_h = max(1.8, extent * cfg['top_h'] / cfg['width'])
    camera = ('<camera name="top" pos="%.3f %.3f 6" xyaxes="1 0 0 0 1 0" orthographic="true" fovy="%.3f"/>'
              % ((d['belt_x0'] + d['view_x1']) / 2, cfg['belt_w'] / 2, top_h))
    return f"""<mujoco model="plough_singulator">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="{cfg['dt']}" gravity="0 0 -9.81" cone="elliptic" impratio="10" integrator="implicitfast" iterations="{cfg['solver_iterations']}" tolerance="{cfg['solver_tolerance']}"><flag multiccd="enable"/></option>
  <size memory="300M"/>
  <visual><global offwidth="{cfg['width']}" offheight="{cfg['height']}"/><quality shadowsize="4096"/>
    <headlight ambient=".45 .45 .45" diffuse=".55 .55 .55" specular=".1 .1 .1"/></visual>
  <default>
    <geom condim="{cfg['contact_condim']}" solref="{cfg['solref']} {cfg['dampratio']}" solimp=".95 .99 .001" friction="{cfg['friction_block']} {secondary}"/>
    <default class="equip"><geom contype="2" conaffinity="0"/></default>
    <default class="belt"><geom contype="2" conaffinity="0" priority="1" friction="{cfg['friction_belt']} {secondary}"/></default>
    <default class="wall"><geom contype="2" conaffinity="0" priority="1" friction="{cfg['friction_steel']} {secondary}"/></default>
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
<equality>{sep_equality}</equality><actuator>{face_actuator + sep_actuator + gate_actuators}</actuator></mujoco>"""


def categorise(model):
    """Category index of every geom (see CATS), by the names the sections give their parts."""
    import mujoco
    G, B = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
    cat = np.zeros(model.ngeom, dtype=np.int8)
    for g in range(model.ngeom):
        gname = mujoco.mj_id2name(model, G, g) or ''
        bname = mujoco.mj_id2name(model, B, model.geom_bodyid[g]) or ''
        if bname in (plough.MAIN_BELT, feed_belt.BODY, feed_belt.DRUM, plough.TAIL):
            cat[g] = 1
        elif gname.startswith('plough') or (bname.startswith('fr') and bname[2:].isdigit()):
            cat[g] = 2
        elif gname.startswith('bg'):
            cat[g] = 3
        elif gname.startswith('skirt'):
            cat[g] = 4
        elif gname == 'lane_out':
            cat[g] = 5
        elif bname.startswith('vslat'):
            cat[g] = 6
        elif bname == station.BUFFER or gname.startswith('bskirt'):
            cat[g] = 7
        elif bname == station.MEASURE or gname.startswith('mskirt'):
            cat[g] = 8
        elif bname.startswith(separator.PREFIX):
            cat[g] = 9
        elif gname.startswith('dev_'):
            cat[g] = 10
        elif gname.startswith('push_'):
            cat[g] = 11
    return cat


def weigh_frame(model):
    """The geoms the load cells carry: the measuring belt and its skirts."""
    cat = categorise(model)
    return {g for g in range(model.ngeom) if cat[g] == CATS.index('weigher')}


def motion_clearance(cfg, d, steps=26):
    """Sweep each moving mechanism through its stroke; smallest distance to any other collidable equipment.

    Returns {mechanism: dict(min_distance_m, pair, at)}; negative distance = the parts intersect.
    """
    import mujoco
    model = mujoco.MjModel.from_xml_string(build_xml(cfg, d, []))
    data = mujoco.MjData(model)
    # Slat chains start with all slide joints at zero in model.qpos0. Put each slat at its actual
    # initial chain position before sweeping: otherwise the whole side belt sits at x = 0 and a face
    # collision farther downstream is silently absent from this equipment-only check.
    chain_q = []
    chain_x = []
    for i in range(d['n_sbslat']):
        chain_q.append(model.jnt_qposadr[model.joint('vj%d' % i).id])
        chain_x.append(d['sb_x0'] + i * plough.SIDE_BELT['pitch'])
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

    # (mechanism, its geoms, its joint, stroke positions)
    moves = [(plough.FACE, subtree(model.body(plough.FACE).id), model.jnt_qposadr[model.joint(plough.FACE_JOINT).id],
              np.radians(np.linspace(0., cfg['face_swing_deg'], steps)))]
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
        else:
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
                for e in (1e-3, 5e-3, 1e-2):
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
