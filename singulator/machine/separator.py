"""Hinged coal/gangue flip separator (排料板): geometry, MJCF and checks. SI units; dimensions are provisional.

Plate down (closed_deg): coal slides along it and leaves over the far end. Plate up (open_deg): the inlet
end rises, the pivot stays put, and gangue drops through the gap between the feeding belt and the plate.

Two users of the same geometry:
  * designs/flip_separator/model.py -- the standalone model (renders, interactive viewer, checks.json);
    build() is its whole MJCF, with the keyframes 'closed_coal' / 'open_gangue';
  * the line (machine/station.py, machine/assembly.py) -- embed() places the same parts behind the
    measuring belt, names prefixed 'sep_', and place() poses the linkage in the running model.

Config, layout() and state() are pure kinematics; everything that builds or poses a model imports MuJoCo when it
is called, so the line's geometry (machine/layout.py) can be derived without it.

The preview prescribes consistent joint positions (kinematics, not a load test).
The MJCF contains an actual barrel hinge / piston slide / pin closure loop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np

# rib pitch of the standalone 1.20 m plate (11 ribs, 35 mm from each edge); a narrower plate keeps the pitch
RIB_PITCH = (1.20 - .07) / 10


@dataclass
class Config:
    width: float = 1.20  # the standalone model (the line's plate is for_width(station_w), 0.70 m)
    length: float = 1.60
    pivot_fraction: float = .72  # distance from inlet / length
    inlet_height: float = 1.45
    closed_deg: float = 20.
    open_deg: float = 75.
    thickness: float = .018
    rib_height: float = .014
    rib_count: int = 11
    deck_above_pivot: float = .24  # deck surface above shaft, in plate coordinates
    cylinder_closed_pin_length: float = .80
    cylinder_stroke: float = .45
    # side walls along both deck edges, riding on the plate (user 2026-09-30: 加侧挡板). Without them a coal
    # lump sliding down the closed plate can drift off its side: seed 4007 left it 0.2 m past the edge
    side_wall_height: float = .25  # above the deck; 0 = none
    side_wall_thickness: float = .01

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('All parameters must be finite')
        if min(self.width, self.length, self.inlet_height, self.thickness,
               self.cylinder_closed_pin_length, self.cylinder_stroke, self.deck_above_pivot) <= 0:
            raise ValueError('Dimensions must be positive')
        if self.side_wall_height < 0 or self.side_wall_thickness <= 0:
            raise ValueError('Side wall height must be >= 0 and its thickness > 0')
        if not 0 < self.pivot_fraction < 1 or not 0 <= self.closed_deg < self.open_deg < 89:
            raise ValueError('Require 0 < pivot_fraction < 1 and 0 <= closed < open < 89 degrees')
        if self.rib_count < 2 or self.rib_height < 0:
            raise ValueError('Invalid ribs')


def for_width(width, **kw):
    """The standalone plate at another width: same rib pitch, everything else unchanged."""
    return Config(width=width, rib_count=max(2, int(round((width - .07) / RIB_PITCH)) + 1), **kw)


def ry(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def layout(c):
    c.validate()
    a = math.radians(c.closed_deg)
    p = np.array([c.pivot_fraction*c.length*math.cos(a)-c.deck_above_pivot*math.sin(a), 0,
                  c.inlet_height-c.pivot_fraction*c.length*math.sin(a)-c.deck_above_pivot*math.cos(a)])
    cy = -c.width/2-.32
    fixed = p + np.array([.65, cy, .60])
    lug = np.array([min(.35, (1-c.pivot_fraction)*c.length*.80), cy, -.16])
    return p, fixed, lug


def state(c, deg):
    p, fixed, lug = layout(c)
    r = ry(math.radians(deg))
    moving = p+r@lug
    delta = moving-fixed
    pin_length = float(np.linalg.norm(delta))
    phi = math.atan2(delta[0], delta[2])
    inlet = p+r@np.array([-c.pivot_fraction*c.length, 0, c.deck_above_pivot])
    outlet = p+r@np.array([(1-c.pivot_fraction)*c.length, 0, c.deck_above_pivot])
    return dict(angle_deg=deg, pin_length_m=pin_length,
                piston_extension_m=pin_length-c.cylinder_closed_pin_length,
                barrel_angle_rad=phi, inlet=inlet.tolist(), outlet=outlet.tolist(),
                moving_pin=moving.tolist())


def fmt(a):
    return ' '.join(f'{float(v):.9g}' for v in a)


def node(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in kw.items()})


def build(c):
    import mujoco
    p, fixed, lug = layout(c)
    root = ET.Element('mujoco', model='standalone_flip_separator')
    node(root, 'compiler', angle='radian', autolimits='true')
    node(root, 'option', timestep='.001', integrator='implicitfast', gravity='0 0 -9.81')
    visual = node(root, 'visual')
    node(visual, 'global', offwidth='1440', offheight='900')
    node(visual, 'headlight', diffuse='.65 .65 .65', ambient='.35 .35 .35')
    node(visual, 'rgba', haze='.94 .95 .97 1')
    assets = node(root, 'asset')
    node(assets, 'texture', name='sky', type='skybox', builtin='gradient',
         rgb1='.88 .91 .95', rgb2='.98 .98 .99', width='512', height='3072')
    default = node(root, 'default')
    node(default, 'geom', rgba='.40 .43 .49 1', density='7850', friction='.45 .005 .0001')
    node(default, 'joint', damping='2')
    w = node(root, 'worldbody')
    node(w, 'light', pos='0 -2 5', dir='0 0 -1', directional='true')
    node(w, 'geom', name='ground', type='plane', size='5 5 .1', rgba='.94 .95 .97 1')

    def box(parent, name, pos, size, **kw):
        return node(parent, 'geom', name=name, type='box', pos=fmt(pos), size=fmt(size), **kw)

    def cyl(parent, name, a, b, radius, **kw):
        return node(parent, 'geom', name=name, type='cylinder', fromto=fmt([*a, *b]), size=radius, **kw)

    # Steel frame leaves the entire material drop window open.
    x0, x1 = -.12, p[0]+.84
    side = c.width/2+.16
    for y in (-side, side):
        box(w, f'base_side_{y}', [(x0+x1)/2, y, .085], [(x1-x0)/2, .045, .065])
        for x in (x0, x1):
            top = c.inlet_height+.03 if x == x0 else fixed[2]+.06
            box(w, f'corner_{x}_{y}', [x, y, top/2], [.045, .045, top/2])
        box(w, f'bearing_post_{y}', [p[0], y, p[2]/2], [.06, .055, p[2]/2-.06])
        box(w, f'bearing_housing_{y}', [p[0], y, p[2]], [.10, .065, .095],
            rgba='.10 .40 .25 1', contype='0', conaffinity='0')
        cyl(w, f'bearing_{y}', [p[0], y-.07, p[2]], [p[0], y+.07, p[2]], .072,
            rgba='.68 .70 .72 1', contype='0', conaffinity='0')
    for x in (x0, x1):
        box(w, f'base_end_{x}', [x, 0, .085], [.045, side+.045, .065])
    box(w, 'cylinder_mount', [(fixed[0]+x1)/2, (fixed[1]-side)/2, fixed[2]],
        [abs(x1-fixed[0])/2+.08, abs(fixed[1]+side)/2+.09, .075],
        contype='0', conaffinity='0')

    plate = node(w, 'body', name='plate', pos=fmt(p))
    node(plate, 'joint', name='plate_hinge', type='hinge', axis='0 1 0',
         range=fmt(np.radians([c.closed_deg, c.open_deg])))
    mid = (.5-c.pivot_fraction)*c.length
    h = c.deck_above_pivot
    box(plate, 'deck', [mid, 0, h-c.thickness/2], [c.length/2, c.width/2, c.thickness/2])
    for i, y in enumerate(np.linspace(-c.width/2+.035, c.width/2-.035, c.rib_count)):
        if c.rib_height:
            box(plate, f'fixed_rib_{i}', [mid, y, h+c.rib_height/2],
                [c.length/2, .008, c.rib_height/2], rgba='.61 .64 .68 1')
    for x in (-c.pivot_fraction*c.length+.08, (1-c.pivot_fraction)*c.length-.08):
        box(plate, f'underbeam_{x}', [x, 0, h-.055], [.025, c.width/2, .035])
    if c.side_wall_height:
        # the side walls stand on the deck edges over the whole length and swing with the plate
        for wall, sign in (('in', -1), ('out', 1)):
            box(plate, f'side_wall_{wall}', [mid, sign*(c.width/2+c.side_wall_thickness/2), h+c.side_wall_height/2],
                [c.length/2, c.side_wall_thickness/2, c.side_wall_height/2], rgba='.52 .55 .60 1')
    cyl(plate, 'shaft', [0, -side-.10, 0], [0, side+.10, 0], .04,
        rgba='.60 .62 .65 1')
    green = dict(rgba='.12 .43 .28 1')
    # Saddles connect the lowered shaft to the underside, never through the conveying face.
    for y in (-c.width/2+.055, c.width/2-.055):
        box(plate, f'shaft_saddle_{y}', [0, y, (h-c.thickness)/2],
            [.06, .035, (h-c.thickness)/2], **green)
    # Route the linkage outside the bearing post before dropping to the lower pin.
    # A low straight crossbar would sweep straight through the fixed bearing post.
    bridge_z = h-.055
    box(plate, 'moving_lug_bridge', [lug[0], (lug[1]-c.width/2)/2, bridge_z],
        [.035, abs(lug[1]+c.width/2)/2+.035, .025], **green)
    box(plate, 'moving_lug_arm', [lug[0], lug[1], (bridge_z+lug[2])/2],
        [.035, .035, (bridge_z-lug[2])/2+.025], **green)
    node(plate, 'site', name='plate_pin', pos=fmt(lug), size='.018', rgba='1 .7 0 1')

    # A true R-P-R linkage: fixed eye hinge, sliding piston, moving eye equality.
    barrel = node(w, 'body', name='barrel', pos=fmt(fixed))
    node(barrel, 'joint', name='barrel_hinge', type='hinge', axis='0 1 0', limited='false')
    base = c.cylinder_closed_pin_length
    red = dict(rgba='.73 .045 .035 1', contype='0', conaffinity='0')
    cyl(barrel, 'barrel_shell', [0, 0, .055], [0, 0, base-.13], .063, **red)
    cyl(barrel, 'fixed_eye', [0, -.042, 0], [0, .042, 0], .04, **red)
    rod = node(barrel, 'body', name='rod', pos=fmt([0, 0, base]))
    node(rod, 'joint', name='piston', type='slide', axis='0 0 1',
         range=fmt([0, c.cylinder_stroke]))
    cyl(rod, 'piston_rod', [0, 0, -(base-.20)], [0, 0, -.03], .027,
        rgba='.77 .79 .81 1', contype='0', conaffinity='0')
    cyl(rod, 'moving_eye', [0, -.043, 0], [0, .043, 0], .043, **red)
    node(rod, 'site', name='rod_pin', pos='0 0 0', size='.015', rgba='1 .7 0 1')
    eq = node(root, 'equality')
    node(eq, 'connect', name='moving_pin_closure', site1='rod_pin', site2='plate_pin', solref='.006 1')
    actuator = node(root, 'actuator')
    node(actuator, 'position', name='cylinder_position', joint='piston', kp='1000000', kv='30000',
         ctrlrange=fmt([0, c.cylinder_stroke]), forcerange='-20000 20000')
    # Context only: this blue upstream surface is not a driven conveyor.
    box(w, 'upstream_reference', [-.42, 0, c.inlet_height-.05], [.40, c.width/2, .035],
        rgba='.21 .39 .49 .45', contype='0', conaffinity='0')
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    pose(model, data, c, c.closed_deg)
    keyframe = node(root, 'keyframe')
    node(keyframe, 'key', name='closed_coal', qpos=fmt(data.qpos), ctrl=fmt(data.ctrl))
    pose(model, data, c, c.open_deg)
    node(keyframe, 'key', name='open_gangue', qpos=fmt(data.qpos), ctrl=fmt(data.ctrl))
    ET.indent(root)
    return ET.tostring(root, encoding='unicode')


def pose(model, data, c, deg):
    import mujoco
    s = state(c, deg)
    for name, value in [('plate_hinge', math.radians(deg)),
                        ('barrel_hinge', s['barrel_angle_rad']), ('piston', s['piston_extension_m'])]:
        data.qpos[model.joint(name).qposadr[0]] = value
    data.qvel[:] = 0
    data.ctrl[0] = s['piston_extension_m']
    mujoco.mj_forward(model, data)
    return s


def verify(model, data, c):
    errors, extensions, lengths, angles = [], [], [], np.linspace(c.closed_deg, c.open_deg, 181)
    unwanted_contacts = []
    for deg in angles:
        s = pose(model, data, c, float(deg))
        errors.append(float(np.linalg.norm(data.site('rod_pin').xpos-data.site('plate_pin').xpos)))
        extensions.append(s['piston_extension_m'])
        lengths.append(s['pin_length_m'])
        for contact in data.contact:
            if contact.dist < -1e-5:
                unwanted_contacts.append(dict(angle_deg=float(deg), depth_m=float(-contact.dist),
                    geoms=[model.geom(int(g)).name for g in contact.geom]))
    if max(errors) > 1e-6:
        raise ValueError('Cylinder pin loop does not close')
    if min(extensions) < 0 or max(extensions) > c.cylinder_stroke:
        raise ValueError('Cylinder stroke exceeded; adjust linkage/dimensions')
    derivatives = np.diff(lengths)/np.diff(np.radians(angles))
    if not (np.all(derivatives > .01) or np.all(derivatives < -.01)):
        raise ValueError('Cylinder dead centre or direction reversal within working range')
    if unwanted_contacts:
        raise ValueError(f'Penetrating physical geometry: {unwanted_contacts[:3]}')
    return dict(status='KINEMATIC_GEOMETRY_CHECKED', sample_count=len(angles),
                max_pin_closure_error_m=max(errors), piston_extension_range_m=[min(extensions), max(extensions)],
                required_stroke_m=max(extensions)-min(extensions),
                minimum_cylinder_moment_arm_m=float(np.min(np.abs(derivatives))),
                physical_geometry_penetrations=unwanted_contacts,
                excluded_from_contact_check=['bearing housings', 'cylinder and fixed pin support details',
                                             'upstream reference surface'],
                NOT_VALIDATED=['material separation', 'hydraulic sizing', 'structural strength',
                               'detector timing', 'all hardware clearances'])


def material_clearance(xml, c):
    """Distance checks with 300/500 mm spherical probes; prescribed paths, not DEM.

    Explicit distances also inspect contact-disabled visual hardware. Coal probes run
    along the face/its straight downstream extension; gangue probes follow an
    assumed free-flight path from the upstream lip. Departure tipping is not solved.
    Only velocities 0.4/1.2 m/s and laterally contained probes are covered.
    """
    import mujoco
    tree = ET.fromstring(xml)
    body = node(tree.find('worldbody'), 'body', name='clearance_probe', mocap='true')
    node(body, 'geom', name='clearance_probe_geom', type='sphere', size='.25',
         contype='0', conaffinity='0', rgba='1 .5 0 .35')
    m = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
    d = mujoco.MjData(m)
    probe = m.geom('clearance_probe_geom').id
    hardware = [i for i in range(m.ngeom) if m.geom(i).name not in
                ('ground', 'upstream_reference', 'clearance_probe_geom', 'deck')
                and not m.geom(i).name.startswith(('fixed_rib_', 'side_wall_'))]    # the conveying face
    p, _, _ = layout(c)
    minima, hits = {}, []
    def check(label, centre, targets):
        d.mocap_pos[0] = centre
        mujoco.mj_forward(m, d)
        for g in targets:
            dist = mujoco.mj_geomDistance(m, d, probe, g, 5., None)
            if label not in minima or dist < minima[label]['clearance_m']:
                minima[label] = dict(clearance_m=float(dist), geom=m.geom(g).name,
                                     probe_center_m=np.asarray(centre).tolist())
            if dist < -1e-5 and len(hits) < 10:
                hits.append(dict(path=label, geom=m.geom(g).name, penetration_m=float(-dist)))

    for radius in (.15, .25):
        m.geom_size[probe, 0] = radius
        for deg in np.linspace(c.closed_deg, c.open_deg, 181):
            pose(m, d, c, float(deg))
            r = ry(math.radians(deg))
            for y in (-c.width/2+radius, 0, c.width/2-radius):
                # Include the shaft station explicitly: a regular grid can miss
                # a narrow protrusion between its sample positions.
                stations = np.unique(np.r_[np.linspace(-c.pivot_fraction*c.length,
                    (1-c.pivot_fraction)*c.length, 17), 0., layout(c)[2][0]])
                for x in stations:
                    check('face_to_hardware', p+r@np.array([x, y, c.deck_above_pivot+c.rib_height+radius+.005]), hardware)
        for mode in ('coal_departure', 'gangue_drop'):
            deg = c.closed_deg if mode == 'coal_departure' else c.open_deg
            pose(m, d, c, deg)
            r = ry(math.radians(deg))
            for speed in (.4, 1.2):
                for y in (-c.width/2+radius, 0, c.width/2-radius):
                    start = (np.array(state(c, deg)['outlet'])+r@np.array([0, y, radius+c.rib_height+.005])
                             if mode == 'coal_departure' else np.array([-.02, y, c.inlet_height+radius]))
                    velocity = r@np.array([speed, 0, 0]) if mode == 'coal_departure' else np.array([speed, 0, 0])
                    for t in np.arange(0., 1.2, .01):
                        # Coal: inspect the continuation corridor, not a fictitious
                        # free fall while the trailing half is still supported by the lip.
                        gravity = np.array([0, 0, -4.905*t*t]) if mode == 'gangue_drop' else np.zeros(3)
                        centre = start+velocity*t+gravity
                        if centre[2]-radius < .25:
                            break  # below the specified inspection region, receiver not modelled
                        targets = hardware+([m.geom('deck').id] if mode == 'gangue_drop' else [])
                        check(mode, centre, targets)
    report = dict(status='PRESCRIBED_PATH_CLEARANCE_ONLY', probe_diameters_m=[.30, .50],
                  speeds_m_s=[.4, 1.2], stop_height_probe_bottom_m=.25,
                  minima=minima, intersections=hits,
                  limits='Coal uses a straight face-extension corridor; gangue uses assumed free flight. No departure tipping, tumbling, impact rebound, oversize or timing validation')
    if hits:
        raise ValueError(f'Material path intersects hardware: {hits}')
    return report


# ---- in the line -----------------------------------------------------------------------------------
PREFIX = 'sep_'


def embed(c, origin, deck_friction, secondary='.005 .0001'):
    """The standalone parts for the line: (worldbody XML, equality XML, actuator XML).

    Everything build() puts in the world except the light, the ground and the blue reference surface,
    wrapped in one static body at `origin` (the standalone origin: inlet edge x, plate centre line y,
    ground z) with childclass 'separator' (equipment: touches lumps only; steel, density 7850, joint
    damping 2 -- the standalone defaults). Every name gets the prefix 'sep_'. The deck and the ribs carry
    `deck_friction` (the lining the material slides on) and the line's `secondary` (torsional and rolling)
    coefficients."""
    root = ET.fromstring(build(c))
    wrap = ET.Element('body', {'name': PREFIX + 'frame', 'pos': fmt(origin), 'childclass': 'separator'})
    for el in root.find('worldbody'):
        if el.tag == 'light' or el.get('name') in ('ground', 'upstream_reference'):
            continue
        wrap.append(el)
    equality, actuator = root.find('equality'), root.find('actuator')
    for top in (wrap, equality, actuator):
        for el in top.iter():
            for key in ('name', 'site1', 'site2', 'joint'):
                if key in el.attrib and el is not wrap:
                    el.set(key, PREFIX + el.attrib[key])
    for el in wrap.iter('geom'):
        if el.get('name') == PREFIX + 'deck' or el.get('name', '').startswith(PREFIX + 'fixed_rib_'):
            el.set('friction', '%.6g %s' % (deck_friction, secondary))
    ET.indent(wrap)
    inner = lambda top: ''.join(ET.tostring(el, encoding='unicode') for el in top)
    return ET.tostring(wrap, encoding='unicode'), inner(equality), inner(actuator)


def place(model, data, c, deg):
    """Pose the linkage of the embedded separator (joints only; nothing else in the model is touched)."""
    s = state(c, deg)
    for name, value in (('plate_hinge', math.radians(deg)), ('barrel_hinge', s['barrel_angle_rad']),
                        ('piston', s['piston_extension_m'])):
        j = model.joint(PREFIX + name)
        data.qpos[j.qposadr[0]] = value
        data.qvel[j.dofadr[0]] = 0.
    data.ctrl[model.actuator(PREFIX + 'cylinder_position').id] = s['piston_extension_m']
    return s
