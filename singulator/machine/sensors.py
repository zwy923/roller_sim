"""Sensors of the line as hardware: every light beam and camera with its mount, housing
and optics, and the regions each one must see. What the controllers read from them is modelled in
singulator/sensing/ (estimated outlines, debounced beams with their diagnoses, the scanner's verdict).

  * a light beam is a through-beam pair: an emitter and a receiver housing just outside the skirts (the
    skirt needs a slot at the beam), the beam drawn between them. Its line is the one sensing.beams.Beam
    tests, so moving the beam here moves the signal;
  * a camera has a housing on a mast or gantry, a MuJoCo <camera> with its lens (vertical field of view,
    image aspect), and a list of regions -- boxes in which lumps must be visible -- that its controller
    relies on. coverage() checks every corner of every region against the view frustum;
  * the volume scanner is three depth heads on a gantry over the measuring belt; every corner of the
    measuring zone must be seen by at least two of them.
Housings, masts and gantries are equipment: lumps hit them, equipment does not. Lens fields of view,
mount heights and every dimension here are placement assumptions, not a selected product.
"""
import math

import numpy as np

from .parts import LUMP_TOP, SKIRT

BEAM_HOUSING = (.02, .025, .04)      # half sizes (x, y, z) of an emitter / receiver housing
BEAM_LENS_UP = .012                  # the lens sits this far above the housing's bottom face
CAMERA_HOUSING = (.07, .05, .045)
HEAD_HOUSING = (.05, .04, .035)      # a depth head of the volume scanner
POST = .03                           # half section of a mast / gantry post
CAMERA = dict(fovy=45., aspect=16 / 9)       # overhead cameras (1920 x 1080)
LATENCY_S = .05                      # s: capture + processing; the controllers see the scene as it was this long ago
# longest a lump can keep a beam blocked while the belt under it runs (longer: something is stuck), s
MAX_BLOCK_S = dict(beam_feed=6., beam_in=6., beam_stop=6., beam_gangue=2.)
DEPTH = dict(fovy=50., aspect=4 / 3)         # scanner heads


def look_at(pos, target, x_hint=(1., 0., 0.)):
    """Camera axes (x right, y up in the image, z backwards: MuJoCo looks along -z) aimed at target."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
    x /= np.linalg.norm(x)
    return x, np.cross(z, x), z


def in_view(cam, P):
    """Which of the world points P (n, 3) lie inside the camera's view frustum."""
    R = np.stack([cam['x'], cam['y'], cam['z']], 1)
    pc = (np.atleast_2d(P) - cam['pos']) @ R
    depth = -pc[:, 2]
    ty = math.tan(math.radians(cam['fovy']) / 2)
    return (depth > 0) & (np.abs(pc[:, 0]) <= depth * ty * cam['aspect'] + 1e-9) & (np.abs(pc[:, 1]) <= depth * ty + 1e-9)


def corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _camera(name, role, pos, target, optics, sees, mount, x_hint=(1., 0., 0.)):
    x, y, z = look_at(pos, target, x_hint)
    return dict(name=name, role=role, pos=np.asarray(pos, float), x=x, y=y, z=z, sees=sees, mount=mount, **optics)


def layout(cfg, d):
    """Beams, cameras and the volume scanner of the line, in world coordinates."""
    st, fd = d['station'], d['feeder']
    floor = st['separator']['floor_z']
    T = SKIRT['thickness']
    # ---- beams: the line the controller tests, spanning between the housings' lenses ------------
    beams = [dict(name='beam_feed', role='feed belt: a lump tipped over the head edge stops the feed belt',
                  x=fd['beam']['x'], z=fd['beam']['z'], y0=cfg['lane_y'] - T, y1=cfg['belt_w'] + T, post=False,
                  max_block_s=MAX_BLOCK_S['beam_feed'])]
    half = st['width_m'] / 2 + T
    for key, role in (('beam_in', 'buffer belt entry: confirms each hand-over (interruptions and their length, checked '
                                  'against the camera objects)'),
                      ('beam_stop', 'buffer belt staging stop: the lead lump waits here while the measuring '
                                    'belt is busy'),
                      ('beam_gangue', 'gangue gap under the separator inlet: the gangue has dropped past it; the '
                                      'plate may move once the plate camera also sees its zone empty')):
        b = st[key]
        gangue = key == 'beam_gangue'
        beams.append(dict(name=key, role=role, x=b['x'], z=b['z'], y0=st['y_c'] - half - .03 * gangue,
                          y1=st['y_c'] + half + .03 * gangue, post=gangue, max_block_s=MAX_BLOCK_S[key]))
    # ---- cameras -----------------------------------------------------------------------------------
    y_c, top, P0x, x_e = st['y_c'], st['top_z'], float(d['P0'][0]), st['buffer']['x0']
    m_end = st['measure']['x1']
    sp = st['separator']
    lo_x = fd['x1'] - .25
    fx = (fd['x0'] + fd['x1']) / 2
    # the station camera looks along the belts: mounted high enough to see the approach, the buffer belt and
    # the measuring belt at lump-top height -- the approach lies a step higher (2.1 m over the belts up to a
    # 2.2 m span; higher for a longer buffer)
    span = m_end - (x_e - .40)
    h_st = max(2.1, -top + LUMP_TOP + span / 2 / (math.tan(math.radians(CAMERA['fovy']) / 2) * CAMERA['aspect']) + .05)
    # the singulator camera sees from 0.25 m behind the feed belt's head edge, at the top of a lump lying a step up,
    # to the main belt's head edge. 2.45 m is what the line with the 0.65 m lane needs; a narrower lane moves the
    # bend and the head edge downstream (0.60 m: by 7 cm), and the camera goes up with the span
    h_sg = max(2.45, fd['step_m'] + LUMP_TOP + (x_e - lo_x) / 2 / (math.tan(math.radians(CAMERA['fovy']) / 2)
                                                                    * CAMERA['aspect']) + .002)
    cams = [
        _camera('cam_feed', 'feed belt: the queue (staging, feed belt empty) and the lumps coming to the head edge',
                (fx, cfg['belt_w'] / 2, 2.45), (fx, cfg['belt_w'] / 2, 0.), CAMERA,
                [('feed belt', (fd['x0'], cfg['lane_y'], fd['step_m']),
                  (fd['x1'], cfg['belt_w'], fd['step_m'] + LUMP_TOP))],
                dict(kind='gantry', x=fx, ys=(-.25, cfg['belt_w'] + .25))),
        _camera('cam_singulator', 'feed belt head (creep, release), plough funnel and lane (release timing, '
                                  'stall detection)',
                ((lo_x + x_e) / 2, cfg['belt_w'] / 2, h_sg), ((lo_x + x_e) / 2, cfg['belt_w'] / 2, 0.), CAMERA,
                [('feed belt head', (lo_x, cfg['lane_y'], fd['step_m']), (fd['x1'] + .25, cfg['belt_w'],
                                                                           fd['step_m'] + LUMP_TOP)),
                 ('funnel and lane', (P0x - .80, cfg['lane_y'], 0.), (x_e, cfg['belt_w'], LUMP_TOP))],
                # the far post 0.40 m off the belt: the retracted face's top roller swings out to 0.28 m off it
                # (0.25 m cleared the face of the 0.65 m lane only)
                dict(kind='gantry', x=(lo_x + x_e) / 2, ys=(-.25, cfg['belt_w'] + .40))),
        _camera('cam_station', 'approach to the main belt head edge (hold), buffer and measuring belts',
                ((x_e - .40 + m_end) / 2, y_c, top + h_st), ((x_e - .40 + m_end) / 2, y_c, top), CAMERA,
                [('approach to the head edge', (x_e - .40, cfg['lane_y'], 0.), (x_e, cfg['lane_y'] + cfg['lane_w'],
                                                                                 LUMP_TOP)),
                 ('buffer and measuring belts', (x_e, y_c - st['width_m'] / 2, top),
                  (m_end, y_c + st['width_m'] / 2, top + LUMP_TOP))],
                dict(kind='gantry', x=(x_e - .40 + m_end) / 2, ys=(y_c - .62, y_c + .62))),
        _camera('cam_separator', 'separator plate and the gangue gap: the plate zone must be seen empty before '
                                 'the plate moves',
                ((m_end + sp['outlet_closed'][0]) / 2, y_c, 1.65), ((m_end + sp['outlet_closed'][0]) / 2, y_c, -.5),
                CAMERA,
                [('plate zone', tuple(st['plate_zone'][0]), tuple(st['plate_zone'][1]))],
                dict(kind='arm', x=(m_end + sp['outlet_closed'][0]) / 2, y=y_c + .85)),
    ]
    # ---- volume scanner: three depth heads on a gantry over the measuring belt ----------------------
    m0 = st['measure']['x0']
    zone = ((m0 + .02, y_c - st['width_m'] / 2 + .02, top), (m_end - .02, y_c + st['width_m'] / 2 - .02, top + LUMP_TOP - .05))
    gx, gz = (m0 + m_end) / 2, top + 1.30
    heads = [_camera('scan_%s' % side, 'volume scanner depth head (%s)' % side, pos, target, DEPTH,
                     [('measuring zone', zone[0], zone[1])], dict(kind='head'))
             for side, pos, target in (('top', (gx, y_c, gz - .05), (gx, y_c, top)),
                                       ('in', (gx, y_c - .55, top + 1.20), (gx, y_c + .10, top)),
                                       ('out', (gx, y_c + .55, top + 1.20), (gx, y_c - .10, top)))]
    scanner = dict(heads=heads, x=gx, ys=(y_c - .62, y_c + .62), z=gz, zone=zone, min_views=2,
                   clear_height=gz - top - .05)
    return dict(beams=beams, cameras=cams, scanner=scanner, floor=floor)


def coverage(dev):
    """Per camera and region: corners outside the view; for the scanner, measuring-zone corners seen by
    fewer than min_views heads. Empty lists everywhere = every region is covered."""
    out = {}
    for cam in dev['cameras']:
        for label, lo, hi in cam['sees']:
            P = corners(lo, hi)
            out['%s: %s' % (cam['name'], label)] = P[~in_view(cam, P)].round(3).tolist()
    sc = dev['scanner']
    P = corners(*sc['zone'])
    views = sum(in_view(h, P).astype(int) for h in sc['heads'])
    out['scanner: measuring zone seen by >= %d heads' % sc['min_views']] = P[views < sc['min_views']].round(3).tolist()
    return out


def report(dev):
    """JSON-safe summary for result.json."""
    cam = lambda c: dict(name=c['name'], role=c['role'], pos=c['pos'].round(4).tolist(),
                         looks_along=(-c['z']).round(4).tolist(), fovy_deg=c['fovy'], aspect=round(c['aspect'], 4))
    return dict(beams=[{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in b.items()}
                       for b in dev['beams']],
                cameras=[cam(c) for c in dev['cameras']],
                scanner=dict(heads=[cam(h) for h in dev['scanner']['heads']],
                             clear_height_m=round(dev['scanner']['clear_height'], 3),
                             min_views=dev['scanner']['min_views']),
                note='placement and optics are assumptions; what the controllers read is modelled in singulator/sensing/')


# ---- MJCF ---------------------------------------------------------------------------------------------
def _box(name, c, h, rgba, visual=False):
    cls = 'contype="0" conaffinity="0"' if visual else 'class="equip"'
    return ('<geom name="%s" type="box" %s pos="%.4f %.4f %.4f" size="%.4f %.4f %.4f" rgba="%s"/>'
            % (name, cls, c[0], c[1], c[2], h[0], h[1], h[2], rgba))


def _post(name, x, y, z0, z1):
    return _box(name, (x, y, (z0 + z1) / 2), (POST, POST, (z1 - z0) / 2), '.35 .37 .40 1')


def _housing(name, cam, half):
    """A camera or depth-head body, aligned with its view axis, with the lens on its front face."""
    R = np.stack([cam['x'], cam['y'], cam['z']], 1)
    w, x, y, z = _quat(R)
    lens = cam['pos'] - cam['z'] * (half[2] + .005)
    return ['<geom name="%s" type="box" class="equip" pos="%.4f %.4f %.4f" quat="%.6f %.6f %.6f %.6f" '
            'size="%.4f %.4f %.4f" rgba=".12 .13 .15 1"/>' % ((name,) + tuple(cam['pos']) + (w, x, y, z) + tuple(half)),
            '<geom name="%s_lens" type="cylinder" contype="0" conaffinity="0" pos="%.4f %.4f %.4f" '
            'quat="%.6f %.6f %.6f %.6f" size="%.4f .006" rgba=".15 .45 .85 1"/>'
            % ((name,) + tuple(lens) + (w, x, y, z) + (min(half[:2]) * .6,)),
            '<camera name="%s" pos="%.4f %.4f %.4f" xyaxes="%s" fovy="%.4g"/>'
            % ((name + '_view',) + tuple(cam['pos']) + (' '.join('%.6f' % v for v in np.r_[cam['x'], cam['y']]), cam['fovy']))]


def _quat(R):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    w = math.sqrt(max(0., 1. + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0., 1. + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0., 1. - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0., 1. - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return w, x, y, z


def xml(dev):
    """World-body geoms and cameras of every device, all named dev_*."""
    out, floor = [], dev['floor']
    hx, hy, hz = BEAM_HOUSING
    for b in dev['beams']:
        zc = b['z'] - BEAM_LENS_UP + hz
        for end, y in (('tx', b['y0'] - hy), ('rx', b['y1'] + hy)):
            out.append(_box('dev_%s_%s' % (b['name'], end), (b['x'], y, zc), BEAM_HOUSING, '.95 .75 .10 1'))
            if b['post']:
                out.append(_post('dev_%s_%s_post' % (b['name'], end), b['x'], y, floor, zc - hz))
        out.append('<geom name="dev_%s_light" type="cylinder" contype="0" conaffinity="0" fromto="%.4f %.4f %.4f '
                   '%.4f %.4f %.4f" size=".003" rgba="1 .1 .1 .7"/>'
                   % (b['name'], b['x'], b['y0'], b['z'], b['x'], b['y1'], b['z']))
    for cam in dev['cameras']:
        m, name = cam['mount'], 'dev_' + cam['name']
        top = cam['pos'][2] + CAMERA_HOUSING[2] + .10
        if m['kind'] == 'gantry':
            for i, y in enumerate(m['ys']):
                out.append(_post('%s_post%d' % (name, i), m['x'], y, floor, top + POST))
            out.append(_box(name + '_beam', (m['x'], sum(m['ys']) / 2, top), (POST, (m['ys'][1] - m['ys'][0]) / 2 + POST, POST),
                            '.35 .37 .40 1'))
        else:                                                     # a post beside the belt and an arm over it
            out.append(_post(name + '_post', m['x'], m['y'], floor, top + POST))
            out.append(_box(name + '_arm', (m['x'], (m['y'] + cam['pos'][1]) / 2, top),
                            (POST, abs(m['y'] - cam['pos'][1]) / 2 + POST, POST), '.35 .37 .40 1'))
        out.append(_box(name + '_stem', (cam['pos'][0], cam['pos'][1], (cam['pos'][2] + top) / 2),
                        (.012, .012, (top - cam['pos'][2]) / 2), '.35 .37 .40 1', visual=True))
        out += _housing(name, cam, CAMERA_HOUSING)
    sc = dev['scanner']
    for i, y in enumerate(sc['ys']):
        out.append(_post('dev_scanner_post%d' % i, sc['x'], y, floor, sc['z'] + POST))
    out.append(_box('dev_scanner_beam', (sc['x'], sum(sc['ys']) / 2, sc['z']), (POST, (sc['ys'][1] - sc['ys'][0]) / 2 + POST, POST),
                    '.85 .55 .15 1'))
    for h in sc['heads']:
        out += _housing('dev_' + h['name'], h, HEAD_HOUSING)
    return out
