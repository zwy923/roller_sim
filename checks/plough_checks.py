"""Assembly, drive and kinematic checks for the line's singulation part (feed belt, plough, lane). Stops at the
first failure.

1. Input validation: derive() refuses the combinations that used to build a silently nonsensical machine
   (a lane wider than the belt, a face turning the wrong way) and run() refuses solref < 2*dt.
2. Compiled geometry: the lane's clear width between the two walls really is lane_w, the side belt's inner face
   on the lane datum, main belt top at z = 0; with --face-kind plate the plate's material face lies on the face
   polyline (first box at curve_top, last at curve_exit), plate bottom gap / height.
3. Drive kinematics as compiled by MuJoCo: main belt, side belt, feed belt, buffer and measuring belts run at
   the speeds the report claims.
4. The face's budget: the stagger and the transit of the turning face against their closed forms, the speeds
   at the lane handoff, and the friction locking bound tan(beta) < 1/mu_face at the steep top.
5. Block mass = density x convex-hull volume, and 'screen size' really is the intermediate axis; the long-axis
   cap is an axis extent, not a caliper size.
6. Virtual motor: free run reaches rated speed, droop at the limit, stall on overload.
7. The line's face: hinge at the bend, roller face, scatter layout on the feed belt.
8. Lump outlines are placed with the geom frame of the compiled mesh (the observation bug of 2026-09-22).
9. Moving parts (swinging face, separator) clear all other equipment over their whole stroke.
10. Load bookkeeping: net and one-sided loads kept apart.
11. Face retract (the phase machine on an ideal servo): closing only with an empty sweep, pause if a lump
    enters it, reset fault if the face does not get home.
12. Feed belt: its geometry and step above the main belt, the drop beam, the release controller on the oracle
    view (approach, creep near the edge, release by centroid, stop on the beam, after-stop lumps, beam-miss
    and hanging-lump fallbacks, the 'lane' rule's next release only once the released lumps are through the
    funnel and the face is home), the drawn release line, and 12 s of a real batch.
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the project folder
from singulator import assembly, drives, face, feeder, lumps, machine, perception, simulate, tracking  # noqa: E402
from singulator.config import parse_config  # noqa: E402

LINE = parse_config(['--no-video', '--seed', '2'])


class IdealServo:
    """A face servo that follows its reference exactly (the interface of face.FaceServo, no MuJoCo): the phase
    machine in face.FaceRetract is checked on it. blocked(): it does not move this step (a jammed face)."""

    def __init__(self, cfg, d, dt, blocked=lambda phase: False):
        self.dt, self.blocked = dt, blocked
        self.parts, self.piv = machine.face_parts(cfg, d)
        self.lever = float(np.linalg.norm(d['P1'] - d['P0']))
        self.delta, self.swing_s = math.radians(cfg['face_swing_deg']), cfg['face_swing_s']
        self.drive = drives.make_motor(cfg['face_force_max'], 50., 1., cfg['motor_slip'])
        self.theta = self.actual_omega = 0.
        self.reached = False
        self.loads, self.hold, self.by_phase = [], [], dict(out=[], back=[])

    def command(self, phase, data):
        goal = self.delta if phase == 'out' else 0. if phase == 'back' else self.theta
        rate = abs(self.delta) / self.swing_s * self.dt
        step = 0. if self.blocked(phase) else float(np.clip(goal - self.theta, -rate, rate))
        self.theta, self.actual_omega = self.theta + step, step / self.dt

    def record(self, t, phase, data, ramp):
        pass

    sweep = face.FaceServo.sweep

    def report(self):
        return {}


def config(**over):
    cfg = dict(LINE)
    cfg.update(over)
    return cfg


def build(**over):
    """Config -> derived geometry -> compiled model, as run() does it."""
    cfg = config(**over)
    d = machine.derive(cfg)
    blocks = lumps.make_blocks(cfg, np.random.default_rng(cfg['seed']))
    m = mujoco.MjModel.from_xml_string(assembly.build_xml(cfg, d, blocks))
    return cfg, d, blocks, m, mujoco.MjData(m)


def check(ok, text):
    print(('PASS ' if ok else 'FAIL ') + text)
    if not ok:
        sys.exit(1)


def geom(model, data, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xpos[i], model.geom_size[i], data.geom_xmat[i].reshape(3, 3)


def body_geom(model, data, name):
    """First geom of a body (the belt pieces have no geom names of their own)."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    i = model.body_geomadr[b]
    return data.geom_xpos[i], model.geom_size[i], data.geom_xmat[i].reshape(3, 3)


def body_vel(model, data, name):
    res = np.zeros(6)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
    return res[3:]


def raises(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return True
    except Exception:
        return False
    return False


def oracle_observe(f, t, V, states, home):
    """The feed belt controller on the oracle view: one object per lump on the belt (true centroid and hull
    extents), the drop beam as a line test on the true hulls."""
    b = f.g['beam']
    view = {k: dict(cx=float(v[:, 0].min() + v[:, 0].max()) / 2, x0=float(v[:, 0].min()), x1=float(v[:, 0].max()),
                    y1=float(v[:, 1].max())) for k, v in enumerate(V) if states[k] == 'on_belt'}
    f.observe(t, view, any(states[k] == 'on_belt' and perception.cuts_line(v, b['x'], b['z'])
                           for k, v in enumerate(V)), home)


def main():
    print('--- 1. input validation (the geometry that used to silently invert) ---')
    bad = [('lane_w = 1.2 m (lane outside the belt)', dict(lane_w=1.2)),
           ('a face at 90 deg at the belt edge', dict(curve_top_deg=90.)),
           ('a face with its exit not below its top', dict(curve_top_deg=30., curve_exit_deg=55.))]
    for label, over in bad:
        check(raises(lambda o=over: machine.derive(config(**o))), 'rejects %s' % label)
    check(not raises(lambda: machine.derive(config())), 'accepts the line')
    check(raises(lambda: simulate.run(config(solref=.0005, dt=.0005)), ValueError),
          'run() rejects solref 0.5 ms with dt 0.5 ms (needs solref >= 2*dt)')

    print('--- 2. compiled geometry: lane %.2f m, plate face on the polyline ---' % LINE['lane_w'])
    cfg, d, blocks, m, data = build(face_kind='plate')
    mujoco.mj_forward(m, data)
    ci, si, _ = geom(m, data, 'skirt_in')
    co, so, _ = geom(m, data, 'lane_out')
    cs, ss, _ = body_geom(m, data, 'vslat0')
    low = cs[1] + ss[1] if cfg['side_belt'] else ci[1] + si[1]      # the lane's low wall: the side belt
    lane_clear = (co[1] - so[1]) - low
    check(abs(lane_clear - cfg['lane_w']) < 1e-9 and abs(low - cfg['lane_y']) < 1e-9
          and abs((ci[1] + si[1]) - cfg['lane_y']) < 1e-9,
          'lane clear width %.4f m = lane_w; side belt inner face and low skirt face at y %.4f = lane_y'
          % (lane_clear, low))
    # the plate face: one box per segment of the face polyline (a derive() product; the line's rollers sit on
    # this same polyline, checked in section 7). The XML is written with %.4f sizes/positions and %.6f radians,
    # so 0.1 mm is the resolution here
    tol = 2e-4
    segs = len(d['face_pts']) - 1
    c0, s0, R0 = geom(m, data, 'plough0')
    cn, sn, Rn = geom(m, data, 'plough%d' % (segs - 1))
    a0 = math.degrees(math.atan2(R0[1, 0], R0[0, 0]))
    an = math.degrees(math.atan2(Rn[1, 0], Rn[0, 0]))

    def on_face(c, R, p):
        return abs(abs(float(np.dot(p - c, R[:, 1]))) - machine.PLOUGH['thickness'] / 2) < tol

    lengths = sum(2 * geom(m, data, 'plough%d' % i)[1][0] for i in range(segs))
    check(segs > 5 and abs(a0 + cfg['curve_top_deg']) < 1e-3 and abs(an + cfg['curve_exit_deg']) < 1e-3
          and on_face(c0, R0, d['face_pts'][0]) and on_face(c0, R0, d['face_pts'][1])
          and on_face(cn, Rn, d['face_pts'][segs - 1]) and on_face(cn, Rn, d['face_pts'][segs])
          and np.allclose(d['face_pts'][0], d['P0']) and np.allclose(d['face_pts'][segs], d['P1'])
          and abs(d['P1'][1] - d['lane_top']) < 1e-12
          and abs(lengths - d['report']['diagonal_length_m']) < segs * tol
          and abs(c0[2] - s0[2] - machine.PLOUGH['bottom_gap']) < tol and abs(2 * s0[2] - machine.PLOUGH['height']) < tol,
          '%d face boxes from P0 to the bend P1 on the lane wall line: first %.1f deg at the belt edge, last %.1f deg'
          ' at the lane, %.4f m long in all, bottom gap %.4f m, height %.2f m, material face through the polyline'
          % (segs, -a0, -an, lengths, c0[2] - s0[2], 2 * s0[2]))
    cb, sb, _ = body_geom(m, data, 'mfloor')
    check(abs(cb[2] + sb[2]) < 1e-12 and abs(cb[0] + sb[0] - d['belt_x1']) < 1e-4,
          'main belt top z %.4f (want 0), head edge at x %.3f' % (cb[2] + sb[2], cb[0] + sb[0]))

    print('--- 3. drive kinematics as compiled ---')
    J = mujoco.mjtObj.mjOBJ_JOINT
    st = d['station']
    speeds = dict(mfloorj=cfg['v_belt'], feederj=d['feeder']['speed_m_s'], bbeltj=st['buffer_speed_m_s'],
                  mbeltj=st['speed_m_s'])
    for j, v in speeds.items():
        data.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, J, j)]] = v
    for i in range(d['n_sbslat']):
        data.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, J, 'vj%d' % i)]] = d['v_side']
    mujoco.mj_forward(m, data)
    got = {b: body_vel(m, data, b)[0] for b in ('mfloor', 'feeder', 'bbelt', 'mbelt', 'vslat1')}
    check(all(abs(got[b] - speeds[j]) < 1e-12 for b, j in (('mfloor', 'mfloorj'), ('feeder', 'feederj'),
                                                            ('bbelt', 'bbeltj'), ('mbelt', 'mbeltj')))
          and abs(got['vslat1'] - d['v_side']) < 1e-12,
          'speeds: main belt %.3f, side belt %.3f = %.1fx, feed belt %.3f, buffer belt %.3f, measuring belt %.3f m/s'
          % (got['mfloor'], got['vslat1'], cfg['side_belt_ratio'], got['feeder'], got['bbelt'], got['mbelt']))

    print('--- 4. the budget of the face and the friction locking bound ---')
    g = d['report']
    sweep = g['sweep_m']
    t, e = math.radians(cfg['curve_top_deg']), math.radians(cfg['curve_exit_deg'])
    check(abs(sweep - (cfg['belt_w'] - d['lane_top'])) < 1e-12, 'sweep %.3f m = belt width - lane top' % sweep)
    # the stagger budget of a linearly-turning face has a closed form: sweep * (ln cos top - ln cos exit)/(exit-top)
    analytic = sweep * (math.log(math.cos(t)) - math.log(math.cos(e))) / (e - t)
    got = g['predicted_stagger_of_outermost_lump_m']
    # the 24-box polyline carries ~0.5 % discretisation against the smooth integral, so 1 % is the bar
    check(abs(got - analytic) < 1e-2 * analytic,
          'stagger %.3f m = sum tan(beta) dy over the %d boxes (smooth-curve analytic %.3f, %.1f %% apart); a straight'
          ' %.0f deg face on the same sweep would give %.3f m'
          % (got, segs, analytic, 100 * abs(got - analytic) / analytic, cfg['curve_top_deg'], sweep * math.tan(t)))
    crawl_top = cfg['v_belt'] * math.cos(t) ** 2      # the steep top is the slowest point of the face
    check(abs(g['face_crawl_speed_m_s'] - cfg['v_belt'] * math.cos(e) ** 2) < 1e-12
          and abs(g['deabreast_rate_m_s'] - cfg['v_belt'] * math.sin(e) ** 2) < 1e-12
          and abs(g['face_lateral_speed_m_s'] - cfg['v_belt'] * math.sin(e) * math.cos(e)) < 1e-12
          and abs(g['along_face_speed_m_s'] - cfg['v_belt'] * math.cos(e)) < 1e-12
          and g['face_crawl_speed_m_s'] > crawl_top,
          'at the lane handoff (%g deg): crawl %.3f m/s = v*cos^2, de-abreast rate %.3f m/s = v*sin^2, lateral %.3f m/s;'
          ' the crawl at the %.0f deg top is %.3f m/s (the handoff is %.1fx faster)'
          % (cfg['curve_exit_deg'], g['face_crawl_speed_m_s'], g['deabreast_rate_m_s'], g['face_lateral_speed_m_s'],
             cfg['curve_top_deg'], crawl_top, g['face_crawl_speed_m_s'] / crawl_top))
    # transit likewise: integral dy / (v sin b cos b) = sweep/v * ln(tan exit / tan top) / (exit - top)
    transit_analytic = sweep / cfg['v_belt'] * math.log(math.tan(e) / math.tan(t)) / (e - t)
    check(abs(g['face_transit_s'] - transit_analytic) < 1e-2 * transit_analytic,
          'face transit %.2f s for a lump crossing the whole sweep (analytic %.2f s)'
          % (g['face_transit_s'], transit_analytic))
    f = g['face_locking']
    check(abs(f['ratio_mu_tan_beta'] - cfg['friction_steel'] * math.tan(t)) < 1e-12
          and f['slides_along_face'] and abs(f['lock_ceiling_deg'] - math.degrees(math.atan(1 / cfg['friction_steel']))) < 1e-9,
          'locking at the top: mu*tan(beta) = %.2f < 1, ceiling %.1f deg for mu %.2f (margin %.0f %%)'
          % (f['ratio_mu_tan_beta'], f['lock_ceiling_deg'], f['mu_face'], 100 * f['margin']))
    steep = machine.derive(config(curve_top_deg=70.))['report']['face_locking']
    lined = machine.derive(config(curve_top_deg=65., friction_steel=.20))['report']['face_locking']
    check(not steep['slides_along_face'] and lined['slides_along_face'],
          'the bound bites: a 70 deg top on steel reports LOCKS, 65 deg on a UHMW-PE face (0.20) still slides')

    print('--- 5. blocks and material ---')
    rng = np.random.default_rng(3)
    for fam in lumps.FAMILIES:
        b = dict(lumps.make_block(rng, fam, .40), material='gangue', density=2500.)
        one = lumps.make_blocks(config(count=1), np.random.default_rng(0))
        cfg1, d1 = config(count=1), machine.derive(config(count=1))
        one[0].update(b)
        mm = mujoco.MjModel.from_xml_string(assembly.build_xml(cfg1, d1, one))
        mass = mm.body_mass[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, 'b0')]
        check(abs(mass - ConvexHull(b['vertices']).volume * b['density']) < 1e-6 * mass,
              '%-9s lump mass %.1f kg = density x hull volume' % (fam, mass))
    # screen size class (user, 2026-09-20): `size` is the aperture, so it is the INTERMEDIATE axis and the
    # long axis sticks out past it
    b = lumps.make_block(np.random.default_rng(11), 'bladed', .30, None)
    check(abs(b['axes'][1] - .30) < 1e-12 and b['axes'][0] > .30 and b['axes'][2] < .30,
          'screen size 0.30 m -> axes %s (mid axis = aperture, long axis sticks out)' % [round(a, 3) for a in b['axes']])
    raw = max(lumps.make_block(np.random.default_rng(s), 'bladed', .50, None)['axes'][0] for s in range(200))
    check(raw > 1.0, 'uncapped 0.50 m screen class reaches %.2f m long over 200 bladed draws (size/b, b >= 0.45)' % raw)
    LMAX = cfg['size_long_max']
    cap = max(lumps.make_block(np.random.default_rng(s), fam, .50, LMAX)['axes'][0]
              for s in range(200) for fam in lumps.FAMILIES)
    check(cap <= LMAX + 1e-12, '--size-long-max %.2f m holds over 200 draws x 5 families (longest %.3f m)' % (LMAX, cap))
    free = lumps.make_block(np.random.default_rng(11), 'bladed', .50, None)     # long 1.05 m, the cap bites
    capped = lumps.make_block(np.random.default_rng(11), 'bladed', .50, LMAX)
    check(capped['axes'][1] == free['axes'][1] == .50 and abs(capped['axes'][0] - LMAX) < 1e-12
          and capped['axes'][2] < free['axes'][2],
          'the cap trims the long axis only: mid stays at the aperture 0.500 m, long %.3f -> %.3f m (short %.3f -> %.3f)'
          % (free['axes'][0], capped['axes'][0], free['axes'][2], capped['axes'][2]))
    # the cap bounds the extent along body x, not the largest caliper size: a lump lying flat can still
    # span more than the lane in plan. Pinned here so nobody reads the cap as "fits the lane any way round"
    feret = [pdist(lumps.make_block(np.random.default_rng(s), fam, .50, LMAX)['vertices'][:, :2]).max()
             for s in range(200) for fam in lumps.FAMILIES]
    lane = cfg['lane_w']
    check(max(feret) > .58,
          'the %.2f m cap is an axis extent, not a Feret size: flat-lying plan span reaches %.3f m, %d of %d draws'
          ' exceed the %.2f m lane' % (LMAX, max(feret), sum(f > lane for f in feret), len(feret), lane))
    dens = np.array([lumps.make_blocks(config(count=1), np.random.default_rng(s))[0]['density'] for s in range(200)])
    check(dens.min() >= lumps.DENSITY_RANGE['coal'][0] - 1e-9 and dens.max() <= lumps.DENSITY_RANGE['gangue'][1] + 1e-9
          and (dens > lumps.DENSITY_RANGE['coal'][1]).any(),
          'sampled densities span coal/gangue ranges: %.0f..%.0f kg/m3' % (dens.min(), dens.max()))

    print('--- 6. virtual motor ---')
    dtm = .0005
    mo = drives.make_motor(1000., 300., .40, .05)
    for _ in range(4000):
        drives.motor_update(mo, 1., 0., dtm)
    free = mo['f']
    for _ in range(4000):
        drives.motor_update(mo, 1., -500., dtm)
    loaded = mo['f']
    for _ in range(4000):
        drives.motor_update(mo, 1., -2500., dtm)
    stalled = mo['f']
    big = drives.make_motor(15000., 300., .40, .05)
    for _ in range(4000):
        drives.motor_update(big, 1., -12000., dtm)
    stiff = big['f']
    check(abs(free - 1) < 1e-6 and abs(loaded - .975) < 2e-3 and stalled == 0. and stiff > .9,
          'motor: free %.3f, at half limit %.3f (expect 0.975), overload %.3f, 15 kN drive under 12 kN %.3f'
          % (free, loaded, stalled, stiff))

    print('--- 7. the line: hinged roller face, scatter layout ---')
    cfgv, dv, _, mv, datav = build()
    mujoco.mj_forward(mv, datav)
    Bv = mujoco.mjtObj.mjOBJ_BODY
    fj = mujoco.mj_name2id(mv, J, 'facej')
    fb = mujoco.mj_name2id(mv, Bv, 'face')
    piv = dv['P1']
    check(fj >= 0 and fb >= 0 and mv.jnt_type[fj] == int(mujoco.mjtJoint.mjJNT_HINGE)
          and np.allclose(mv.jnt_axis[fj], [0., 0., 1.])
          and np.allclose(datav.xpos[fb][:2], piv[:2], atol=2e-4)    # pos goes through the XML as %.4f
          and mv.actuator_trnid[mv.actuator('face_drive').id][0] == fj and mv.dof_armature[mv.jnt_dofadr[fj]] == 0.,
          'swing face: hinge about z at the bend (%.3f, %.3f), a position servo on it, swing %+.0f deg'
          % (datav.xpos[fb][0], datav.xpos[fb][1], cfgv['face_swing_deg']))
    # --face-kind rollers: the diagonal becomes free-spinning vertical rollers whose SURFACES are
    # tangent to the same material face the plate presented, so the lane geometry is unchanged.
    B = mujoco.mjtObj.mjOBJ_BODY
    frn = [mujoco.mj_id2name(mv, B, b) for b in range(mv.nbody)
           if (mujoco.mj_id2name(mv, B, b) or '').startswith('fr')
           and (mujoco.mj_id2name(mv, B, b) or 'x')[2:].isdigit()]
    pts = np.array(dv['face_pts'])[:, :2]
    Rr = machine.FACE_ROLLER['d'] / 2
    dev, ctr = [], []
    for n in frn:
        c = datav.xpos[mujoco.mj_name2id(mv, B, n)][:2]
        ctr.append(c)
        best = 9.
        for i in range(len(pts) - 1):
            e_ = pts[i + 1] - pts[i]
            L_ = np.linalg.norm(e_)
            u = e_ / L_
            tt = float(np.clip(np.dot(c - pts[i], u), 0., L_))
            best = min(best, float(np.linalg.norm(c - (pts[i] + tt * u))))
        dev.append(best - Rr)
    gaps = [float(np.linalg.norm(ctr[i + 1] - ctr[i])) - 2 * Rr for i in range(len(ctr) - 1)]
    gi = mv.body_geomadr[mujoco.mj_name2id(mv, B, frn[0])]
    ji = mv.body_jntadr[mujoco.mj_name2id(mv, B, frn[0])]
    check(len(frn) >= 8 and max(abs(x) for x in dev) < 2e-4,
          'face rollers: %d compiled, surfaces tangent to the face line within %.2f mm'
          % (len(frn), 1000 * max(abs(x) for x in dev)))
    check(max(gaps) < cfgv['size_min'] * .45 and min(gaps) > 0,
          'roller gaps %.0f-%.0f mm stay below the smallest lump dimension (~130 mm): nothing wedges in'
          % (1000 * min(gaps), 1000 * max(gaps)))
    # the bottom roller is the fillet of the bend: its low tangent lies on the lane wall line, and the
    # wall (which has to start past the roller's swing) begins downstream of that tangent point
    lo = ctr[-1]
    check(abs(lo[1] - Rr - dv['lane_top']) < 2e-4 and dv['P1'][0] < lo[0] < dv['lane_out_start'],
          'bottom roller low tangent at y %.4f = lane wall line %.4f (flush), tangent point x %.3f, %.0f mm past the'
          ' bend; the wall starts at x %.3f' % (lo[1] - Rr, dv['lane_top'], lo[0], 1000 * (lo[0] - dv['P1'][0]),
                                               dv['lane_out_start']))
    check(mv.jnt_type[ji] == mujoco.mjtJoint.mjJNT_HINGE and abs(mv.jnt_axis[ji][2] - 1) < 1e-12
          and not any(mv.actuator_trnid[a][0] == ji for a in range(mv.nu))
          and abs(2 * mv.geom_size[gi][1] - machine.FACE_ROLLER['height']) < 1e-9,
          'each roller is a free hinge about z (no actuator), D %.0f mm x %.2f m tall'
          % (1000 * 2 * mv.geom_size[gi][0], 2 * mv.geom_size[gi][1]))
    # scatter layout: one de-stacked layer, arbitrary distribution, laid on the feed belt. The whole point is that
    # it may be dense and untidy, so the check is that it is still SINGLE-LAYER (no plan-view overlap), resting on
    # the feed belt, and wholly on it -- not that lumps keep any spacing.
    fdl = dv['feeder']
    worst_gap, fits, depths = 9., True, []
    for sd in range(25):
        rs = np.random.default_rng(sd)
        bl = lumps.make_blocks(cfgv, rs)
        layer, rear, front = lumps.plan_scatter(cfgv, bl, rs, dv)
        hulls = [lumps.hull2d(bl[k]['vertices'], yaw) + (x, y) for k, x, y, z, yaw in layer]
        for i in range(len(hulls)):
            for j in range(i + 1, len(hulls)):
                if lumps.overlap2d(hulls[i], hulls[j], margin=0.):
                    worst_gap = -1.
        # every lump must REST on the feed belt: its lowest vertex, not its centroid (which rises with size)
        lows = [z + lumps.footprint(bl[k]['vertices'], yaw)[4] for k, x, y, z, yaw in layer]
        if max(lows) - min(lows) > 1e-9 or abs(min(lows) - fdl['step_m'] - cfgv['feed_drop_height']) > 1e-9:
            worst_gap = -2.
        fits &= abs(front - (fdl['x1'] - feeder.FRONT_MARGIN)) < 1e-9 and rear >= fdl['x0'] + feeder.REAR_MARGIN - 1e-9
        depths.append(front - rear)
    check(worst_gap > 0 and fits,
          'scatter layout over 25 batches: no two plan-view hulls overlap (single layer), every lump starts on the '
          'feed belt surface, the front %.2f m behind the head edge, the rear on the belt' % feeder.FRONT_MARGIN)
    print('    scatter patch depth over 25 batches of 5: %.2f..%.2f m (feed belt %.2f m)'
          % (min(depths), max(depths), fdl['length_m']))

    print('--- 8. lump outlines are placed with the geom frame (compiled mesh) ---')
    # MuJoCo re-centres a mesh on its centroid and turns it onto its principal axes, compensating in the
    # geom's pos/quat. The observation used the body frame until 2026-09-22 (0.18 m off on seed 81).
    worst_ok, worst_old = 0., 0.
    for sd in (1, 81, 260, 530):
        rng9 = np.random.default_rng(sd)
        bl9 = lumps.make_blocks(cfgv, rng9)
        layer, _, _ = lumps.plan_scatter(cfgv, bl9, rng9, dv)
        m9 = mujoco.MjModel.from_xml_string(assembly.build_xml(cfgv, dv, bl9))
        dat9 = mujoco.MjData(m9)
        for k, x, y, z, yaw in layer:
            q = m9.jnt_qposadr[m9.joint('bj%d' % k).id]
            dat9.qpos[q:q + 3] = (x, y, z)
            dat9.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))
        mujoco.mj_forward(m9, dat9)
        for k, blk9 in enumerate(bl9):
            g9, b9 = m9.geom('bg%d' % k).id, m9.body('b%d' % k).id
            local = m9.mesh_vert[m9.mesh_vertadr[m9.geom_dataid[g9]]:][:m9.mesh_vertnum[m9.geom_dataid[g9]]]
            V = lumps.block_world_vertices(dat9, dict(geom=g9, local=local))
            ref = dat9.xpos[b9] + blk9['vertices'] @ dat9.xmat[b9].reshape(3, 3).T
            old = dat9.xpos[b9] + local @ dat9.xmat[b9].reshape(3, 3).T
            box = lambda P: np.array([P.min(0), P.max(0)])
            worst_ok = max(worst_ok, float(abs(box(V) - box(ref)).max()))
            worst_old = max(worst_old, float(abs(box(old) - box(ref)).max()))
    check(worst_ok < 1e-6 and worst_old > .05,
          'geom-frame outline matches the generated lump within %.1e m (the old body-frame placement was off by'
          ' up to %.3f m on these seeds)' % (worst_ok, worst_old))

    print('--- 9. moving parts clear every other piece of equipment over their whole stroke ---')
    # equipment never collides with equipment in the simulation, so this is the only place an
    # interference would show. The camera masts are placed for the line's roller face: a swinging steel plate
    # (--face-kind plate) would hit the single-file camera's mast (-32 mm, 2026-09-30)
    for label, over in (('the line', {}),):
        cc = config(**over)
        dd = machine.derive(cc)
        res = assembly.motion_clearance(cc, dd)
        worst = min(r['min_distance_m'] for r in res.values())
        check(worst > 1e-3 and not any(r['unresolved'] for r in res.values()),
              '%-18s smallest clearance %.1f mm (%s) | outer skirt ends x %.3f, lane wall starts x %.3f'
              % (label, 1000 * worst, '; '.join('%s %s' % (k, r['pair']) for k, r in res.items()),
                 dd['skirt_out_end'], dd['lane_out_start']))
    old = assembly.motion_clearance(cfgv, dict(dv, skirt_out_end=dv['belt_x1'], lane_out_start=float(dv['P1'][0])))['face']
    check(old['min_distance_m'] < -.01, 'the pre-fix walls (skirt to the belt end) would be %.0f mm inside the swinging'
                                        ' face (%s)' % (-1000 * old['min_distance_m'], old['pair'][1]))

    print('--- 10. load bookkeeping ---')
    dtl = .0005
    series = np.zeros(4000)
    series[1000:1400] = -2500.                 # 0.2 s of resistance, the rest idle
    series[2000:2002] = -40000.                # a 1 ms contact spike
    stl = drives.load_stats(series, dtl, 3000.)
    check(stl['resist_avg50ms_max_N'] == stl['net_avg50ms_resist_max_N'] == 2500. and stl['resist_peak_step_N'] == 40000.
          and abs(stl['resist_over_10pct_limit_s'] - .201) < 1e-9 and stl['assist_avg50ms_max_N'] == 0.
          and np.percentile(series, 99) == 0.,
          'one-sided resistance: 50 ms average %.0f N, spike %.0f N, loaded %.3f s -- the old signed p99 reads %.0f N'
          % (stl['resist_avg50ms_max_N'], stl['resist_peak_step_N'], stl['resist_over_10pct_limit_s'],
             np.percentile(series, 99)))
    alt = drives.load_stats(np.tile([2000., -2000.], 2000), dtl, 3000.)
    check(alt['net_avg50ms_resist_max_N'] == alt['net_avg50ms_assist_max_N'] == 0.
          and alt['resist_avg50ms_max_N'] == alt['assist_avg50ms_max_N'] == 1000.
          and alt['resist_over_10pct_limit_s'] == alt['assist_over_10pct_limit_s'] == 1.,
          'alternating +/-2000 N: net 50 ms average %.0f N (cancels), resistance and assistance %.0f / %.0f N'
          ' each for %.1f s (rectified before averaging)'
          % (alt['net_avg50ms_resist_max_N'], alt['resist_avg50ms_max_N'], alt['assist_avg50ms_max_N'],
             alt['resist_over_10pct_limit_s']))

    print('--- 11. face retract: closing permission, pause, reset fault ---')
    cf, df = cfgv, dv
    dtf = cf['dt']
    sq = lambda x0, x1, y0, y1: np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    check(face.polygons_overlap(sq(0, 1, 0, 1)[None], sq(1.005, 2, 0, 1), .01)[0]
          and not face.polygons_overlap(sq(0, 1, 0, 1)[None], sq(1.02, 2, 0, 1), .01)[0],
          'polygon test: 5 mm apart overlaps within the 10 mm clearance, 20 mm apart does not')
    # the reviewer's case: rear at x 0.30, front already at x 0.75 near the belt edge, inside the plough area
    intruder = {0: sq(.30, .75, .95, 1.18)}
    lane_lump = {1: sq(1.80, 2.20, .10, .50)}

    def cycle(t1, plans=lambda t: {}, busy=lambda t: False, blocked=lambda phase: False, stop_on_timeout=False):
        fr_ = face.FaceRetract(cf, IdealServo(cf, df, dtf, blocked))
        fr_.start(0.)
        tt, n, log = 0., 0, []
        while tt < t1 and fr_.phase not in ('idle', 'fault'):
            fr_.command(tt, None)
            fr_.record(tt, None)
            if n % 20 == 0:
                fr_.zone_busy = busy(tt)
                fr_.check_sweep(plans(tt))
                log.append((round(tt, 2), fr_.phase))
                if stop_on_timeout and fr_.hold_timed_out(tt):
                    return fr_, tt, log
            tt, n = tt + dtf, n + 1
        return fr_, tt, log
    swing = cf['face_swing_s']
    ok, t_ok, _ = cycle(10., plans=lambda t: lane_lump)
    check(ok.phase == 'idle' and abs(ok.theta) < 1e-9 and ok.fault is None and ok.closed_on_empty and ok.at_home
          and ok.face.reached and 2.2 < t_ok < 2.5,
          'clear section, lump only in the lane: out, hold, close -> home at %.2f s' % t_ok)
    held, t_held, log = cycle(40., plans=lambda t: intruder, stop_on_timeout=True)
    check(held.phase == 'hold' and held.in_sweep == [0] and abs(t_held - (swing + cf['face_hold_max_s'])) < .05
          and 'back' not in {p for _, p in log},
          'lump reaching into the closing sweep: face never starts closing, hold timeout at %.2f s (face open)'
          % t_held)
    paused, t_p, log = cycle(12., plans=lambda t: intruder if swing + .6 < t < swing + 2. else {})
    phases = [p for _, p in log]
    check(paused.phase == 'idle' and abs(paused.theta) < 1e-9 and paused.events[-1]['pauses'] == 1
          and phases.index('back') < len(phases) - 1 - phases[::-1].index('hold'),
          'lump enters the sweep while closing: face pauses (back -> hold), resumes when clear, home at %.2f s' % t_p)
    # the face cannot move while closing (a load far above the servo's limit): it does not get home
    jammed_face, t_f, _ = cycle(30., blocked=lambda phase: phase == 'back')
    check(jammed_face.phase == 'fault' and jammed_face.fault['reason'] == 'reset_not_reached'
          and jammed_face.theta != 0. and abs(t_f - (1.3 + 6 * swing)) < .05 and not jammed_face.at_home,
          'face blocked while closing: reset fault at %.2f s, %.1f deg from home -- not idle'
          % (t_f, jammed_face.fault['angle_deg']))
    stuck_out, t_s, _ = cycle(30., blocked=lambda phase: phase == 'out')
    check(stuck_out.phase == 'fault' and stuck_out.fault['reason'] == 'retract_not_reached' and abs(t_s - 6 * swing) < .05,
          'face blocked while retracting: fault at %.2f s, no closing attempted' % t_s)

    print('--- 12. feed belt: step-down transfer released lump by lump ---')
    cq = config(feeder_release='lane', sensing='oracle')
    dq = machine.derive(cq)
    fq = dq['feeder']
    check(raises(lambda: machine.derive(config(feeder_speed=.40)))
          and raises(lambda: machine.derive(config(feeder_creep_speed=.20)))
          and raises(lambda: machine.derive(config(feeder_step=0.)))
          and raises(lambda: machine.derive(config(feeder_gap=.70))),
          'rejects a feed belt as fast as the main belt, a creep faster than the approach, no step, and a gap shorter '
          'than one lump + the stopping distance')
    check(abs(fq['x1'] - (cq['plough_x'] - cq['feeder_gap'])) < 1e-12
          and abs(fq['x1'] - fq['x0'] - cq['feeder_len']) < 1e-12 and dq['belt_x0'] == fq['x0']
          and fq['longest_plan_extent_m'] + fq['stop_distance_m'] <= cq['feeder_gap']
          and abs(fq['beam']['x'] - fq['x1'] - feeder.BEAM_X) < 1e-12 and abs(fq['beam']['z'] - fq['step_m'] / 2) < 1e-12,
          'feed belt x %.3f..%.3f (%.2f m), %.2f m of main belt to the plough start >= %.3f m plan extent + %.3f m'
          ' stopping distance; drop beam %.2f m past the edge at z %.3f m; skirts start at the feed belt'
          % (fq['x0'], fq['x1'], fq['length_m'], cq['feeder_gap'], fq['longest_plan_extent_m'], fq['stop_distance_m'],
             feeder.BEAM_X, fq['beam']['z']))
    mq = mujoco.MjModel.from_xml_string(assembly.build_xml(cq, dq, []))
    datq = mujoco.MjData(mq)
    mujoco.mj_forward(mq, datq)
    cfd, sfd, _ = body_geom(mq, datq, 'feeder')
    cmf, smf, _ = body_geom(mq, datq, 'mfloor')
    check(abs((cfd[2] + sfd[2]) - (cmf[2] + smf[2]) - fq['step_m']) < 1e-9 and abs(cfd[0] + sfd[0] - fq['x1']) < 1e-4
          and abs(cmf[0] - smf[0] - fq['x1']) < 1e-4 and abs(cmf[0] + smf[0] - dq['belt_x1']) < 1e-4
          and abs(sfd[1] - smf[1]) < 1e-9
          and tracking.CATS[tracking.categorise(mq)[mq.body_geomadr[mq.body('feeder').id]]] == 'belt',
          'feed belt top %.3f m above the main belt top, head edge at x %.3f where the main belt starts, same width;'
          ' the main belt runs on straight to its head edge x %.3f; the feed belt counts as belt contact'
          % (fq['step_m'], fq['x1'], dq['belt_x1']))

    # the drop beam on synthetic hulls: boxes as 8 vertices, X = the head edge, h = the step
    X, h = fq['x1'], fq['step_m']
    bx = lambda x0, x1, z0, z1: np.array([[x, y, z] for x in (x0, x1) for y in (.2, .6) for z in (z0, z1)])
    beam = lambda V: perception.cuts_line(V, fq['beam']['x'], fq['beam']['z'])
    # tipped nose-down about the edge (X, h) until its underside at the beam is a quarter step above the main belt
    th, rel = math.atan(.75 * h / feeder.BEAM_X), bx(X - .2, X + .2, h, h + .3) - (X, 0., h)
    tilt = np.stack([rel[:, 0] * np.cos(th) + rel[:, 2] * np.sin(th), rel[:, 1],
                     -rel[:, 0] * np.sin(th) + rel[:, 2] * np.cos(th)], 1) + (X, 0., h)
    check(not beam(bx(X - .25, X + .20, h, h + .30)) and beam(bx(X - .05, X + .40, 0., .30))
          and not beam(bx(X + .15, X + .50, 0., .30)) and beam(tilt),
          'drop beam: a lump overhanging the edge on the feed belt passes above it; one lying on the main belt across'
          ' it, or tipped nose-down over the edge, cuts it; one already past it does not')

    # the controller on synthetic views: every lump a box, centroid = box centre
    def view(xs, zs=None):
        zs = zs or [(h, h + .3)] * len(xs)
        return [bx(a, b_, z0, z1) for (a, b_), (z0, z1) in zip(xs, zs)]
    fd_ = feeder.Feeder(cq, dq, 3)
    stt = ['on_belt'] * 3
    fd_.start(0.)
    xs = [(X - .45, X - .05), (X - .60, X + .02), (X - 1.2, X - .8)]     # 0 leads by centroid; 1 leads by front, overhanging
    oracle_observe(fd_, 1., view(xs), stt, True)
    approach = fd_.phase == 'feeding' and fd_.goal == 1. and not fd_.released
    for k in range(40):
        fd_.drive_target(.01)
    up = fd_.target == 1.
    xs = [(a + .21, b_ + .21) for a, b_ in xs]                            # 0's centroid 4 cm short of the edge
    oracle_observe(fd_, 3.1, view(xs), stt, True)
    creep = fd_.goal == fd_.creep and fd_.releases[0]['t_creep_s'] == 3.1
    for k in range(40):
        fd_.drive_target(.01)
    creeping = abs(fd_.target - fd_.creep) < 1e-12
    xs = [(a + .05, b_ + .05) for a, b_ in xs]                            # 0 past the edge; 1's front well over, centroid not
    V = view(xs)
    oracle_observe(fd_, 4.8, V, stt, True)
    released = fd_.released == [0] and fd_.phase == 'feeding' and xs[1][1] > X + .05
    V[0] = bx(xs[0][0], xs[0][1], 0., .3)                                 # 0 has come down across the beam
    oracle_observe(fd_, 5.2, V, stt, True)
    stopped = fd_.phase == 'stopped' and fd_.goal == 0. and fd_.releases[0]['stop'] == 'beam'
    for k in range(40):
        fd_.drive_target(.01)
    check(approach and up and creep and creeping and released and stopped and fd_.target == 0.,
          'approach at full speed, creep (%.2f of speed) once the leading centroid is within %.2f m of the edge; a lump'
          ' is released when its centroid passes the edge (the one overhanging beside it is not); stop on the beam'
          % (fd_.creep, feeder.CREEP_ZONE))
    xs[0], xs[1] = (X + .15, X + .55), (X - .15, X + .30)                 # 0 taken away; 1 goes late, during the stop
    oracle_observe(fd_, 5.4, view(xs), stt, True)
    late = fd_.releases[0]['after_stop'] == [1] and fd_.counts['after_stop'] == 1 and fd_.released == [0, 1]
    Lin = dq['lane_out_start']                                            # the lane entry: the outer wall starts here
    oracle_observe(fd_, 12., view([(Lin - .02, Lin + .38), (Lin + .05, Lin + .45), xs[2]]), stt, True)
    wait_lane = fd_.phase == 'stopped'                                    # 0's rear still in the funnel mouth
    V = view([(Lin + .01, Lin + .41), (Lin + .05, Lin + .45), xs[2]])
    oracle_observe(fd_, 13., V, stt, False)                               # both in the lane, face not home yet
    wait_face = fd_.phase == 'stopped'
    oracle_observe(fd_, 14., V, stt, True)
    check(late and wait_lane and wait_face and fd_.phase == 'feeding' and len(fd_.releases) == 2
          and fd_.releases[1]['t_start_s'] == 14. and Lin + .45 < dq['exit_x'] and Lin >= dq['P1'][0],
          "a lump that goes after the stop counts with that release (after_stop); the 'lane' rule's next release waits"
          ' until every released lump is through the funnel (rear past the lane entry x %.3f, %.2f m before the lane'
          ' exit) AND the face is home, then starts at once' % (Lin, dq['exit_x'] - Lin))
    rl = mq.geom('release_line').id
    mline = mujoco.MjModel.from_xml_string(assembly.build_xml(cfgv, dv, []))
    check(abs(mq.geom_pos[rl][0] - Lin) < 1e-4 and mq.geom_contype[rl] == 0 and mq.geom_conaffinity[rl] == 0
          and mujoco.mj_name2id(mline, mujoco.mjtObj.mjOBJ_GEOM, 'release_line') == -1,
          "the release line is drawn at the lane entry x %.3f with the 'lane' rule only, and nothing touches it" % Lin)
    # fallbacks: a lump that went without cutting the beam; a released lump hanging on the edge gets a creep jog
    fm = feeder.Feeder(cq, dq, 2)
    fm.start(0.)
    xs = [(X - .15, X + .25), (X - 1.2, X - .8)]
    for i in range(130):
        oracle_observe(fm, i * .01, view(xs), ['on_belt'] * 2, True)
    missed = fm.releases[0]['stop'] == 'beam_missed' and fm.counts['beam_missed'] == 1
    jogging = fm.jog == 0 and fm.phase == 'feeding' and fm.goal == fm.creep and fm.counts['jogs'] == 1
    oracle_observe(fm, 1.31, view([(X + .03, X + .43), xs[1]]), ['on_belt'] * 2, True)
    check(missed and jogging and fm.phase == 'stopped' and fm.jog is None and fm.releases[0]['stop'] == 'beam_missed',
          'a lump past the edge that never cuts the beam stops the release after %.1f s; left hanging across the edge'
          ' with no progress it gets a creep jog until its rear is %.2f m clear' % (feeder.BEAM_MISS_S, feeder.CLEAR))
    fe = feeder.Feeder(cq, dq, 1)
    fe.start(0.)
    oracle_observe(fe, 0.5, view([(X + .9, X + 1.3)]), ['passed'], True)
    check(fe.phase == 'stopped' and fe.releases[0]['stop'] == 'feed_belt_empty',
          'a release with nothing left on the feed belt stops at once')

    # physics: the first 12 s of a real batch on the line. One lump tips over, the beam stops the belt; the next
    # release starts before that lump reaches the lane exit
    with tempfile.TemporaryDirectory() as tmp:
        cr = config(seed=392, duration=12., out_dir=Path(tmp) / 'r')
        r = simulate.run(cr)
        with np.load(Path(tmp) / 'r' / 'trajectory.npz') as zf:   # closed before the folder goes (Windows)
            z = {k: zf[k] for k in zf.files}
    fr = r['feeder']
    rel0 = fr['releases'][0]
    stop_t = rel0['t_stop_s']
    t_next = fr['releases'][1]['t_start_s'] if len(fr['releases']) > 1 else None
    went = rel0.get('lumps') or rel0.get('members', [])    # the cameras' view: verification record; else its own
    lead = r['blocks'][went[0]] if len(went) == 1 else {}
    fcol = list(z['drive_names']).index('feeder')
    v_run = float(z['drive_f'][int(np.searchsorted(z['t'], 1.5)), fcol])
    check(stop_t is not None and stop_t < 9. and rel0['stop'] == 'beam' and len(went) == 1
          and t_next is not None and t_next > stop_t and (lead.get('lane_tail_s') or 1e9) > t_next
          and abs(v_run - 1.) < .05 and r['numerics']['ok'],
          'seed 392, first 12 s: lump %s tipped over the edge, the beam stopped the feed belt at %.2f s (drive at %.2f'
          ' of speed while approaching); the next release started at %.2f s, before that lump left the lane (%s)'
          % (went, stop_t, v_run, t_next,
             'at %.2f s' % lead['lane_tail_s'] if lead.get('lane_tail_s') else 'not by 12 s'))

    print('\nwall heights: skirt/side belt %.2f m, plough %.2f m against lump screen size %.2f-%.2f m'
          % (machine.SKIRT['height'], machine.PLOUGH['height'], cfg['size_min'], cfg['size_max']))
    print('all checks passed')


if __name__ == '__main__':
    main()
