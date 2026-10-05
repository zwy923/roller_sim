"""Checks of the station part of the line: buffer belt, measuring belt, flip separator, sensors, void items
held, and the controllers on sensor signals.

Run: python checks/station_checks.py            (python checks/station_checks.py Control  -- one class)
1. Geometry: the buffer belt one step below the main belt from its head edge, the measuring belt flush after
   it, both 0.70 m wide; the four beams (S2 where a lump lies flat again, low; S3 far enough back for the
   buffer belt's stop); the plate zone; the separator docked behind the measuring belt; what is rejected.
2. Devices: every beam has an emitter and a receiver on its line, every camera region (the feed belt camera
   and the plate zone included) and the measuring zone (two heads) is in view, no device reaches into the
   lump corridors, the scanner clears a standing lump.
3. Compiled model: belt tops and extents, contact categories, the separator's closed loop in both positions,
   and every moving part clearing all other equipment (devices included) over its stroke.
4. Weighing: a lump at rest on the measuring belt reads its own weight. Tracker: the cameras' lump count of an
   object does not grow when a ghost track runs into it; the pieces of a blob that comes apart keep its lineage.
5. Control on synthetic sensor signals (no physics): a coal lump and a gangue lump through; a void item for
   every reason is HELD -- no discharge, no plate move, no release -- and taken off the line the line runs on;
   an item found good is still held if a lump reaches it while it waits for the plate;
   the plate path needs S4 and the plate-zone camera, with a timeout; staging and the hand-over during a
   discharge; the section held only when the buffer cannot take a lump; the line frozen while the cameras are
   not healthy (stopped if that lasts), stopped by a dirty beam; the stall stop. Rules: the two optional ones
   (--weigh-stop centre, --buffer-approach slow). Baseline: the defaults fixed on 2026-10-05 -- the measuring
   device reads scan_s after the belt is at rest whatever the reading does (Control drives the steady-reading
   indicator, --weigh-model steady).
6. Physics (slow, about 15 min): the acceptance tests of 2026-09-30 -- two lumps laid touching on the buffer
   belt (weighed together), a lump laid across the joint against the weighed one, the weighed lump pushed back
   across the joint, the first scan failing: none is sorted as a valid result, each is held and taken off; the
   touching pair laid in the lane instead (the step parts them): nothing wrong passes; and seed 392 through the
   whole line with every item valid.
These establish the model's decisions and geometry, not what the hardware would do.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator import geom2d, machine, simulate  # noqa: E402
from singulator.config import SAMPLE_S, parse_config  # noqa: E402
from singulator.control import station  # noqa: E402
from singulator.lumps import make_blocks, set_mass_properties  # noqa: E402
from singulator.machine import assembly, sensors, separator, station as hw  # noqa: E402
from singulator.physics.lumps import END_STATES, Lump  # noqa: E402
from singulator.sensing import beams, vision  # noqa: E402
from singulator.sensing import weigher as scale  # noqa: E402
from singulator.sim.scenarios import items_gone  # noqa: E402
from singulator.verify.station import StationWitness, frame_contacts  # noqa: E402

CFG = parse_config(['--no-video', '--weigh-model', 'steady'])    # Control drives the indicator; see Baseline
D = machine.derive(CFG)
ST = D['station']
EDGE, XJ, MEND = ST['buffer']['x0'], ST['buffer']['x1'], ST['measure']['x1']
TOP, SC = ST['top_z'], ST['sep']
STOP_X, S2_X = ST['beam_stop']['x'], ST['beam_in']['x']
RATE = (SC.open_deg - SC.closed_deg) / CFG['separator_swing_s']


def derive(*args):
    return machine.derive(parse_config(['--no-video'] + list(args)))


def box(x0, x1, z0=0., z1=.30, y0=.2, y1=.55):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def on_b(x0, x1, **kw):
    return box(x0, x1, TOP, TOP + .30, **kw)


class Geometry(unittest.TestCase):
    def test_layout(self):
        sp = ST['separator']
        self.assertAlmostEqual(EDGE, D['belt_x1'])                     # B starts at the main belt's head edge
        self.assertAlmostEqual(D['final_x'], EDGE)                     # ... which is the measuring plane now
        self.assertAlmostEqual(ST['width_m'], .70)
        self.assertAlmostEqual(XJ - EDGE, CFG['buffer_len'])
        self.assertAlmostEqual(ST['measure']['x0'], XJ)                # M butts B
        self.assertAlmostEqual(MEND - XJ, CFG['measure_len'])
        self.assertAlmostEqual(TOP, -CFG['station_step'])
        self.assertAlmostEqual(ST['y_c'], CFG['lane_y'] + CFG['lane_w'] / 2)
        self.assertEqual((S2_X, ST['beam_in']['z']), (EDGE + hw.S2_X, TOP + hw.S2_Z))
        self.assertLessEqual(S2_X + hw.S2_ROOM, STOP_X + 1e-9)     # a lump lies flat past S2 before S3
        b_stop = CFG['buffer_speed'] * (hw.RAMP_S / 2 + SAMPLE_S + sensors.LATENCY_S)
        self.assertAlmostEqual(XJ - STOP_X, max(hw.STOP_BACK, b_stop + hw.STAGE_ROOM))
        self.assertAlmostEqual(ST['beam_stop']['z'], TOP + hw.STOP_Z)
        self.assertTrue(MEND < ST['beam_gangue']['x'] < sp['pivot'][0])
        self.assertLess(ST['beam_gangue']['z'], sp['inlet'][2] - .3)  # under the plate's closing sweep
        (zx0, _, zz0), (zx1, _, _) = ST['plate_zone']
        self.assertAlmostEqual(zx0, MEND, places=3)
        self.assertGreater(zx1, sp['outlet_closed'][0])
        self.assertLess(zz0, min(ST['beam_gangue']['z'], sp['outlet_closed'][2]))
        self.assertAlmostEqual(sp['inlet'][0], MEND + hw.GAP, places=3)
        self.assertAlmostEqual(sp['inlet'][2] + SC.rib_height, TOP - hw.DROP, places=3)
        self.assertAlmostEqual(sp['inlet'][2] - sp['floor_z'], 1.45, places=3)
        self.assertEqual((SC.width, SC.rib_count), (.70, 7))
        self.assertTrue(sp['coal_slides_when_closed'])               # tan 20 deg > 0.20 lining
        self.assertEqual((CFG['feeder_release'], CFG['sensing']), ('predict', 'vision'))   # the station defaults
        self.assertIn('taken_off', END_STATES)

    def test_rejects(self):
        for args in (['--lane-w', '.72'], ['--buffer-len', '.80'], ['--measure-len', '.80'], ['--scan-s', '0'],
                     ['--station-step', '0'], ['--sort-density', '-1']):
            with self.subTest(args=args), self.assertRaises(ValueError):
                derive(*args)
        for args in (('--buffer-speed', '4'), ('--buffer-len', '.9')):   # room for S2, S3 and B's stop
            with self.subTest(args=args), self.assertRaises(ValueError):
                derive(*args)


class Devices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = D['devices']
        cls.m = mujoco.MjModel.from_xml_string(assembly.build_xml(CFG, D, []))
        cls.d = mujoco.MjData(cls.m)
        separator.place(cls.m, cls.d, SC, SC.closed_deg)
        mujoco.mj_forward(cls.m, cls.d)

    def test_beams_are_the_controllers_lines(self):
        lines = {b['name']: b for b in self.dev['beams']}
        self.assertEqual((lines['beam_feed']['x'], lines['beam_feed']['z']), (D['feeder']['beam']['x'], D['feeder']['beam']['z']))
        for key in ('beam_in', 'beam_stop', 'beam_gangue'):
            self.assertEqual((lines[key]['x'], lines[key]['z']), (ST[key]['x'], ST[key]['z']))
        for b in self.dev['beams']:
            self.assertEqual(b['max_block_s'], sensors.MAX_BLOCK_S[b['name']])
            light = self.m.geom('dev_%s_light' % b['name'])
            self.assertAlmostEqual(self.d.geom_xpos[light.id][0], b['x'], places=4)
            self.assertAlmostEqual(self.d.geom_xpos[light.id][2], b['z'], places=4)
            for end, y in (('tx', b['y0']), ('rx', b['y1'])):
                h = self.m.geom('dev_%s_%s' % (b['name'], end)).id
                self.assertEqual(self.m.geom_contype[h], 2)             # equipment: lumps hit it
                c, s = self.d.geom_xpos[h], self.m.geom_size[h]
                self.assertLess(c[2] - s[2], b['z'])                     # the lens height is inside the housing
                self.assertLess(abs(abs(c[1] - y) - s[1]), 2e-4)         # the housing's face is on the line's end

    def test_cameras_see_what_their_controllers_use(self):
        self.assertEqual([c['name'] for c in self.dev['cameras']],
                         ['cam_feed', 'cam_singulator', 'cam_station', 'cam_separator'])
        cov = sensors.coverage(self.dev)
        self.assertEqual({k: v for k, v in cov.items() if v}, {})
        for cam in self.dev['cameras'] + self.dev['scanner']['heads']:
            c = self.m.camera('dev_%s_view' % cam['name'])
            self.assertAlmostEqual(self.m.cam_fovy[c.id], cam['fovy'], places=3)
            self.assertLess(np.linalg.norm(self.d.cam_xpos[c.id] - cam['pos']), 2e-4)

    def test_no_device_reaches_into_a_lump_corridor(self):
        corridors = [((D['feeder']['x0'], CFG['lane_y'], 0.), (EDGE, CFG['belt_w'], .70)),
                     ((EDGE, ST['y_c'] - ST['width_m'] / 2, TOP), (MEND, ST['y_c'] + ST['width_m'] / 2, TOP + .70)),
                     ((MEND, ST['y_c'] - ST['width_m'] / 2, ST['separator']['floor_z'] + .30),
                      (MEND + .40, ST['y_c'] + ST['width_m'] / 2, TOP))]
        hit = []
        for g in range(self.m.ngeom):
            name = self.m.geom(g).name
            if not name.startswith('dev_') or self.m.geom_contype[g] == 0:
                continue
            half = np.abs(self.d.geom_xmat[g].reshape(3, 3)) @ self.m.geom_size[g]     # AABB of the box
            lo, hi = self.d.geom_xpos[g] - half, self.d.geom_xpos[g] + half
            for clo, chi in corridors:
                if np.all(lo < chi) and np.all(hi > clo):
                    hit.append(name)
        self.assertEqual(hit, [])
        self.assertGreater(self.dev['scanner']['clear_height'], .70)    # a lump on end: largest caliper 0.67 m


class CompiledModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = mujoco.MjModel.from_xml_string(assembly.build_xml(CFG, D, []))
        cls.d = mujoco.MjData(cls.m)
        separator.place(cls.m, cls.d, SC, SC.closed_deg)
        mujoco.mj_forward(cls.m, cls.d)

    def geom_box(self, body):
        g = self.m.body_geomadr[self.m.body(body).id]
        c, s = self.d.geom_xpos[g], self.m.geom_size[g]
        return c - s, c + s

    def test_belts_and_categories(self):
        lo, hi = self.geom_box('mfloor')
        self.assertAlmostEqual(hi[2], 0., places=9)
        self.assertAlmostEqual(hi[0], EDGE, delta=2e-4)
        for body, key in (('bbelt', 'buffer'), ('mbelt', 'measure')):
            lo, hi = self.geom_box(body)
            self.assertAlmostEqual(hi[2], TOP, places=9)
            self.assertAlmostEqual(lo[0], ST[key]['x0'], delta=2e-4)
            self.assertAlmostEqual(hi[0], ST[key]['x1'], delta=2e-4)
            self.assertAlmostEqual(hi[1] - lo[1], .70, delta=2e-4)
        cat = assembly.categorise(self.m)
        kind = lambda test: {assembly.CATS[cat[g]] for g in range(self.m.ngeom) if test(self.m.geom(g).name,
                                                                                         self.m.body(self.m.geom_bodyid[g]).name)}
        self.assertEqual(kind(lambda g, b: b == 'mbelt' or g.startswith('mskirt')), {'weigher'})
        self.assertEqual(kind(lambda g, b: b == 'bbelt' or g.startswith('bskirt')), {'buffer'})
        self.assertEqual(kind(lambda g, b: g.startswith('dev_')), {'device'})
        self.assertEqual(assembly.CATS[cat[self.m.geom('sep_deck').id]], 'separator')
        self.assertAlmostEqual(self.m.geom_friction[self.m.geom('sep_deck').id][0], CFG['separator_friction'])

    def test_linkage_closes_in_both_positions(self):
        for deg in (SC.closed_deg, SC.open_deg):
            separator.place(self.m, self.d, SC, deg)
            mujoco.mj_forward(self.m, self.d)
            pin = np.linalg.norm(self.d.site('sep_rod_pin').xpos - self.d.site('sep_plate_pin').xpos)
            self.assertLess(pin, 1e-6)
            self.assertAlmostEqual(math.degrees(self.d.joint('sep_plate_hinge').qpos[0]), deg)
        separator.place(self.m, self.d, SC, SC.closed_deg)
        mujoco.mj_forward(self.m, self.d)

    def test_moving_parts_clear(self):
        res = assembly.motion_clearance(CFG, D)
        self.assertIn('separator', res)
        for mech, r in res.items():
            with self.subTest(mech=mech):
                self.assertGreater(r['min_distance_m'], 1e-3)
                self.assertFalse(r['unresolved'])


class Weighing(unittest.TestCase):
    def test_lump_at_rest_reads_its_weight(self):
        blocks = make_blocks(CFG, np.random.default_rng(7))[:1]
        m = mujoco.MjModel.from_xml_string(assembly.build_xml(CFG, D, blocks))
        d = mujoco.MjData(m)
        set_mass_properties(blocks[0], m.body_mass[m.body('b0').id])
        separator.place(m, d, SC, SC.closed_deg)
        L = Lump(m, 0)
        L.place(d, XJ + .45, ST['y_c'], TOP + .30, 0.)
        mujoco.mj_forward(m, d)
        low = L.world(d)[:, 2].min()
        d.qpos[L.q + 2] += TOP + .002 - low                            # 2 mm above the measuring belt
        belts = [(m.jnt_qposadr[m.joint(j).id], m.jnt_dofadr[m.joint(j).id])
                 for j in ('mfloorj', 'feederj', 'bbeltj', 'mbeltj')]
        frame = assembly.weigh_frame(m)
        self.assertEqual(frame, {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == 'mbelt'
                                 or m.geom(g).name.startswith('mskirt')})        # M and its skirts
        cell = scale.LoadCell(m, frame, {L.geom: 0})
        loads = []
        for i in range(int(2. / m.opt.timestep)):
            for q, v in belts:                                        # every belt held at rest
                d.qpos[q], d.qvel[v] = 0., 0.
            mujoco.mj_step(m, d)
            if i * m.opt.timestep > 1.:
                loads.append(cell.force(d))
        on, other = frame_contacts(d, frame, {L.geom: 0})
        self.assertLess(abs(np.mean(loads) / scale.G - blocks[0]['mass_kg']) / blocks[0]['mass_kg'], .005)
        self.assertEqual(on, {0})
        self.assertFalse(other.get(0, set()))                         # touches nothing off the weigh frame


class Tracker(unittest.TestCase):
    """sensing.vision.Vision's lump count of an object (n_est: the station holds an item 'multi' above 1)."""

    def test_a_ghost_track_adds_no_lumps_to_a_merge(self):
        """A ghost: a track no blob has matched for a few frames -- what is left when a blob comes apart and its
        pieces go on under other ids. It still lies inside one of them, and the next frame merges the two tracks.
        Until 2026-10-04 the merge added the ghost's count: one lump lying next to others on the feed belt was
        counted as 13 after 19 s (scatter 7001), and the station held it -- a single lump void."""
        lump = box(-1.5, -1.1, D['feeder']['step_m'], D['feeder']['step_m'] + .30, y0=.3, y1=.7)   # on the feed belt

        v = vision.Vision(D['devices']['cameras'], np.random.default_rng(0), exits=lambda x, y: False)
        for i in range(3):
            v.frame(i * .01, {0: lump}, lambda a, b: 1.)
        (tr,) = v.tracks.values()
        v.tracks[99] = dict(tr, tid=99, lineage=frozenset({99}), count=2, seen=-.01,
                            hist=[(-.01, tr['m']['cx'], tr['m']['cy'])])
        v.next_tid = 100
        v.frame(.03, {0: lump}, lambda a, b: 1.)
        (merged,) = v.tracks.values()
        self.assertEqual((v.counts['merges'], merged['lineage'] >= {99}), (1, True))
        self.assertEqual(merged['count'], 1)                        # the ghost is absorbed: still one lump (its
                                                                    # count of 2 was added until 2026-10-04: 3)

    def test_the_pieces_of_a_blob_that_comes_apart_keep_its_lineage(self):
        """Two lumps in a row are one blob; its outline centroid lies between them, more than MATCH_GATE from
        either lump's own. When they part, neither piece is matched to the blob's track. Until 2026-10-04 both were
        new objects without a past (and the blob's track a ghost): the station found an item's object gone and an
        unknown one in its place, and held a single lump 'track_lost' (touching 7004)."""
        z = D['feeder']['step_m']
        lump = lambda x: box(x, x + .50, z, z + .30, y0=.3, y1=.7)     # on the feed belt, 0.50 m long

        v = vision.Vision(D['devices']['cameras'], np.random.default_rng(0), exits=lambda x, y: False)
        for i in range(3):
            v.frame(i * .01, {0: lump(-1.5), 1: lump(-2.)}, lambda a, b: 0.)        # touching: one blob
        (blob,) = v.tracks
        v.frame(.03, {0: lump(-1.44), 1: lump(-2.06)}, lambda a, b: .12)            # 12 cm apart
        pieces = [tr for tr in v.tracks.values() if tr['seen'] > .025]
        self.assertEqual((len(pieces), [blob in tr['lineage'] for tr in pieces]), (2, [True, True]))
        self.assertEqual((blob in v.tracks, v.counts['births'], v.counts['apart']), (False, 1, 1))


# ---- control on synthetic sensor signals --------------------------------------------------------------
class Line:
    """Synthetic sensors for the station controller: every box (8 corners) is a lump and one camera object
    (tid = its index) unless told otherwise; the beams are sensing.beams.Beam line tests on the boxes; the weigher
    is the configured device (--weigh-model) on a load cell that reads `load` (N); this object stands in for the
    volume device and answers self.verdict scan_s after the belt came to rest. A witness holds the truth the
    harness states (which lumps are on the scale, where they landed)."""

    def __init__(self, densities=(1300., 2500., 1350.), cfg=CFG, d=D):
        self.blocks = [dict(mass_kg=50., volume_m3=50. / rho, density=rho, material='coal' if rho < 1800 else 'gangue')
                       for rho in densities]
        self.n, self.cfg = len(self.blocks), cfg
        self.weigher = scale.make_weigher(cfg)
        self.s = station.Station(cfg, d, self.weigher, self, margin=3 * vision.POS_SIGMA)
        self.witness = StationWitness(cfg, self.blocks).attach(self.s)
        self.beams = {b['name']: beams.Beam(b['name'], b['x'], b['z'], b['max_block_s'], oracle=True)
                      for b in d['devices']['beams']}
        self.t, self.verdict, self.scans = 0., dict(valid=True, reasons=[]), 0
        self.on_m = None

    # ---- the volume device (the interface of sensing.volume.Scanner) ----
    def read(self, t, rest):
        if t < rest + self.cfg['scan_s'] - 1e-9:
            return None
        self.scans += 1
        v = dict(self.verdict)
        if v['valid']:
            v['volume_m3'] = self.blocks[self.on_m]['volume_m3']
        return v

    def describe(self):
        return dict(volume='the harness')

    def report(self):
        """The station's report with the verification, as a run writes it."""
        return self.witness.report(self.s.report())

    def take_off(self, lumps):
        """What sim.scenarios.TakeOff does to the station."""
        self.s.took_off(self.t, items_gone(self.s, lumps), lumps)

    def see(self, boxes, load=None, plate=None, n_est=None, conf=None, health=True, states=None, bins=None,
            merge=None, on=(), lin=None):
        """One sample 10 ms on. boxes: {lump: box}; merge: (a, b) -> one object for lumps a and b (tid 100 + a,
        lineage of both); load: the measuring belt's force (N) for the physics steps since the last sample."""
        s = self.s
        self.t = round(self.t + .01, 4)
        view = vision.Frame()
        for k, b in boxes.items():
            if merge and k == merge[1]:
                continue
            V = np.concatenate([b, boxes[merge[1]]]) if merge and k == merge[0] else b
            P = V[:, :2]
            o = dict(pts=geom2d.hull2(P), cx=float((P[:, 0].min() + P[:, 0].max()) / 2),
                     cy=float((P[:, 1].min() + P[:, 1].max()) / 2), x0=float(P[:, 0].min()), x1=float(P[:, 0].max()),
                     y0=float(P[:, 1].min()), y1=float(P[:, 1].max()), ztop=float(V[:, 2].max()), solid=1., vis=1.,
                     _truth=(k,), tid=k, lineage=frozenset({k}), conf=1. if conf is None else conf.get(k, 1.),
                     coasting=False, n_est=1 if n_est is None else n_est.get(k, 1), vx=None, vy=None, age_s=1.)
            if merge and k == merge[0]:
                o.update(tid=100 + k, lineage=frozenset({k, merge[1], 100 + k}), n_est=2, _truth=tuple(merge))
            elif lin and k in lin:                                      # a split: the halves remember the blob
                o['lineage'] = frozenset(lin[k]) | {k}
            view[o['tid']] = o
        view.t, view.health = self.t, dict(ok=health, why=None if health else 'camera_outage')
        for b in self.beams.values():
            b.sample(self.t, list(boxes.values()))
        s.u, s.b, s.m = s.u_goal, s.b_goal, s.m_goal                  # the drives follow their goals at once
        if load is not None and s.weighing:
            self.weigher.load_step(load)
        self.weigher.sample(0. if load is None else load)
        self.witness.truth(states or ['on_belt'] * self.n, bins or [None] * self.n, set(on), {})
        s.observe(self.t, view, self.beams, s.plate_deg if plate is None else plate)

    def run_to(self, k, x_front, boxes, step=.01, **kw):
        """Carry lump k (0.40 m long, lying on B / M) until its front reaches x_front, a sample per step."""
        x = boxes[k][:, 0].max()
        while x < x_front - 1e-9:
            x = min(x_front, x + step)
            boxes[k] = on_b(x - .40, x) if x - .40 > EDGE else box(x - .40, x)
            self.see(boxes, **kw)

    def onto_measuring_belt(self, boxes, k=0, **kw):
        """Lump k over the head edge, along B past S2 and S3 (M empty: no stop), across and wholly onto M."""
        s = self.s
        boxes[k] = box(EDGE - .30, EDGE + .10)
        self.see(boxes, **kw)
        self.run_to(k, XJ + .45, boxes, **kw)
        it = s.items[-1]
        self.on_m = k
        return it

    def measure(self, boxes, it, k=0, mass=50., steady=True, extra=0):
        """Readings at rest until the route is known (or the item is held), then `extra` samples more."""
        i = 0
        while it['stage'] == 'measure' or extra > 0:
            if it['stage'] != 'measure':
                extra -= 1
            i += 1
            # unsteady: a slow 5 % swing (a lump still rocking) -- a sample-to-sample alternation the
            # indicator's 0.3 s average would smooth away
            f = mass * scale.G * (1. if steady else 1. + .05 * math.sin(2 * math.pi * i * .01 / .6))
            self.see(boxes, load=f, on=[k])
        return i


class Control(unittest.TestCase):
    def setUp(self):
        self.L = Line()
        self.s = self.L.s

    def far(self, i):
        return box(EDGE - 3. - i, EDGE - 2.6 - i)

    def test_coal_lump_through(self):
        L, s = self.L, self.s
        boxes = {0: self.far(0)}
        it = L.onto_measuring_belt(boxes)
        self.assertEqual((it['stage'], s.m_goal), ('measure', 0.))
        self.assertEqual(it['handover']['interruptions'], 1)             # S2 saw it once
        self.assertEqual(it['reasons'], [])
        L.measure(boxes, it)
        self.assertEqual((it['route'], it['void'], it['steady']), ('coal', False, True))
        self.assertAlmostEqual(it['density_kg_m3'], 1300., places=0)
        self.assertEqual((it['stage'], it['plate_at_start']), ('discharge', 'closed'))
        boxes[0] = box(MEND + .03, MEND + .43, TOP - .10, TOP + .20)
        L.see(boxes)
        self.assertEqual(it['stage'], 'sent')
        del boxes[0]                                                    # slid off the plate, out of the zone
        L.see(boxes, states=['sorted', 'on_belt', 'on_belt'], bins=['coal', None, None])
        self.assertEqual((it['bins'], it['bins_match_route'], it['t_clear_s'] is not None), ({'0': 'coal'}, True, True))
        self.assertEqual(L.report()['verification'], dict(L.report()['verification'], false_valid=[]))

    def held(self, it):
        """A held item: on M, no discharge, no plate move, no release."""
        s = self.s
        self.assertEqual(it['stage'], 'hold')
        self.assertTrue(it['void'])
        self.assertNotIn('route', it)
        self.assertEqual((s.m_goal, s.plate, s.plate_goal), (0., 'closed', SC.closed_deg))
        self.assertTrue(s.inhibit_release)

    def test_unsteady_reading_is_held(self):
        L = self.L
        boxes = {0: self.far(0)}
        it = L.onto_measuring_belt(boxes)
        L.measure(boxes, it, steady=False, extra=300)                  # 3 s more: nothing moves
        self.assertEqual(it['reasons'], ['unsteady'])
        self.held(it)
        self.assertNotIn('t_discharge_s', it)

    def test_two_objects_or_two_lumps_in_one_item_are_held(self):
        L = self.L
        boxes = {0: self.far(0), 1: self.far(1)}
        boxes[0], boxes[1] = box(EDGE - .30, EDGE + .10), box(EDGE - .75, EDGE - .35)
        L.see(boxes)
        for i in range(60):                                            # both over, 5 cm apart: one item
            boxes[0] = on_b(EDGE - .20 + .01 * i, EDGE + .20 + .01 * i) if EDGE - .2 + .01 * i > EDGE else box(EDGE - .30 + .01 * i, EDGE + .10 + .01 * i)
            boxes[1] = box(EDGE - .65 + .01 * i, EDGE - .25 + .01 * i) if i < 45 else on_b(EDGE - .65 + .01 * i, EDGE - .25 + .01 * i)
            L.see(boxes)
        self.assertEqual(len(L.s.items), 1)
        it = L.s.items[0]
        self.assertEqual(sorted(it['now']), [0, 1])
        M = Line()
        boxes = {0: self.far(0)}
        it = M.onto_measuring_belt(boxes, n_est={0: 2})               # one object the cameras count as two lumps
        M.measure(boxes, it)
        self.assertIn('multi', it['reasons'])
        self.assertEqual(it['stage'], 'hold')

    def test_lumps_drawn_apart_on_the_buffer_become_two_items(self):
        # two lumps go over the head edge as one blob (one item); B running at twice the main belt's speed pulls
        # the lead away while the other is still on the main belt. Drawn GROUP_GAP apart they become two items,
        # and S2 confirms each hand-over with its own interruption -- none of it is void
        cfg = parse_config(['--no-video', '--buffer-len', '1.5', '--buffer-speed', '.8'])
        d = machine.derive(cfg)
        L = Line(cfg=cfg, d=d)
        s, edge, top = L.s, d['station']['buffer']['x0'], d['station']['top_z']
        vb, vm = cfg['buffer_speed'] * .01, cfg['v_belt'] * .01             # per 10 ms sample
        lead, rear = edge + .42, edge + .02                                # fronts of the two 0.40 m lumps
        boxes = {0: box(lead - .40, lead, top, top + .30), 1: box(rear - .40, rear)}
        L.see(boxes, merge=(0, 1))
        self.assertEqual(len(s.items), 1)
        for _ in range(300):
            lead += vb
            rear += vb if rear - .20 > edge else vm                          # on B once its centroid is over the edge
            boxes[0] = box(lead - .40, lead, top, top + .30)
            boxes[1] = box(rear - .40, rear, top, top + .30) if rear - .40 > edge else box(rear - .40, rear)
            L.see(boxes, lin={0: {100}, 1: {100}})
            if rear - .40 > d['station']['beam_in']['x'] + .05:
                break
        L.see(boxes, lin={0: {100}, 1: {100}})
        self.assertEqual(len(s.items), 2)
        a, b = s.items
        self.assertEqual((a['split'], b['split_from'], s.counts['splits']), ([1], 0, 1))
        self.assertEqual((a['tids'], b['tids']), ({0}, {1}))
        self.assertEqual((a['handover']['interruptions'], b['handover']['interruptions']), (1, 1))
        self.assertEqual((a['reasons'], b['reasons'], a['n_obj_max'], b['n_obj_max']), ([], [], 1, 1))

    def test_nothing_leaves_the_measuring_belt_before_it_is_measured(self):
        # one item whose two lumps draw apart only once the lead is crossing onto M (seed 4014; drawn apart on B
        # they would have become two items): the lead one must not ride off M's head edge while M waits for the
        # other to come wholly on -- M stops, and the item is held
        L, s = self.L, self.s
        L.on_m = 0
        boxes = {0: on_b(EDGE + .02, EDGE + .42), 1: box(EDGE - .38, EDGE + .02)}
        L.see(boxes, merge=(0, 1))                                     # they go over the edge as one blob ...
        it = s.items[0]
        lin = {0: {100}, 1: {100}}
        x, r = EDGE + .42, EDGE + .02
        while it['stage'] in ('buffer', 'transfer', 'onm') and x < MEND + .2:
            x += .01                                                    # ... together along B, then split on M,
            r += .01 if x < XJ + .05 else .0033                         # the lead running ahead
            boxes[0] = on_b(x - .40, x)
            boxes[1] = on_b(r - .40, r) if r - .40 > EDGE else box(r - .40, r)
            L.see(boxes, lin=lin)
        self.assertLess(boxes[0][:, 0].max(), MEND)                    # stopped short of the head edge
        self.assertEqual((it['stage'], s.m_goal), ('measure', 0.))
        L.measure(boxes, it, extra=100)
        self.assertIn('outside', it['reasons'])
        self.held(it)

    def test_outside_the_weigh_zone_or_a_neighbour_too_close_is_held(self):
        L = self.L
        boxes = {0: self.far(0), 1: self.far(1)}
        it = L.onto_measuring_belt(boxes)
        boxes[1] = on_b(XJ - .42, XJ - .02)                            # a lump on B 2 cm short of the joint ...
        boxes[0] = on_b(XJ + .01, XJ + .41)                            # ... and the weighed one just past it
        L.measure(boxes, it)
        self.assertIn('outside', it['reasons'])
        self.assertEqual(it['stage'], 'hold')

    def test_a_lump_that_reaches_a_decided_item_makes_it_void(self):
        """A gangue lump is weighed and found good; the plate needs two seconds to open for it and it waits on M.
        A lump on B rolls on over the joint against it. Until 2026-10-04 nothing looked any more once the route was
        known: the two went over the plate together, the second one unmeasured (drum head trial, aligned 7011)."""
        L = Line(densities=(2500., 1300.))
        s, boxes = L.s, {0: self.far(0), 1: self.far(1)}
        it = L.onto_measuring_belt(boxes)
        L.measure(boxes, it, mass=50.)
        self.assertEqual((it['stage'], it['void'], it['route'], s.plate), ('decided', False, 'gangue', 'opening'))
        boxes[1] = on_b(XJ - .38, XJ + .02)                             # 3 cm from the waiting item
        L.see(boxes)
        self.assertEqual((it['stage'], it['void'], it['reasons']), ('hold', True, ['outside']))
        self.assertEqual(it['revoked']['route'], 'gangue')
        self.assertNotIn('route', it)
        self.assertNotIn('t_discharge_s', it)
        self.assertEqual((s.m_goal, s.counts['holds']), (0., 1))
        self.assertTrue(s.inhibit_release)
        self.assertEqual(s.report()['counts']['void_discharged'], 0)

    def test_a_failed_scan_is_held(self):
        L = self.L
        L.verdict = dict(valid=False, reasons=['scan_failed'])
        boxes = {0: self.far(0)}
        it = L.onto_measuring_belt(boxes)
        L.measure(boxes, it)
        self.assertEqual((it['reasons'], it['scan']['reasons']), (['scan'], ['scan_failed']))
        self.held(it)

    def test_an_implausible_density_is_held(self):
        L = self.L
        boxes = {0: self.far(0)}
        it = L.onto_measuring_belt(boxes)
        L.measure(boxes, it, mass=15.)                                 # 15 kg for a 38 L lump: 390 kg/m3
        self.assertIn('implausible', it['reasons'])
        self.held(it)

    def test_handover_needs_s2(self):
        L = self.L
        L.beams['beam_in'].fault = 'dead'
        boxes = {0: self.far(0), 1: self.far(1)}
        it = L.onto_measuring_belt(boxes)
        self.assertEqual((it['reasons'], it['reason_info']['handover']['detail']), (['handover'], 'S2 saw nothing'))
        L.measure(boxes, it)
        self.held(it)
        L.take_off([0])                                                # take it off: the second hand-over ...
        del boxes[0]
        it2 = L.onto_measuring_belt(boxes, k=1)
        self.assertEqual((L.beams['beam_in'].alarm or {}).get('why'), 'dead')   # ... two in a row: S2 is dead
        self.assertEqual(L.s.fault['reason'], 'beam_dead')
        self.assertTrue(L.s.frozen)
        self.assertIn('handover', it2['reasons'])

    def test_a_held_item_taken_off_frees_the_line(self):
        L, s = self.L, self.s
        boxes = {0: self.far(0), 1: self.far(1)}
        it = L.onto_measuring_belt(boxes)
        L.measure(boxes, it, steady=False)
        self.held(it)
        L.take_off([0])
        del boxes[0]
        self.assertEqual((it['stage'], it['taken_off'], s.line), ('taken_off', [0], []))
        self.assertFalse(s.inhibit_release)
        L.see(boxes)
        self.assertEqual((s.m_goal, s.b_goal), (1., 1.))
        it2 = L.onto_measuring_belt(boxes, k=1)                         # the next lump is measured as usual
        L.measure(boxes, it2, k=1)
        self.assertEqual((it2['void'], it2['route']), (False, 'gangue'))
        self.assertEqual(s.report()['counts']['holds'], 1)

    def test_an_item_taken_off_with_the_held_one_leaves_the_line(self):
        # lane 0.58 m trial, aligned 6006: the next lump had run on past S3 and its front was across the joint when
        # the held one was taken off, so it went too; its item stayed on M and M never stopped for the one after
        L, s = self.L, self.s
        boxes = {0: self.far(0), 1: self.far(1), 2: self.far(2)}
        it = L.onto_measuring_belt(boxes)
        L.measure(boxes, it, steady=False)
        self.held(it)
        boxes[1] = on_b(XJ - .39, XJ + .01)
        L.see(boxes)
        it1 = s.items[-1]
        self.assertEqual(it1['stage'], 'transfer')
        L.take_off([0, 1])
        del boxes[0], boxes[1]
        self.assertEqual((it1['stage'], s.line), ('taken_off', []))
        L.see(boxes)
        it2 = L.onto_measuring_belt(boxes, k=2)
        self.assertEqual(it2['stage'], 'measure')                       # M stops for it as usual

    def test_gangue_plate_path_needs_s4_and_the_zone_camera(self):
        L, s = self.L, self.s
        boxes = {1: self.far(1)}
        it = L.onto_measuring_belt(boxes, k=1)
        L.measure(boxes, it, k=1)
        self.assertEqual((it['route'], s.plate, it['t_plate_move_s']), ('gangue', 'opening', it['t_decided_s']))
        for _ in range(400):                                            # plate swings; the lump discharges
            L.see(boxes, plate=min(SC.open_deg - .1, s.plate_deg + RATE * .01))
            if it['stage'] == 'discharge':
                break
        boxes[1] = box(MEND + .03, MEND + .43, TOP - .10, TOP + .20)
        L.see(boxes, plate=SC.open_deg - .1)
        self.assertEqual((it['stage'], s.plate), ('sent', 'open'))
        gz = ST['beam_gangue']['z']
        boxes[1] = box(MEND + .05, MEND + .45, gz - .10, gz + .20)       # falling through the gap: S4 blocked
        L.see(boxes, plate=SC.open_deg - .1)
        L.see(boxes, plate=SC.open_deg - .1)
        stuck = box(MEND + .3, MEND + .7, ST['separator']['inlet'][2] - .2, ST['separator']['inlet'][2] + .1)
        boxes[1] = stuck                                                # S4 clear again, but something is still in the zone
        for _ in range(50):
            L.see(boxes, plate=SC.open_deg - .1)
        self.assertNotIn('t_clear_s', it)
        self.assertEqual(s.plate, 'open')                               # the plate does not close on it
        del boxes[1]
        L.see(boxes, plate=SC.open_deg - .1, states=['on_belt', 'sorted', 'on_belt'], bins=[None, 'gangue', None])
        self.assertEqual((it['path_confirmed_by'], s.plate), ('S4 + plate-zone camera', 'closing'))

    def test_plate_path_timeout_stops_the_line(self):
        L, s = self.L, self.s
        boxes = {1: self.far(1)}
        it = L.onto_measuring_belt(boxes, k=1)
        L.measure(boxes, it, k=1)
        for _ in range(400):
            L.see(boxes, plate=min(SC.open_deg - .1, s.plate_deg + RATE * .01))
            if it['stage'] == 'discharge':
                break
        boxes[1] = box(MEND + .03, MEND + .43, TOP - .10, TOP + .20)
        L.see(boxes, plate=SC.open_deg - .1)
        del boxes[1]                                                    # gone from view, S4 never saw it
        for _ in range(int(station.PATH_TIMEOUT_S / .01) + 5):
            L.see(boxes, plate=SC.open_deg - .1)
        self.assertEqual(s.fault['reason'], 'separator_path_timeout')

    def test_buffer_stages_while_busy_and_hands_over_during_the_discharge(self):
        L, s = self.L, self.s
        boxes = {0: self.far(0), 1: self.far(1)}
        a = L.onto_measuring_belt(boxes)
        boxes[1] = box(EDGE - .30, EDGE + .10)
        weigh = dict(load=50. * scale.G, on=[0])
        L.see(boxes, **weigh)
        L.run_to(1, STOP_X + .01, boxes, **weigh)                       # cuts the staging beam, M busy
        b = s.items[s.items.index(a) + 1]
        self.assertEqual((b['stage'], s.b_goal, s.counts['buffer_stops']), ('buffer', 0., 1))
        L.measure(boxes, a)                                             # coal: discharge at once
        self.assertEqual((a['stage'], s.b_goal), ('discharge', 0.))
        boxes[0] = on_b(XJ + station.ACCEPT_AFTER + .01, XJ + station.ACCEPT_AFTER + .41)
        L.see(boxes)
        self.assertEqual(s.b_goal, 1.)                                  # M takes the next one while discharging

    def test_section_held_only_when_the_buffer_cannot_take_a_lump(self):
        L, s = self.L, self.s
        boxes = {0: self.far(0), 1: self.far(1), 2: self.far(2)}
        a = L.onto_measuring_belt(boxes)
        weigh = dict(load=50. * scale.G, on=[0])
        boxes[1] = box(EDGE - .30, EDGE + .10)
        L.see(boxes, **weigh)
        L.run_to(1, STOP_X + .01, boxes, **weigh)
        self.assertEqual((s.b_goal, s.u_goal), (0., 1.))
        boxes[2] = box(EDGE - .40, EDGE - .08)                          # centroid 0.24 m back: not yet
        L.see(boxes, **weigh)
        self.assertEqual(s.u_goal, 1.)
        boxes[2] = box(EDGE - .27, EDGE + .11)                          # centroid within HOLD_ZONE, B stopped
        L.see(boxes, **weigh)
        self.assertEqual((s.u_goal, s.section_held, s.counts['section_holds']), (0., True, 1))

    def test_the_line_freezes_while_the_cameras_are_not_healthy(self):
        L, s = self.L, self.s
        boxes = {0: self.far(0)}
        L.see(boxes, health=False)
        self.assertTrue(s.frozen and s.inhibit_release)
        self.assertEqual((s.u_goal, s.b_goal, s.m_goal), (0., 0., 0.))
        L.see(boxes)
        self.assertFalse(s.frozen)
        self.assertEqual(s.u_goal, 1.)
        for _ in range(int(station.FREEZE_MAX_S / .01) + 5):            # not healthy again, and it lasts
            L.see(boxes, health=False)
        self.assertEqual(s.fault['reason'], 'vision_lost')

    def test_a_dirty_beam_stops_the_line(self):
        L, s = self.L, self.s
        boxes = {0: self.far(0)}
        L.beams['beam_stop'].fault = 'dirty'                            # reads blocked, nothing near it
        for _ in range(int(beams.DIAG_S / .01) + 5):
            L.see(boxes)
        self.assertEqual((s.alarm['why'], s.fault['reason'], s.frozen), ('dirty', 'beam_dirty', True))

    def test_stall_stops_the_line(self):
        L, s = self.L, self.s
        s.sent.append(dict(item=0, tids={0}, route='coal', t_discharge_s=L.t, stage='sent', now=[], objs=[],
                           n_obj_max=1, n_est_max=1, truth_lumps=[0]))
        boxes = {0: box(MEND + .5, MEND + .9, TOP - .3, TOP)}
        for _ in range(int(station.STALL_S / .01) + 5):
            L.see(boxes)
        self.assertIn(s.fault['reason'], ('station_stall', 'separator_path_timeout'))


# ---- physics: the acceptance tests ---------------------------------------------------------------------
class Rules(unittest.TestCase):
    """The two optional rules of 2026-10-04 (off by default), on synthetic sensor signals."""

    def line(self, *args):
        cfg = parse_config(['--no-video'] + list(args))
        return Line(cfg=cfg, d=machine.derive(cfg))

    def test_weigh_stop_centre_carries_the_item_to_the_middle_of_the_weigh_zone(self):
        for stop, front in (('rear', XJ + .45), ('centre', XJ + .59)):
            with self.subTest(weigh_stop=stop):
                L = self.line('--weigh-stop', stop)
                s, boxes = L.s, {0: box(EDGE - .30, EDGE + .10)}
                L.see(boxes)
                L.run_to(0, XJ + .45, boxes)                            # 0.40 m long: wholly on M, rear 5 cm on
                it = s.items[0]
                self.assertEqual((it['stage'], s.m_goal), ('measure', 0.) if stop == 'rear' else ('onm', 1.))
                L.run_to(0, front, boxes)                               # centre: on, to one stop distance short
                self.assertEqual((it['stage'], s.m_goal), ('measure', 0.))    # ... of the middle (0.386 m)
                self.assertAlmostEqual(it['box'][1], front, places=6)
                L.on_m = 0
                L.measure(boxes, it)
                self.assertEqual((it['reasons'], it['void'], it['route']), ([], False, 'coal'))

    def test_weigh_stop_centre_still_holds_an_item_that_is_not_wholly_on(self):
        # the lead of two lumps in one item is at the stop position while the other is still crossing: as before
        L = self.line('--weigh-stop', 'centre')
        s = L.s
        boxes = {0: on_b(EDGE + .02, EDGE + .42), 1: box(EDGE - .38, EDGE + .02)}
        L.see(boxes, merge=(0, 1))
        it, lin = s.items[0], {0: {100}, 1: {100}}
        x, r = EDGE + .42, EDGE + .02
        while it['stage'] in ('buffer', 'transfer', 'onm') and x < MEND + .2:
            x += .01
            r += .01 if x < XJ + .05 else .0033
            boxes[0] = on_b(x - .40, x)
            boxes[1] = on_b(r - .40, r) if r - .40 > EDGE else box(r - .40, r)
            L.see(boxes, lin=lin)
        self.assertLess(boxes[0][:, 0].max(), MEND)
        self.assertEqual((it['stage'], s.m_goal, it['reasons']), ('measure', 0., ['outside']))

    def test_buffer_approach_slow_runs_the_last_stretch_at_the_measuring_belts_speed(self):
        half = CFG['station_speed'] / CFG['buffer_speed']
        for approach in ('full', 'slow'):
            with self.subTest(buffer_approach=approach):
                L = self.line('--buffer-approach', approach)
                s, boxes = L.s, {0: box(EDGE - .30, EDGE + .10)}
                L.see(boxes)
                L.run_to(0, XJ - station.APPROACH - .01, boxes)
                self.assertEqual(s.b_goal, 1.)
                L.run_to(0, XJ - station.APPROACH + .02, boxes)         # the lead's front within APPROACH
                self.assertEqual(s.b_goal, half if approach == 'slow' else 1.)
                L.run_to(0, XJ - .01, boxes)                            # past S3, M free: on, no stop
                self.assertEqual((s.b_goal, s.counts['buffer_stops']), (half if approach == 'slow' else 1., 0))
                L.run_to(0, XJ + .10, boxes)                            # crossing: at M's speed in both
                self.assertEqual((s.items[0]['stage'], s.b_goal), ('transfer', half))

    def test_buffer_approach_slow_still_stages_at_s3(self):
        L = self.line('--buffer-approach', 'slow')
        s = L.s
        boxes = {0: box(EDGE - 3., EDGE - 2.6), 1: box(EDGE - 4., EDGE - 3.6)}
        a = L.onto_measuring_belt(boxes)
        weigh = dict(load=50. * scale.G, on=[0])
        boxes[1] = box(EDGE - .30, EDGE + .10)
        L.see(boxes, **weigh)
        L.run_to(1, STOP_X + .01, boxes, **weigh)                       # cuts the staging beam, M busy
        self.assertEqual((a['stage'], s.items[1]['stage'], s.b_goal, s.counts['buffer_stops']),
                         ('measure', 'buffer', 0., 1))


class Baseline(unittest.TestCase):
    """The line as fixed on 2026-10-05 (user): the feed belt controller reads true centroids, the measuring device
    is a concept that takes scan_s; the station still reads the cameras."""

    def test_defaults(self):
        cfg = parse_config(['--no-video'])
        self.assertEqual((cfg['sensing'], cfg['feeder_sensing'], cfg['weigh_model'], cfg['scan_s']),
                         ('vision', 'oracle', 'fixed', 1.))
        self.assertEqual(Line(cfg=cfg, d=machine.derive(cfg)).s.geometry()['measure']['weigh_model'], 'fixed')

    def test_the_fixed_device_reads_after_scan_s_whatever_the_reading_does(self):
        cfg = parse_config(['--no-video'])
        for steady in (True, False):
            with self.subTest(steady=steady):
                L = Line(cfg=cfg, d=machine.derive(cfg))
                boxes = {0: box(EDGE - 3., EDGE - 2.6)}
                it = L.onto_measuring_belt(boxes)
                L.measure(boxes, it, steady=steady)
                self.assertEqual((it['void'], it['reasons'], it['route'], it['steady']), (False, [], 'coal', steady))
                self.assertAlmostEqual(it['weigh_wait_s'], cfg['scan_s'], places=2)
                self.assertEqual(L.scans, 1)
                self.assertAlmostEqual(it['mass_kg'], 50., delta=50. * .05)    # a rocking reading: within its swing

    def test_the_steady_indicator_is_an_option(self):
        cfg = parse_config(['--no-video', '--weigh-model', 'steady'])
        L = Line(cfg=cfg, d=machine.derive(cfg))
        boxes = {0: box(EDGE - 3., EDGE - 2.6)}
        it = L.onto_measuring_belt(boxes)
        L.measure(boxes, it, steady=False)
        self.assertEqual((it['void'], it['reasons']), (True, ['unsteady']))


def sound(r):
    """The numerical screen passed -- or all it found is a lump landing on the floor of a bin deeper than the
    screening limit for a moment: that lump has left the machine. (48 of the 54 batches over the limit in S6 are
    this; with the baseline of 2026-10-05 the failed-scan run is one: 5.2 mm for 1 ms.)"""
    n = r['numerics']
    return n['ok'] or (n['failure'] is None and not n['warnings'] and 'floor' in n['worst_contact']['geoms'])


def run(*args, duration=180):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = parse_config(['--duration', str(duration), '--no-video'] + list(args))
        cfg['out_dir'] = Path(tmp) / 'r'
        return simulate.run(cfg)


class Acceptance(unittest.TestCase):
    """2026-09-30, user: 故意让两块相贴、料搭在秤外、扫描失败，系统都不能把它们当成有效结果放走."""

    def check_held(self, r, reason):
        st = r['station']
        void = [it for it in st['items'] if it.get('void')]
        self.assertTrue(void, 'nothing was held')
        first = void[0]
        self.assertIn(reason, first['reasons'])
        self.assertNotIn('t_discharge_s', first)                        # never sent over the plate
        self.assertEqual(first['stage'], 'taken_off')
        self.assertTrue(set(first['truth_lumps']) <= set(first['taken_off']))
        self.assertEqual(st['counts']['void_discharged'], 0)
        self.assertEqual(st['verification']['false_valid'], [])         # nothing wrong got through as valid
        self.assertEqual(st['verification']['landed_unmeasured'], [])   # nothing reached a bin unmeasured
        self.assertIsNotNone(r['outcome']['line_clear_s'])              # the line ran on and cleared
        self.assertTrue(sound(r), r['numerics'])
        return first

    def test_two_lumps_laid_touching(self):
        r = run('--scenario', 'touching', '--seed', '4004')           # on the buffer belt: weighed together
        first = self.check_held(r, 'multi')
        self.assertEqual(sorted(first['truth_lumps']), sorted(r['scenario']['lumps']))

    def test_two_lumps_laid_touching_in_the_lane(self):
        r = run('--scenario', 'touching_lane', '--seed', '4004')      # the step parts them on the way
        st = r['station']
        self.assertEqual((st['counts']['void_discharged'], st['verification']['false_valid'],
                          st['verification']['landed_unmeasured']), (0, [], []))
        taken = {k for it in st['items'] if it['stage'] == 'taken_off' for k in it['taken_off']}
        for k in r['scenario']['lumps']:                              # measured alone and sorted right, or held and taken off
            ok = [it for it in st['items'] if not it.get('void') and it.get('truth_lumps') == [k]]
            self.assertTrue(k in taken or (ok and ok[-1]['route_as_ideal']), 'lump %d' % k)
        self.assertIsNotNone(r['outcome']['line_clear_s'])

    def test_a_lump_laid_across_the_joint_against_the_weighed_one(self):
        r = run('--scenario', 'off_scale_neighbour', '--seed', '392')
        first = self.check_held(r, 'outside')
        self.assertIn(r['scenario']['laid_against'], first['taken_off'])

    def test_the_weighed_lump_pushed_back_across_the_joint(self):
        r = run('--scenario', 'off_scale_bridge', '--seed', '392')
        self.check_held(r, 'outside')

    def test_the_first_scan_fails(self):
        r = run('--fault', 'scan_fail:0', '--seed', '392')
        first = self.check_held(r, 'scan')
        self.assertEqual(first['scan']['reasons'], ['scan_failed'])

    def test_seed_392_through_the_line(self):
        r = run('--seed', '392', duration=150)
        st = r['station']
        self.assertEqual((st['counts']['void'], st['counts']['landed'], st['counts']['route_not_as_ideal']), (0, 5, 0))
        self.assertEqual(st['verification'], dict(st['verification'], false_valid=[], false_void=[], landed_unmeasured=[]))
        self.assertLess(st['mass_error_pct_abs_max'], 1.)
        self.assertEqual(r['outcome']['funnel']['max_lumps'], 1)
        self.assertTrue(r['numerics']['ok'])
        self.assertLess(r['perception']['vision']['centroid_error_m']['p95'], .04)


if __name__ == '__main__':
    unittest.main(verbosity=2)
