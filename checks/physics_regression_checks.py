"""Physical regression probes. These verify mechanisms and diagnostics, not equipment performance."""
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator import lumps, simulate
from singulator.config import parse_config
from singulator.control.face import FaceRetract
from singulator.machine import assembly, derive, plough
from singulator.physics import drives
from singulator.physics.actuators import FaceServo
from singulator.physics.numerics import Diagnostics
from singulator.sim import layouts


class PhysicsChecks(unittest.TestCase):
    def cfg(self, *args):
        return parse_config(['--no-video', *args])

    def build(self, cfg, blocks=()):
        d = derive(cfg)
        m = mujoco.MjModel.from_xml_string(assembly.build_xml(cfg, d, blocks))
        data = mujoco.MjData(m)
        belts = drives.Conveyors(cfg, d, m)
        belts.command(data)
        mujoco.mj_forward(m, data)
        return d, m, data

    def test_layout_clears_actual_low_wall(self):
        cfg = self.cfg()
        d = derive(cfg)
        for seed in range(100):
            rng = np.random.default_rng(seed)
            b = lumps.make_blocks(cfg, rng)
            layout, _, _ = layouts.plan_scatter(cfg, b, rng, d)
            for k, x, y, z, yaw in layout:
                fp = layouts.footprint(b[k]['vertices'], yaw)
                self.assertGreaterEqual(y + fp[2], cfg['lane_y'] + .005 - 1e-12)

    def test_touching_layout_lays_nothing_into_anything(self):
        """Until 2026-10-06 the touching layout closed each lump onto the one ahead only: it could reach into the
        lump abreast (55 seeds in 7001-9000, up to 11.5 cm: 7134) or past the low-side skirt (7 seeds, 8.5 cm:
        7277), and the line refused those batches. Now they are laid clear; a layout that was clear is unchanged."""
        from singulator.sim.line import Line
        for seed, refit in ((7028, True), (7029, True), (7041, True), (7083, True), (7111, True), (7117, True),
                            (7134, True), (7277, True), (7101, False), (7106, False)):
            ln = Line(self.cfg('--layout', 'touching', '--seed', str(seed), '--count', str(3 + seed % 3)))
            self.assertEqual(ln.initial_pen, 0., seed)
            self.assertEqual('refit' in ln.layout, refit, seed)
            V = [L.world(ln.data) for L in ln.lumps]
            self.assertGreaterEqual(min(v[:, 1].min() for v in V), ln.cfg['lane_y'] + .005 - 1e-9, seed)

    def test_invalid_inputs_fail_before_build(self):
        for args in (('--dt', '0'), ('--dt', '0.0006'), ('--v-belt', 'nan'), ('--count', '0'),
                     ('--gangue-fraction', '1.1'), ('--size-long-max', '.4'), ('--unjam-max', '-1'),
                     ('--rolling-friction', '.001'), ('--solref', '.0001'), ('--face-swing-deg', '0')):
            with self.assertRaises(ValueError, msg=str(args)):
                self.cfg(*args)

    def test_brake_holds_only_within_capacity(self):
        m = drives.make_motor(100., 100., 1., .05)
        drives.brake_update(m, 90., .01)
        self.assertEqual(m['f'], 0.)
        drives.brake_update(m, 200., .01)
        self.assertAlmostEqual(m['f'], .01)
        self.assertAlmostEqual(m['brake_overload_s'], .01)
        m['f'] = 1.
        drives.brake_update(m, 0., .01)
        self.assertAlmostEqual(m['f'], .99)  # inertia prevents instantaneous stopping
        m['f'] = 0.
        drives.brake_update(m, -200., .01)
        self.assertAlmostEqual(m['f'], -.01)

    def test_sliding_cube_coulomb_impulse_and_reaction(self):
        vertices = np.array([[x,y,z] for x in (-.05,.05) for y in (-.05,.05) for z in (-.05,.05)])
        for dt in (.0005, .00025):
            cfg = self.cfg('--friction-belt', '.4', '--dt', str(dt))
            d, m, data = self.build(cfg, [dict(vertices=vertices, material='coal', density=1300.)])
            belt = drives.Conveyors(cfg, d, m)
            j = m.joint('bj0').id
            q, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            data.qpos[q:q+7] = [.1, .6, .05, 1., 0., 0., 0.]     # on the main belt, before the plough
            for _ in range(round(.1/dt)):
                belt.command(data)
                mujoco.mj_step(m, data)
            data.qvel[v] = 1.
            vz0 = data.qvel[v+2]
            impulse_x = impulse_z = reaction = 0.
            for _ in range(round(.1/dt)):
                belt.command(data)
                mujoco.mj_step(m, data)
                impulse_x += data.qfrc_constraint[v]*dt
                impulse_z += data.qfrc_constraint[v+2]*dt
                reaction += belt.main.load(data)*dt
            # An instantaneous horizontal velocity change excites soft normal contact and rocking:
            # N is NOT identically mg. Use measured normal impulse, including vertical momentum.
            self.assertAlmostEqual(impulse_x, -.4*impulse_z, delta=.001)
            self.assertAlmostEqual(impulse_x, -reaction, delta=.001)
            self.assertAlmostEqual(data.qvel[v], 1.-.4*(9.81*.1+data.qvel[v+2]-vz0), delta=.002)

    def test_face_is_dynamic_and_torque_limited(self):
        cfg = self.cfg()
        d, m, data = self.build(cfg)
        a, v = m.actuator(plough.FACE_ACTUATOR).id, int(m.jnt_dofadr[m.joint(plough.FACE_JOINT).id])
        f = FaceRetract(cfg, FaceServo(cfg, d, m, data))
        self.assertEqual(m.dof_armature[v], 0.)
        f.start(0.)
        before = data.qpos.copy()
        f.command(0.)
        np.testing.assert_array_equal(data.qpos, before)  # commands cannot move a mechanism
        limit = cfg['face_force_max']*f.lever
        # External torque above actuator capacity prevents retraction, against the home stop.
        data.qfrc_applied[v] = 2*limit
        for _ in range(round(6.1/cfg['dt'])):
            t = data.time
            f.command(t)
            mujoco.mj_step(m, data)
            f.record(t)
            self.assertLessEqual(abs(data.actuator_force[a]), limit + 1e-8)
            if f.fault:
                break
        self.assertEqual(f.fault['reason'], 'retract_not_reached')
        self.assertFalse(f.face.reached)

    def test_unloaded_face_cycle_reaches_home(self):
        cfg = self.cfg()
        d, m, data = self.build(cfg)
        f = FaceRetract(cfg, FaceServo(cfg, d, m, data))
        f.zone_busy = False
        f.start(0.)
        for _ in range(round(8/cfg['dt'])):
            t = data.time
            f.command(t)
            mujoco.mj_step(m, data)
            f.record(t)
            if f.phase in ('idle', 'fault'):
                break
        self.assertEqual(f.phase, 'idle')
        self.assertTrue(f.face.reached)
        self.assertLess(abs(f.theta), cfg['face_position_tol'])
        self.assertTrue(f.at_home)
        f.face.theta = -.05
        self.assertFalse(f.at_home)  # a physically deflected face must not release the next row

    def test_startup_penetration_and_warning_are_not_hidden(self):
        cfg = self.cfg()
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
            '<body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        diag = Diagnostics(m, cfg)
        diag.observe(data, 0.)
        self.assertAlmostEqual(diag.peak, .01)
        self.assertFalse(diag.report(data)['ok'])
        data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number = 1
        diag.observe(data, 0.)
        self.assertIsNotNone(diag.failure)

    def test_landing_is_recorded_apart_not_screened(self):
        """A lump landing on the floor (the bins), or on another lump that has left the line, is recorded under
        'landed' and does not fail the screen (2026-10-06; in S6 41 of 46 batches over the limit were only that).
        Anything else, a landed lump against equipment or a lump still on the line included, is screened."""
        cfg = self.cfg()
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
            '<geom name="frame" type="box" size=".1 .1 .1" pos="3 0 .1"/>'
            '<body pos="0 0 .09"><freejoint/><geom name="a" type="sphere" size=".1"/></body>'
            '<body pos="0 1 .1"><freejoint/><geom name="b" type="sphere" size=".1"/></body>'
            '<body pos="0 1.19 .1"><freejoint/><geom name="c" type="sphere" size=".1"/></body>'
            '</worldbody></mujoco>')
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        g = lambda n: m.geom(n).id
        diag = Diagnostics(m, cfg)
        diag.set_landed({g('a'), g('b'), g('c')})
        diag.observe(data, 0.)
        r = diag.report(data)
        self.assertAlmostEqual(r['landed']['max_penetration_m'], .01)    # a 1 cm into the floor ...
        self.assertEqual(r['max_penetration_m'], 0.)                     # ... b and c 1 cm into each other: landed too
        self.assertTrue(r['ok'])
        diag = Diagnostics(m, cfg)
        diag.set_landed({g('a'), g('b')})                                # c is still on the line
        diag.observe(data, 0.)
        r = diag.report(data)
        self.assertAlmostEqual(r['max_penetration_m'], .01)
        self.assertEqual(sorted(r['worst_contact']['geoms']), ['b', 'c'])
        self.assertFalse(r['ok'])
        data.qpos[m.jnt_qposadr[0]:m.jnt_qposadr[0] + 3] = (3., 0., .29)  # a landed lump 1 cm into equipment
        mujoco.mj_forward(m, data)
        diag = Diagnostics(m, cfg)
        diag.set_landed({g('a'), g('b'), g('c')})
        diag.observe(data, 0.)
        self.assertEqual(sorted(diag.report(data)['worst_contact']['geoms']), ['a', 'frame'])
        self.assertFalse(diag.report(data)['ok'])

    def test_contact_options_reach_compiled_contacts(self):
        cfg = self.cfg('--contact-condim', '6', '--torsional-friction', '.001', '--rolling-friction', '.0001')
        b = lumps.make_blocks(cfg, np.random.default_rng(3))
        d, m, data = self.build(cfg, b)
        q = m.jnt_qposadr[m.joint('bj0').id]
        data.qpos[q:q+3] = [.1, .6, -b[0]['vertices'][:, 2].min()-.001]     # on the main belt, before the plough
        mujoco.mj_forward(m, data)
        contacts = [c for c in data.contact[:data.ncon] if m.geom('bg0').id in c.geom]
        self.assertTrue(contacts)
        self.assertTrue(all(c.dim == 6 for c in contacts))
        self.assertTrue(all(np.allclose(c.friction[2:], [.001, .0001, .0001]) for c in contacts))

    def test_sample_times_and_startup_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg('--seed', '392', '--duration', '.2')
            cfg['out_dir'] = Path(tmp)
            result = simulate.run(cfg)
            with np.load(Path(tmp)/'trajectory.npz') as tr:
                np.testing.assert_allclose(tr['t'], np.arange(1, 21)*.01, atol=1e-10)
            self.assertGreater(result['drives']['feeder']['resist_peak_step_N'], 0.)
            self.assertFalse(result['provenance']['calibrated'])
            self.assertEqual(result['outcome']['face_drive_model'], 'dynamic')


if __name__ == '__main__':
    unittest.main(verbosity=2)
