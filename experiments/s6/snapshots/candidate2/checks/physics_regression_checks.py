"""Physical regression probes. These verify mechanisms and diagnostics, not equipment performance."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator import assembly, drives, face, lumps, machine, physics, simulate
from singulator.config import parse_config


class PhysicsChecks(unittest.TestCase):
    def cfg(self, *args):
        return parse_config(['--no-video', *args])

    def build(self, cfg, blocks=()):
        d = machine.derive(cfg)
        m = mujoco.MjModel.from_xml_string(assembly.build_xml(cfg, d, blocks))
        data = mujoco.MjData(m)
        belts = drives.Conveyors(cfg, d, m)
        belts.command(data)
        mujoco.mj_forward(m, data)
        return d, m, data

    def test_layout_clears_actual_low_wall(self):
        for mode in ('scatter', 'rows'):
            cfg = self.cfg('--layout', mode)
            d = machine.derive(cfg)
            for seed in range(100):
                rng = np.random.default_rng(seed)
                b = lumps.make_blocks(cfg, rng)
                layout, _, _ = getattr(lumps, 'plan_' + mode)(cfg, b, rng, d)
                for k, x, y, z, yaw in layout:
                    fp = lumps.footprint(b[k]['vertices'], yaw)
                    self.assertGreaterEqual(y + fp[2], cfg['lane_y'] + .005 - 1e-12)

    def test_invalid_inputs_fail_before_build(self):
        for args in (('--dt', '0'), ('--dt', '0.0006'), ('--v-belt', 'nan'), ('--count', '0'),
                     ('--gangue-fraction', '1.1'), ('--size-long-max', '.4'),
                     ('--rolling-friction', '.001'), ('--solref', '.0001')):
            with self.assertRaises(ValueError, msg=str(args)):
                self.cfg(*args)

    def test_brake_holds_only_within_capacity(self):
        m = drives.make_motor(True, 100., 100., 1., .05)
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
            cfg = self.cfg('--no-unjam',
                           '--friction-belt', '.4', '--dt', str(dt))
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
                reaction += data.qfrc_constraint[belt.main_dof]*dt
            # An instantaneous horizontal velocity change excites soft normal contact and rocking:
            # N is NOT identically mg. Use measured normal impulse, including vertical momentum.
            self.assertAlmostEqual(impulse_x, -.4*impulse_z, delta=.001)
            self.assertAlmostEqual(impulse_x, -reaction, delta=.001)
            self.assertAlmostEqual(data.qvel[v], 1.-.4*(9.81*.1+data.qvel[v+2]-vz0), delta=.002)

    def test_face_is_dynamic_and_torque_limited(self):
        cfg = self.cfg()
        d, m, data = self.build(cfg)
        j, a = m.joint('facej').id, m.actuator('face_drive').id
        q, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        f = face.FaceRetract(cfg, d, cfg['dt'], (q, v), actuator=a)
        self.assertEqual(m.dof_armature[v], 0.)
        f.start(0.)
        before = data.qpos.copy()
        f.command(0., data)
        np.testing.assert_array_equal(data.qpos, before)  # commands cannot move a mechanism
        limit = cfg['face_force_max']*f.lever
        # External torque above actuator capacity prevents retraction, against the home stop.
        data.qfrc_applied[v] = 2*limit
        for _ in range(round(6.1/cfg['dt'])):
            t = data.time
            f.command(t, data)
            mujoco.mj_step(m, data)
            f.record(t, data)
            self.assertLessEqual(abs(data.actuator_force[a]), limit + 1e-8)
            if f.fault:
                break
        self.assertEqual(f.fault['reason'], 'retract_not_reached')
        self.assertFalse(f.face.reached)

    def test_unloaded_face_cycle_reaches_home(self):
        cfg = self.cfg()
        d, m, data = self.build(cfg)
        j = m.joint('facej').id
        f = face.FaceRetract(cfg, d, cfg['dt'], (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])),
                             actuator=m.actuator('face_drive').id)
        f.zone_busy = False
        f.start(0.)
        for _ in range(round(8/cfg['dt'])):
            t = data.time
            f.command(t, data)
            mujoco.mj_step(m, data)
            f.record(t, data)
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
        diag = physics.Diagnostics(m, cfg)
        diag.observe(data, 0.)
        self.assertAlmostEqual(diag.peak, .01)
        self.assertFalse(diag.report(data)['ok'])
        data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number = 1
        diag.observe(data, 0.)
        self.assertIsNotNone(diag.failure)

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
