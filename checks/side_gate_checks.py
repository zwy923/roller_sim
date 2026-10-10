"""Side gate: swept clearance, paired initial states, the plan from the true lumps, how a hold ends, finite force."""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator.config import default_config
from singulator.control import feeder
from singulator.control.side_gate import SideGate
from singulator.machine import derive, side_gate
from singulator.machine.assembly import build_xml
from singulator.physics.drives import Conveyors
from singulator.physics.side_gate import GateDrive
from singulator.sim.line import Line


def obj(tid, x0, x1, y0, y1):
    return dict(tid=tid, x0=x0, x1=x1, y0=y0, y1=y1, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2)


def feedback(load=0.):
    return dict(position_m=[.4, 0.], speed_m_s=[0., 0.], force_N=[0., 0.], load_N=[load, 0.], buffer_m=[0., 0.])


class Checks(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config(side_pusher='active', video=False, feeder_sensing='oracle')
        self.d = derive(self.cfg)
        self.x1 = self.d['feeder']['x1']

    def test_swept_equipment_clearance(self):
        m = mujoco.MjModel.from_xml_string(build_xml(self.cfg, self.d, []))
        a = mujoco.MjData(m)
        Conveyors(self.cfg, self.d, m).command(a)
        v = GateDrive(self.cfg, side_gate.geometry(self.cfg, self.d), m)
        ids = list(v.geom)
        buf = np.zeros(6)
        for i in range(2):
            for q, b in ((q, b) for q in np.linspace(0, self.cfg['gate_stroke'], 31)
                         for b in (0., side_gate.BUFFER_STROKE / 2, side_gate.BUFFER_STROKE)):
                a.qpos[v.q] = 0
                a.qpos[v.q[i]] = q
                a.qpos[v.bq[i]] = b
                mujoco.mj_forward(m, a)
                for j in range(m.ngeom):
                    if j == ids[i] or not (m.geom_contype[j] or m.geom_conaffinity[j]):
                        continue
                    z = mujoco.mj_geomDistance(m, a, ids[i], j, .01, buf)
                    self.assertGreaterEqual(z, -1e-6, (i, q, m.geom(j).name, z))

    def test_identical_paired_initial_state(self):
        a, b = Line(self.cfg), Line(dict(self.cfg, side_pusher='parked'))
        self.assertEqual(a.xml, b.xml)
        np.testing.assert_array_equal(a.data.qpos, b.data.qpos)

    def test_plan_holds_side_by_side_neighbour(self):
        c, x1 = SideGate(self.cfg, self.d), self.x1
        lead, beside = obj(1, x1 - .45, x1 - .12, .62, .98), obj(2, x1 - .46, x1 - .13, .10, .48)
        p, why = c.plan({1: lead, 2: beside})
        self.assertEqual((p['free'], p['held'], p['side']), (1, [2], 0), why)
        beside = dict(beside, x1=x1 + .01)       # its front already over the edge: free it instead, hold the lead
        p, why = c.plan({1: lead, 2: beside})
        self.assertEqual((p['free'], p['held'], p['side']), (2, [1], 1), why)
        lead = dict(lead, x1=x1 + .01)           # both fronts over: the plate cannot go in
        self.assertEqual(c.plan({1: lead, 2: beside})[1], 'insertion_plane_occupied')
        far = obj(2, x1 - .80, x1 - .45, .10, .48)
        self.assertEqual(c.plan({1: obj(1, x1 - .45, x1 - .12, .62, .98), 2: far})[1], 'no_pair')

    def test_hold_ends_when_the_free_lump_went_or_on_overload(self):
        x1 = self.x1
        went = obj(1, x1 - .30, x1 + .04, .62, .98)
        went['cx'] = x1 + feeder.STOP_PAST + .001
        for view, load, end in (({1: went, 2: obj(2, x1 - .46, x1 - .13, .10, .48)}, 0., 'went'),
                                ({1: obj(1, x1 - .45, x1 - .12, .62, .98)}, 2600., 'load_trip')):
            c = SideGate(self.cfg, self.d)
            c.phase, c.blocking, c.current = 'hold', False, dict(side=0, free=1, held=[2])
            c.observe(1., view, feedback(load), .1)
            self.assertEqual((c.phase, c.blocking, c.current['push_end']), ('stop', True, end))

    def test_active_needs_true_lumps(self):
        with self.assertRaises(ValueError):
            default_config(side_pusher='active', feeder_sensing='vision')
        default_config(side_pusher='parked', feeder_sensing='vision')

    def test_finite_force_reference(self):
        m = mujoco.MjModel.from_xml_string(build_xml(self.cfg, self.d, []))
        a = mujoco.MjData(m)
        v = GateDrive(self.cfg, side_gate.geometry(self.cfg, self.d), m)
        v.command(a, [.8, 0.], .01)
        self.assertAlmostEqual(v.ref[0], .005)
        self.assertTrue(np.all(m.actuator_forcelimited[v.a]))


if __name__ == '__main__':
    unittest.main()
