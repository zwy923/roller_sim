"""Independent checks for the provisional standalone separator linkage."""
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from designs.flip_separator.model import Config, build, state, verify, material_clearance
from singulator.separator import for_width
from singulator.config import parse_config


class SeparatorChecks(unittest.TestCase):
    def test_line_plate_matches_the_measuring_belt(self):
        cfg = parse_config(['--no-video'])
        self.assertEqual(for_width(cfg['station_w']).width, cfg['station_w'])

    def test_linkage_and_direction(self):
        for length in (1.4, 1.6, 1.8):
            with self.subTest(length=length):
                c = Config(length=length)
                model = mujoco.MjModel.from_xml_string(build(c))
                report = verify(model, mujoco.MjData(model), c)
                self.assertLess(report['max_pin_closure_error_m'], 1e-6)
                down, up = state(c, c.closed_deg), state(c, c.open_deg)
                self.assertGreater(up['inlet'][0], down['inlet'][0])
                self.assertGreater(up['inlet'][2], down['inlet'][2])
                self.assertLess(up['outlet'][2], down['outlet'][2])

    def test_keyframes_hold_without_numerical_failure(self):
        c = Config()
        m = mujoco.MjModel.from_xml_string(build(c))
        d = mujoco.MjData(m)
        for key in range(m.nkey):
            mujoco.mj_resetDataKeyframe(m, d, key)
            mujoco.mj_forward(m, d)
            start = d.qpos.copy()
            for _ in range(500):
                mujoco.mj_step(m, d)
            self.assertTrue(np.isfinite(d.qpos).all())
            self.assertLess(float(np.max(np.abs(d.qpos-start))), .05)
            self.assertLess(float(np.linalg.norm(d.site('rod_pin').xpos-d.site('plate_pin').xpos)), .002)

    def test_invalid_dimensions_and_short_cylinder_rejected(self):
        for c in (Config(width=-1), Config(open_deg=10), Config(pivot_fraction=1.2)):
            with self.assertRaises(ValueError):
                build(c)
        c = Config(cylinder_stroke=.10)
        m = mujoco.MjModel.from_xml_string(build(c))
        with self.assertRaises(ValueError):
            verify(m, mujoco.MjData(m), c)

    def test_material_corridor_and_protruding_shaft_regression(self):
        c = Config()
        report = material_clearance(build(c), c)
        self.assertEqual(report['intersections'], [])
        # Reproduce a shaft too near the conveying face. The old no-load
        # same-body collision scan cannot detect a shaft protruding through its deck.
        bad = Config(deck_above_pivot=.019)
        with self.assertRaisesRegex(ValueError, 'Material path intersects'):
            material_clearance(build(bad), bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
