"""Material bookkeeping, compiled mass and scope regression; not feed calibration."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator import assembly, lumps, machine, simulate
from singulator.config import parse_config


class MaterialChecks(unittest.TestCase):
    def cfg(self, *args):
        return parse_config(['--no-video', *args])

    def test_component_mass_and_volume_match_engine(self):
        cfg = self.cfg('--count', '30')
        blocks = lumps.make_blocks(cfg, np.random.default_rng(392))
        model = mujoco.MjModel.from_xml_string(assembly.build_xml(cfg, machine.derive(cfg), blocks))
        self.assertEqual({b['material'] for b in blocks}, {'coal', 'middlings', 'gangue'})
        for k, b in enumerate(blocks):
            c = b['composition']
            self.assertAlmostEqual(sum(c['mass_fractions'].values()), 1.)
            self.assertAlmostEqual(sum(c['volume_fractions'].values()), 1.)
            self.assertAlmostEqual(sum(c['component_mass_kg'].values()), b['mass_kg'])
            # Independent additive component volumes must reconstruct the external hull.
            volume = sum(c['component_mass_kg'][n] / c['endmember_density_kg_m3'][n] for n in ('coal', 'gangue'))
            self.assertAlmostEqual(volume, ConvexHull(b['vertices']).volume, places=12)
            actual = model.body_mass[model.body('b%d' % k).id]
            self.assertAlmostEqual(actual / b['mass_kg'], 1., places=6)
            self.assertAlmostEqual(b['weight_N'], actual * 9.81, places=3)
            self.assertTrue(np.all(model.body_inertia[model.body('b%d' % k).id] > 0))

    def test_geometry_layout_and_repeatability(self):
        new = self.cfg()
        old = dict(new, material_model='legacy_binary')
        for seed in range(10):
            r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
            b1, b2 = lumps.make_blocks(new, r1), lumps.make_blocks(old, r2)
            for a, b in zip(b1, b2):
                np.testing.assert_array_equal(a['vertices'], b['vertices'])
            l1 = lumps.plan_scatter(new, b1, r1, machine.derive(new))
            l2 = lumps.plan_scatter(old, b2, r2, machine.derive(old))
            self.assertEqual(l1, l2)
            again = lumps.make_blocks(new, np.random.default_rng(seed))
            self.assertEqual([b['composition'] for b in b1], [b['composition'] for b in again])

    def test_category_extremes_and_invalid_settings(self):
        for g, mid, expected in ((1., 0., 'gangue'), (0., 1., 'middlings'), (0., 0., 'coal')):
            cfg = self.cfg('--gangue-fraction', str(g), '--middlings-fraction', str(mid))
            self.assertEqual({b['material'] for b in lumps.make_blocks(cfg, np.random.default_rng(7))}, {expected})
        lumps.make_blocks(self.cfg('--gangue-fraction', '.8', '--middlings-fraction', '.2'), np.random.default_rng(1))
        for args in (('--middlings-fraction', '-.1'), ('--middlings-fraction', '.8'),
                     ('--coal-density-min', '0'), ('--coal-density-min', '1500'),
                     ('--gangue-density-max', 'nan')):
            with self.assertRaises(ValueError, msg=str(args)):
                self.cfg(*args)

    def test_batch_fractions_are_mass_weighted(self):
        blocks = [dict(material='coal', mass_kg=10., volume_m3=.01,
                       composition=dict(component_mass_kg=dict(coal=9., gangue=1.))),
                  dict(material='gangue', mass_kg=30., volume_m3=.012,
                       composition=dict(component_mass_kg=dict(coal=3., gangue=27.)))]
        summary = lumps.batch_material_summary(blocks)
        self.assertAlmostEqual(summary['mass_fractions']['gangue'], .7)
        self.assertEqual(summary['category_count_fractions']['gangue'], .5)

    def test_result_serialization_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.cfg('--duration', '.02', '--seed', '392')
            cfg['out_dir'] = Path(folder)
            result = simulate.run(cfg)
            saved = json.loads((Path(folder) / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['feed_material'], result['feed_material'])
            self.assertTrue(saved['numerics']['ok'])
            for b in saved['blocks']:
                self.assertAlmostEqual(sum(b['composition']['component_mass_kg'].values()), b['mass_kg'])
            self.assertAlmostEqual(sum(b['mass_kg'] for b in saved['blocks']), saved['feed_material']['total_mass_kg'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
