"""Checks of the shared singulator.audit on made-up results (no simulation): python experiments/analyze_checks.py

The cases are the ones designs/station/DESIGN.md asks of the whole-line acceptance: every lump good; abreast at the
main belt head but measured one by one; single file but void and taken off; everything refused (safe, not a success);
a lump in a bin without a measurement, a lump with two valid items, a wrong bin, a wrong route; a batch over the
numerical limit and one that ran out of time; and a missing field, which must never count as a pass.
"""
import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import analyze
from analyze import audit, void_cause
from singulator.audit import measurement_report, format_summary


def item(i, lumps, void=False, single=None, isolated=True, scan=True, route='coal', reasons=(), err=.1, steady=True,
         sent=True, **more):
    it = dict(item=i, truth_lumps=list(lumps), void=void, truth_single=len(lumps) == 1 if single is None else single,
              truth_isolated=isolated, reasons=list(reasons), mass_error_pct=err, steady=steady,
              scan=scan if isinstance(scan, dict) else dict(valid=scan, reasons=[], objects=1))
    if not void:
        it.update(route=route, **({'t_discharge_s': 1.} if sent else {}))
    it.update(more)
    return it


def result(blocks, items, cls='single_file', numerics_ok=True, taken=(), geoms=('bg1', 'bg2')):
    """blocks: [(state, bin, density)]."""
    return dict(blocks=[dict(family='blocky', state=s, bin=b, density=d) for s, b, d in blocks],
                config=dict(sort_density=1800., sensing='vision', lane_w=.6, weigh_model='fixed'),
                station=dict(items=items, counts=dict(holds=0),
                             verification=dict(false_valid=[], landed_unmeasured=[])),
                taken_off=[dict(lump=k, items=its) for k, its in taken],
                numerics=dict(ok=numerics_ok, failure=None, warnings={}, worst_contact=dict(geoms=list(geoms))),
                outcome=dict(classification=cls, line_clear_s=30., unjam_pulses=0), perception={}, provenance={})


COAL, GANGUE = ('sorted', 'coal', 1400.), ('sorted', 'gangue', 2400.)


class Audit(unittest.TestCase):
    def fates(self, r):
        L, b = audit(r)
        return [L[k]['fate'] for k in sorted(L)], b

    def test_every_lump_good(self):
        f, b = self.fates(result([COAL, GANGUE], [item(0, [0]), item(1, [1], route='gangue')]))
        self.assertEqual(f, ['measured', 'measured'])
        self.assertTrue(b['passed'] and b['all_measured'])
        self.assertFalse(b['manual'] or b['unfinished'] or b['verification_failed'] or b['numerics_bad'])

    def test_abreast_at_the_head_edge_but_measured_one_by_one(self):
        f, b = self.fates(result([COAL, COAL], [item(0, [0]), item(1, [1])], cls='abreast_at_cut'))
        self.assertEqual(f, ['measured', 'measured'])
        self.assertTrue(b['passed'])

    def test_single_file_but_void_and_taken_off(self):
        r = result([COAL, ('taken_off', None, 1400.), ('taken_off', None, 1400.)],
                   [item(0, [0]), item(1, [1, 2], void=True, reasons=['multi'])], taken=[(1, [1]), (2, [1])])
        f, b = self.fates(r)
        self.assertEqual(f, ['measured', 'multi', 'multi'])         # taken off stays in the count, never a success
        self.assertTrue(b['manual'])
        self.assertFalse(b['passed'] or b['all_measured'] or b['verification_failed'])

    def test_everything_refused_is_safe_but_not_a_success(self):
        r = result([('taken_off', None, 1400.)] * 2, [item(0, [0], void=True, isolated=False, reasons=['outside']),
                                                     item(1, [1], void=True, reasons=['lost'])],
                   taken=[(0, [0]), (1, [1])])
        f, b = self.fates(r)
        self.assertEqual(f, ['touching', 'misjudged'])
        self.assertFalse(b['verification_failed'] or b['passed'] or b['all_measured'])
        self.assertTrue(b['manual'])

    def test_causes_of_a_void_item(self):
        self.assertEqual(void_cause(item(0, [0, 1], void=True)), 'multi')
        self.assertEqual(void_cause(item(0, [0], void=True, isolated=False)), 'touching')
        off = dict(valid=False, reasons=['not_wholly_on_the_belt'])
        self.assertEqual(void_cause(item(0, [0], void=True, scan=off)), 'touching')
        self.assertEqual(void_cause(item(0, [0], void=True, scan=dict(valid=False, reasons=[], objects=2))), 'near')
        self.assertEqual(void_cause(item(0, [0], void=True, reasons=['outside'],
                                         reason_info=dict(outside=dict(detail='another object within 0.05 m')))),
                         'near')
        self.assertEqual(void_cause(item(0, [0], void=True, reasons=['unsteady'])), 'misjudged')

    def test_a_valid_item_that_was_not_one_lump_alone(self):
        f, b = self.fates(result([COAL, COAL], [item(0, [0, 1])]))
        self.assertEqual(f, ['false_valid', 'false_valid'])
        self.assertTrue(b['verification_failed'])
        f, b = self.fates(result([COAL], [item(0, [0], isolated=False)]))
        self.assertEqual(f, ['false_valid'])

    def test_a_missing_field_is_never_a_pass(self):
        for gap in (dict(truth_isolated=None), dict(truth_single=None), dict(scan={})):
            it = item(0, [0])
            it.update(gap)
            f, b = self.fates(result([COAL], [it]))
            self.assertEqual(f, ['undetermined'], gap)
            self.assertTrue(b['verification_failed'] and not b['passed'])
        it = item(0, [0])
        del it['truth_isolated']
        self.assertEqual(self.fates(result([COAL], [it]))[0], ['undetermined'])

    def test_a_lump_in_a_bin_without_a_measurement(self):
        f, b = self.fates(result([COAL, COAL], [item(0, [0])]))
        self.assertEqual(f, ['measured', 'unmeasured_landing'])
        self.assertTrue(b['verification_failed'] and not b['passed'])
        # ... or with a void one only (a void item discharged)
        f, b = self.fates(result([COAL], [item(0, [0], void=True, reasons=['scan'])]))
        self.assertEqual(f, ['unmeasured_landing'])

    def test_a_lump_with_two_valid_items_is_one_success_and_is_listed(self):
        L, b = audit(result([COAL], [item(0, [0]), item(1, [0])]))
        self.assertEqual((L[0]['fate'], L[0]['linked_items'], L[0]['valid_twice']), ('measured', [0, 1], True))
        self.assertTrue(b['verification_failed'] and not b['passed'])

    def test_wrong_route_and_wrong_bin(self):
        L, b = audit(result([GANGUE], [item(0, [0], route='coal')]))        # sent as coal, it is gangue
        self.assertTrue(L[0]['fate'] == 'measured' and L[0]['route_wrong'] and not L[0]['whole_line'])
        self.assertTrue(b['all_measured'] and b['verification_failed'] and not b['passed'])
        L, b = audit(result([('sorted', 'gangue', 1400.)], [item(0, [0], route='coal')]))    # right route, other bin
        self.assertTrue(L[0]['fate'] == 'measured' and L[0]['bin_wrong'] and not L[0]['whole_line'])
        self.assertTrue(b['verification_failed'] and not b['passed'])

    def test_the_reading_is_recorded_not_judged(self):
        L, b = audit(result([COAL, COAL], [item(0, [0], steady=False), item(1, [1], err=3.5)]))
        self.assertEqual([L[k]['fate'] for k in (0, 1)], ['measured', 'measured'])
        self.assertEqual([(L[k]['unsteady'], L[k]['mass_off']) for k in (0, 1)], [(True, False), (False, True)])
        self.assertTrue(b['passed'])

    def test_over_the_numerical_limit_and_out_of_time(self):
        f, b = self.fates(result([COAL], [item(0, [0])], numerics_ok=False, geoms=('floor', 'bg0')))
        self.assertEqual(f, ['measured'])
        self.assertTrue(b['numerics_bad'] and b['numerics_floor_only'] and not b['passed'])
        f, b = self.fates(result([COAL], [item(0, [0])], numerics_ok=False))
        self.assertTrue(b['numerics_bad'] and not b['numerics_floor_only'])
        f, b = self.fates(result([COAL, ('on_belt', None, 1400.)], [item(0, [0])], cls='incomplete'))
        self.assertEqual(f, ['measured', 'left'])
        self.assertTrue(b['unfinished'] and not b['passed'])
        # measured, the run ended before it was discharged: measured alone, the batch unfinished
        L, b = audit(result([('on_belt', None, 1400.)], [item(0, [0], sent=False)], cls='station_fault'))
        self.assertTrue(L[0]['fate'] == 'measured' and not L[0]['whole_line'] and b['unfinished'] and not b['passed'])

    def test_collateral_and_dropped(self):
        r = result([('taken_off', None, 1400.), ('taken_off', None, 1400.), ('dropped', None, 1400.)],
                   [item(0, [0], void=True, isolated=False, reasons=['outside'])], taken=[(0, [0]), (1, [0])])
        L, b = audit(r)
        self.assertEqual([L[k]['fate'] for k in (0, 1, 2)], ['touching', 'collateral', 'dropped'])
        self.assertEqual(L[1]['with_cause'], ['touching'])
        self.assertTrue(b['manual'] and not b['passed'])

    def test_revoked_measurement_is_not_a_success_even_if_void_flag_is_stale(self):
        r = result([COAL], [item(0, [0], revoked=dict(t_s=2., route='coal'))])
        L, b = audit(r)
        self.assertEqual(L[0]['fate'], 'unmeasured_landing')
        self.assertTrue(b['verification_failed'])
        self.assertFalse(b['all_measured'] or b['passed'])

    def test_json_report_keeps_removed_lumps_in_denominator_and_keeps_flags(self):
        r = result([COAL, ('taken_off', None, 1400.)],
                   [item(0, [0]), item(1, [1], void=True, isolated=False)], taken=[(1, [1])])
        report = json.loads(json.dumps(measurement_report(r)))
        self.assertEqual(report['counts'], dict(input=2, measured=1, success_rate=.5,
                                               not_measured=1, failure_causes=dict(touching=1)))
        self.assertEqual(report['batch']['status'], 'manual_required')
        self.assertEqual([f['lump'] for f in report['lumps']], [0, 1])
        self.assertEqual(report['later_checks']['whole_line'], 1)
        self.assertIn('independent measurement: 1/2 (50.0%)', format_summary(report))
        self.assertIn('taken off 1', format_summary(report))

    def test_report_separates_primary_measurement_from_reading_and_sorting(self):
        r = result([('sorted', 'gangue', 1400.)], [item(0, [0], steady=False, err=3.5)])
        report = measurement_report(r)
        self.assertEqual(report['counts']['success_rate'], 1.)
        self.assertEqual(report['later_checks'], dict(unsteady=1, mass_off=1, whole_line=0,
                                                      route_wrong=0, bin_wrong=1))
        self.assertEqual(report['batch']['status'], 'verification_failed')
        report = measurement_report(result([COAL], [item(0, [0])], numerics_ok=False))
        self.assertEqual(report['counts']['measured'], 1)
        self.assertEqual(report['batch']['status'], 'numerical_invalid')
        self.assertFalse(report['batch']['passed'])

    def test_batch_cli_reads_saved_result_and_reports_abnormal_end(self):
        r = result([COAL], [item(0, [0])], cls='station_fault')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'baseline'
            folder.mkdir()
            (folder / 'scatter_9001.json').write_text('{}', encoding='utf-8')
            with gzip.open(folder / 'scatter_9001.result.json.gz', 'wt', encoding='utf-8') as fh:
                json.dump(r, fh)
            output = Path(tmp) / 'report.json'
            stdout = io.StringIO()
            with patch('sys.argv', ['analyze.py', tmp, '--list', '--json', str(output)]), redirect_stdout(stdout):
                analyze.main()
            self.assertIn('station_fault', stdout.getvalue())
            self.assertIn('100.0', stdout.getvalue())
            saved = json.loads(output.read_text(encoding='utf-8'))['baseline']['scatter_9001']
            self.assertTrue(saved['unfinished'])
            self.assertFalse(saved['passed'])
            self.assertEqual(saved['lumps']['0']['fate'], 'measured')


if __name__ == '__main__':
    unittest.main(verbosity=2)
