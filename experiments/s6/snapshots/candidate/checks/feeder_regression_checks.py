"""Feeder controller regressions on synthetic camera views (no physics runtime required).

Run: python checks/feeder_regression_checks.py
FeederChecks (--feeder-release lane) / PredictiveRelease (the default): the oracle view (--sensing oracle: true
centroids and hulls). VisionRelease: the default path on camera objects (singulator/perception.py): the
centroid margin, the release judged by region because track ids churn at the head edge, objects the
cameras cannot vouch for holding the release, no release while the drop beam is cut, the report of which
lumps went, and blobs (lumps the cameras cannot tell apart) led by their front edge: staged and crept from it,
not held for their uncertain count, a lump carried over by staging recorded. TransferStops: the feed belt's stop
record (singulator/trial.py) on a synthetic belt.
These checks establish controller decisions, not whether the step-down transfer works in hardware.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator import feeder, machine, perception, trial  # noqa: E402
from singulator.config import parse_config  # noqa: E402
from singulator.feeder import CLEAR, Feeder  # noqa: E402

CFG = parse_config(['--feeder-release', 'lane', '--sensing', 'oracle', '--no-video'])
D = machine.derive(CFG)
X, H = D['feeder']['x1'], D['feeder']['step_m']
LANE_IN = D['lane_out_start']      # the next release waits for released rears to pass this
CFGP = parse_config(['--sensing', 'oracle', '--no-video'])   # predictive release
DP = machine.derive(CFGP)


def box(x0, x1, z0=H, z1=H + .3, y1=.6):
    return np.array([[x, y, z] for x in (x0, x1) for y in (.2, y1) for z in (z0, z1)])


def observe(f, t, boxes, states=None, home=True):
    """The oracle view: one object per lump on the belt (true centroid and hull extents), and the drop beam as a
    line test on the true hulls."""
    states = states or ['on_belt'] * len(boxes)
    b = f.g['beam']
    view = {k: dict(cx=float(V[:, 0].min() + V[:, 0].max()) / 2, x0=float(V[:, 0].min()), x1=float(V[:, 0].max()),
                    y1=float(V[:, 1].max())) for k, V in enumerate(boxes) if states[k] == 'on_belt'}
    f.observe(t, view, any(states[k] == 'on_belt' and perception.cuts_line(V, b['x'], b['z'])
                           for k, V in enumerate(boxes)), home)


class FeederChecks(unittest.TestCase):
    def test_overhanging_lump_waits_until_its_centroid_passes(self):
        f = Feeder(CFG, D, 1)
        self.assertTrue(f.waiting(0, 'on_belt', X - .01))      # front may be far over; centroid decides
        self.assertFalse(f.waiting(0, 'on_belt', X + .01))
        f.start(0.)
        observe(f, 0., [box(X - .30, X + .25)])                 # centroid 2.5 cm behind: not released
        self.assertEqual(f.released, [])
        observe(f, .1, [box(X - .20, X + .35)])
        self.assertEqual(f.released, [0])
        self.assertFalse(f.waiting(0, 'on_belt', X - .01))      # released lumps never count as waiting again

    def test_lump_released_during_a_jog_counts_as_after_stop(self):
        f = Feeder(CFG, D, 2)
        f.start(0.)
        hang, behind = box(X - .15, X + .25), box(X - .40, X + .05)
        for i in range(130):                                   # beam never cut: stop, then jog the hanging lump
            observe(f, i * .01, [hang, behind])
        self.assertEqual(f.jog, 0)
        observe(f, 1.31, [box(X - .10, X + .30), box(X - .19, X + .26)])   # the jog carries 1 over too
        self.assertEqual(f.releases[0]['after_stop'], [1])
        self.assertEqual(f.counts['after_stop'], 1)
        observe(f, 1.32, [box(X + CLEAR + .01, X + .45), box(X - .18, X + .27)])
        self.assertIsNone(f.jog)
        self.assertEqual(f.phase, 'stopped')

    def test_beam_ignores_lumps_still_on_the_feed_belt_and_next_release_needs_lane_entry_and_face(self):
        f = Feeder(CFG, D, 2)
        f.start(0.)
        observe(f, 0., [box(X - .30, X + .25), box(X - .9, X - .5)])
        self.assertEqual(f.phase, 'feeding')                    # the overhang passes above the beam
        observe(f, .1, [box(X - .15, X + .40, 0., .3), box(X - .9, X - .5)])
        self.assertEqual((f.phase, f.releases[0]['stop']), ('stopped', 'beam'))
        queued = box(X - .9, X - .5)
        observe(f, 5., [box(LANE_IN - .01, LANE_IN + .39, 0., .3), queued])   # rear still in the funnel mouth
        observe(f, 6., [box(LANE_IN + .01, LANE_IN + .41, 0., .3), queued], home=False)
        self.assertEqual(len(f.releases), 1)
        observe(f, 7., [box(LANE_IN + .01, LANE_IN + .41, 0., .3), queued], home=True)
        self.assertEqual(len(f.releases), 2)                    # in the lane, well short of the exit: go
        self.assertLess(LANE_IN + .41, D['exit_x'])
        self.assertEqual(f.releases[1]['t_start_s'], 7.)

    def test_station_hold_freezes_the_beam_miss_clock_and_progress(self):
        f = Feeder(CFG, D, 1)
        f.start(0.)
        hang = box(X - .15, X + .25)                            # went, beam not cut: the miss clock runs
        observe(f, 0., [hang])
        f.pause(.5)                                             # the station holds the section at 0.5 s ...
        self.assertTrue(f.paused)
        self.assertEqual(f.target, 0.)
        f.resume(3.5)                                           # ... for 3 s
        observe(f, 3.9, [hang])
        self.assertEqual(f.phase, 'feeding')                    # 0.9 s of running since it went: no miss yet
        observe(f, 4.1, [hang])
        self.assertEqual((f.phase, f.releases[0]['stop']), ('stopped', 'beam_missed'))
        self.assertEqual(f.held_s, 3.)
        f.resume(5.)                                            # not paused: nothing happens
        self.assertEqual(f.held_s, 3.)

    def test_feed_belt_empty_after_the_last_release(self):
        f = Feeder(CFG, D, 1)
        f.start(0.)
        observe(f, 0., [box(X - .10, X + .40, 0., .3)])
        observe(f, 1., [box(X + 1., X + 1.4)], states=['passed'])
        self.assertEqual(f.phase, 'empty')
        self.assertEqual(f.report()['counts']['releases'], 1)


class PredictiveRelease(unittest.TestCase):
    """--feeder-release predict (the default): staging, and the release by predicted funnel clearance."""

    def setUp(self):
        self.f, self.t, self.q = Feeder(CFGP, DP, 2), 0., X - .80    # q: queued lump 1's rear (0.40 long)
        self.rear = X - .15                                            # lump 0's rear on the main belt
        self.f.start(0.)
        self.see(box(X - .20, X + .35))                                # lump 0 goes over the edge ...
        self.see(box(X - .15, X + .40, 0., .3))                        # ... and cuts the beam: stop
        self.assertEqual((self.f.phase, self.f.released), ('stopped', [0]))

    def see(self, lump0, home=True):
        self.t = round(self.t + .01, 4)
        self.q += self.f.goal * DP['feeder']['speed_m_s'] * .01 if self.f.phase == 'stopped' else 0.
        observe(self.f, self.t, [lump0, box(self.q, self.q + .40)], home=home)

    def march(self, until, v=.40, y1=.9, home=True):
        """Lump 0 on along the main belt at v (face-bound with y1 = 0.9) until until(); returns its rear."""
        for _ in range(1000):
            self.see(box(self.rear, self.rear + .40, 0., .3, y1=y1), home)
            if until():
                return self.rear
            self.rear += v * .01
        self.fail('never')

    def t_arrive(self):
        c = self.q + .20                                              # lump 1's centroid, 0.20 behind its front
        d_c = X - c
        g = DP['feeder']
        t_tip = max(0., d_c - feeder.CREEP_ZONE) / g['speed_m_s'] + min(d_c, feeder.CREEP_ZONE) / g['creep_speed_m_s']
        return t_tip + feeder.TIP_S + (DP['P0'][0] - (X + .20)) / CFGP['v_belt']

    def test_staging_brings_the_next_lump_to_the_creep_zone(self):
        f = self.f
        self.march(lambda: f.staging)
        self.march(lambda: not f.staging)
        self.assertEqual((f.goal, len(f.releases), f.counts['stagings']), (0., 1, 1))
        self.assertAlmostEqual(X - (self.q + .20), feeder.CREEP_ZONE + feeder.STAGE_STOP, delta=.002)

    STAGED = X - .20 - feeder.CREEP_ZONE - feeder.STAGE_STOP          # lump 1's rear once staged

    def test_release_as_soon_as_the_face_bound_lump_before_will_be_out_in_time(self):
        f = self.f
        self.q = self.STAGED
        rear = self.march(lambda: len(f.releases) == 2)
        want = LANE_IN - feeder.V_FACE * (self.t_arrive() - feeder.RELEASE_MARGIN_S)
        self.assertAlmostEqual(rear, want, delta=.01)                  # predicted at V_FACE through the funnel
        self.assertTrue(DP['P0'][0] < rear < LANE_IN)                  # earlier than the lane-entry rule

    def test_a_lump_in_the_lane_band_lets_the_next_go_before_it_reaches_the_funnel(self):
        f = self.f
        self.q = self.STAGED
        rear = self.march(lambda: len(f.releases) == 2, y1=.6)
        want = DP['P0'][0] - CFGP['v_belt'] * (self.t_arrive() - feeder.RELEASE_MARGIN_S
                                                - (LANE_IN - DP['P0'][0]) / CFGP['v_belt'])
        self.assertAlmostEqual(rear, want, delta=.01)                  # predicted at its own speed, 0.40 m/s
        self.assertLess(rear, DP['P0'][0])

    def test_face_away_stalled_or_across_the_beam_holds_the_release(self):
        f = self.f
        self.q = self.STAGED
        self.march(lambda: self.rear > 1.2, home=False)               # face not home: no release
        self.assertEqual((len(f.releases), f.staging), (1, False))
        self.assertAlmostEqual(X - (self.q + .20), feeder.CREEP_ZONE + feeder.STAGE_STOP, delta=.002)
        self.rear = 1.0
        for _ in range(100):                                          # stalled in the funnel
            self.see(box(1.0, 1.4, 0., .3, y1=.9))
        self.assertEqual(len(f.releases), 1)
        g = Feeder(CFGP, DP, 2)
        g.start(0.)
        observe(g, 0., [box(X - .20, X + .35), box(X - .80, X - .40)])
        observe(g, .01, [box(X - .15, X + .40, 0., .3), box(X - .80, X - .40)])
        for i in range(60):                                           # moving, but still across the beam
            observe(g, .02 + i * .01, [box(X + .02 + i * 1e-3, X + .42 + i * 1e-3, 0., .3), box(X - .80, X - .40)])
        self.assertEqual(len(g.releases), 1)


CFGV = parse_config(['--no-video'])       # vision: the default
DV = machine.derive(CFGV)
CFGVL = parse_config(['--feeder-release', 'lane', '--no-video'])      # vision, the 'lane' release rule
DVL = machine.derive(CFGVL)


def vobj(x0, x1, conf=1., vx=None, y1=.6):
    return dict(cx=(x0 + x1) / 2, x0=x0, x1=x1, y1=y1, conf=conf, coasting=False, vx=vx)


def vblob(tid, x0, x1, solid=.80, n_est=2, conf=1., conf_seen=1.):
    """A blob: lumps lying against each other, one object to the cameras (its shape is not one convex lump's)."""
    return dict(vobj(x0, x1, conf=conf), tid=tid, solid=solid, n_est=n_est, conf_seen=conf_seen)


class VisionRelease(unittest.TestCase):
    def test_gone_only_past_the_centroid_margin(self):
        f = Feeder(CFGV, DV, 2)
        self.assertTrue(f.vision)
        f.start(0.)
        f.observe(0., {1: vobj(X - .17, X + .23)}, False, True)        # centroid 3 cm past: the camera's error
        self.assertNotIn('t_first_s', f.releases[0])
        f.observe(.01, {2: vobj(X - .14, X + .26)}, False, True)       # 6 cm past, under a new id: gone
        self.assertEqual(f.releases[0]['t_first_s'], .01)
        f.observe(.02, {3: vobj(X - .10, X + .30, vx=.2)}, True, True)  # the beam stops the feed belt
        self.assertEqual((f.phase, f.releases[0]['stop']), ('stopped', 'beam'))

    def test_a_lump_hanging_on_the_edge_under_new_ids_is_not_the_next_one(self):
        f = Feeder(CFGV, DV, 2)
        f.start(0.)
        f.observe(0., {1: vobj(X - .12, X + .30), 500: vobj(X - .55, X - .13)}, True, True)
        for i in range(1, 45):                                        # stuck across the edge, a new id every frame
            f.observe(i * .01, {10 + i: vobj(X - .12, X + .30, vx=0.), 1000 + i: vobj(X - .55, X - .13)}, False, True)
        self.assertEqual((len(f.releases), f.jog), (1, None))          # no release: something is across the edge
        for i in range(45, 70):
            f.observe(i * .01, {10 + i: vobj(X - .12, X + .30, vx=0.), 1000 + i: vobj(X - .55, X - .13)}, False, True)
        self.assertEqual((len(f.releases), f.jog, f.counts['jogs']), (1, 'across', 1))   # no progress: jog it

    def test_an_object_the_cameras_cannot_vouch_for_holds_the_release(self):
        f = Feeder(CFGV, DV, 2)
        f.start(0.)
        f.observe(0., {1: vobj(X - .12, X + .30)}, True, True)
        staged = vobj(X - .55, X - .066 - .016)                        # the next lump, staged
        far = vobj(LANE_IN + .1, LANE_IN + .5, vx=.4)                  # the lump before, in the lane
        for i, conf in enumerate((.3, .3, 1.)):
            unsure = dict(vobj(DV['P0'][0] - .1, DV['P0'][0] + .2, conf=conf, vx=.4))   # something in the funnel
            f.observe(1. + i * .01, {5: staged, 6: far, 7: unsure}, False, True)
            if conf < .6:
                self.assertEqual(len(f.releases), 1)
        self.assertGreaterEqual(f.counts['unsure_waits'], 2)

    def test_no_release_while_the_drop_beam_is_cut(self):
        """A lump tipping over the edge cuts the beam before the cameras call it gone (outline centroid not yet
        WENT_MARGIN past the edge): it is still queued, and the lead. Until 2026-10-04 a release started on it at
        once and the beam stopped it one sample later, over and over (seed 392: four empty releases in 0.07 s)."""
        for cfg, d in ((CFGV, DV), (CFGVL, DVL)):                     # predictive rule, lane rule
            with self.subTest(release=cfg['feeder_release']):
                f = Feeder(cfg, d, 2)
                f.start(0.)
                tipping = vobj(X - .22, X + .26)                      # centroid 2 cm past the edge: not gone yet
                f.observe(0., {1: tipping}, True, True)               # its nose cuts the beam: stop
                self.assertEqual((f.phase, f.releases[0]['stop']), ('stopped', 'beam'))
                for i in range(1, 8):                                 # still tipping across the beam
                    f.observe(i * .01, {1: tipping}, True, True)
                self.assertEqual((len(f.releases), f.phase, f.goal), (1, 'stopped', 0.))
                # the same view once the beam is clear again (it has settled back on the edge): the next release
                # starts on it, as before
                f.observe(.08, {1: tipping}, False, True)
                self.assertEqual((len(f.releases), f.phase), (2, 'feeding'))

    def test_the_report_names_the_lumps_that_went(self):
        """The cameras' objects carry no lump identity, so f.released (the oracle path's) stays empty; the report
        takes the lumps that went from the verification record. Until 2026-10-04 it listed every lump of a
        vision run as never released, and counted none as gone after the stop."""
        f = Feeder(CFGV, DV, 3)
        f.start(0.)
        f.audit(0., {0: X - .50, 1: X + .01, 2: X - .90})             # lump 1's true centroid is over the edge
        f.observe(.01, {7: vobj(X - .12, X + .30)}, True, True)       # the beam stops the release
        f.audit(.02, {0: X + .02, 1: X + .20, 2: X - .90})            # lump 0 follows after the stop
        r = f.report()
        self.assertEqual(f.released, [])
        self.assertEqual((r['released_order'], r['never_released']), ([1, 0], [2]))
        self.assertEqual((r['counts']['after_stop'], r['counts']['lumps_per_release']), (1, {'2': 1}))

    # ---- blobs: led by the front edge (2026-10-04) ----------------------------------------------------
    FAR = vobj(LANE_IN + .1, LANE_IN + .5, vx=.4)                      # the lump released before, in the lane

    def staged_blob(self, cfg):
        """The first lump has gone; a blob 0.70 m long is staged behind it (the face away: no release yet).
        Returns the feeder and where the blob's front edge stands when the staging stops."""
        f = Feeder(cfg, DV, 3)
        f.start(0.)
        f.observe(0., {1: vobj(X - .12, X + .30)}, True, True)         # the first lump tips and cuts the beam
        front, t = X - .30, 1.
        for _ in range(600):
            f.observe(t, {6: self.FAR, 9: vblob(9, front - .70, front)}, False, False)
            if f.counts['stagings'] and not f.staging:
                return f, front
            front += f.goal * DV['feeder']['speed_m_s'] * .01
            t = round(t + .01, 4)
        self.fail('staging never stopped')

    def test_a_blob_is_staged_by_its_front_edge(self):
        """The lump in front of a blob has its centroid well ahead of the blob's. Until 2026-10-04 the blob's own
        centroid was staged at the creep zone, and that lump went over the edge on the way (S4 batches: 11 of 19
        releases of two or more, up to 9.9 s after the release they were booked to)."""
        nose = feeder.NOSE_SHARE * CFGV['size_min']
        stop = X - feeder.CREEP_ZONE - feeder.STAGE_STOP
        f, front = self.staged_blob(CFGV)
        self.assertAlmostEqual(front - nose, stop, delta=.002)        # the furthest its leading centroid can be
        self.assertLess(front - .20, X)                               # a 0.40 m lump in front has not gone
        g, front_c = self.staged_blob(parse_config(['--blob-lead', 'centroid', '--no-video']))
        self.assertAlmostEqual(front_c - .35, stop, delta=.002)       # as before: the blob's centroid at the zone ...
        self.assertGreater(front_c - .20, X + feeder.WENT_MARGIN)     # ... and that lump is over the edge

    def test_a_release_creeps_a_blob_from_its_front_edge_to_the_beam(self):
        f, front = self.staged_blob(CFGV)
        v, see = DV['feeder']['speed_m_s'], lambda x: {6: self.FAR, 9: vblob(9, x - .70, x)}
        f.observe(10., see(front), False, True)                       # the face is home: the release starts
        self.assertEqual((len(f.releases), f.phase), (2, 'feeding'))
        for i in range(1, 40):
            front += f.goal * v * .01
            f.observe(10. + i * .01, see(front), False, True)
        self.assertEqual(f.goal, f.creep)                             # at creep, its centroid still 0.3 m short
        self.assertLess(front - .35, X - feeder.CREEP_ZONE - .2)
        f.observe(10.5, see(front), True, True)                       # the lump in front tips: the beam stops it
        self.assertEqual((f.phase, f.releases[1]['stop']), ('stopped', 'beam'))

    def test_an_outline_that_is_not_one_convex_lump_is_a_blob(self):
        f = Feeder(CFGV, DV, 2)
        f.start(0.)
        def see(t, solid, n_est=1, tid=4):
            f.observe(t, {tid: vblob(tid, X - .90, X - .30, solid, n_est)}, False, True)
        see(0., .93)
        self.assertEqual(f.blobs, set())                              # one frame: noise
        see(.01, .93)
        self.assertEqual(f.blobs, {4})                                # two running: a blob
        for i in range(feeder.UNBLOB_FRAMES - 1):                     # one convex lump again, not yet for long enough
            see(.02 + i * .01, .99)
        self.assertEqual(f.blobs, {4})
        see(.5, .99)
        self.assertEqual(f.blobs, set())
        see(.51, .99, n_est=2, tid=5)                                 # counted as two: a blob at once
        self.assertEqual((f.blobs, set(f.blob_run)), ({5}, {5}))      # and the track that ended is forgotten

    def test_a_blob_of_uncertain_count_does_not_hold_the_release(self):
        """Its shape says neither one lump nor two (confidence halved), but it is seen well (conf_seen). Led by
        its front edge it needs no count. Led by its centroid it held the release for as long as it looked so."""
        for lead, back, releases in (('front', feeder.NOSE_SHARE * CFGV['size_min'], 2), ('centroid', .35, 1)):
            with self.subTest(blob_lead=lead):
                f = Feeder(parse_config(['--blob-lead', lead, '--no-video']), DV, 3)
                f.start(0.)
                f.observe(0., {1: vobj(X - .12, X + .30)}, True, True)
                front = X - feeder.CREEP_ZONE - feeder.STAGE_STOP + back      # staged, by either rule
                blob = vblob(9, front - .70, front, solid=.93, n_est=1, conf=.5)
                for i in range(1, 6):
                    f.observe(1. + i * .01, {6: self.FAR, 9: blob}, False, True)
                self.assertEqual(len(f.releases), releases)

    def test_a_lump_carried_over_by_staging_is_recorded(self):
        f = Feeder(CFGV, DV, 3)
        f.start(0.)
        f.audit(0., {0: X + .01, 1: X - .40, 2: X - .90})             # lump 0 goes in the release
        f.observe(.01, {7: vobj(X - .12, X + .30)}, True, True)       # the beam stops it
        f.audit(.02, {1: X + .01, 2: X - .90})                        # lump 1 follows within the stopping distance
        f.observe(1., {8: vobj(X + .5, X + .9, vx=.4), 9: vobj(X - .90, X - .50)}, False, False)
        self.assertTrue(f.staging)
        f.audit(1.5, {2: X + .01})                                    # lump 2 goes while the belt is staging
        rel = f.releases[0]
        self.assertEqual((rel['lumps'], rel['lumps_after_stop'], rel['lumps_in_staging']), ([0, 1, 2], [1, 2], [2]))
        self.assertEqual(f.report()['counts']['in_staging'], 1)
        f.start(2.)
        self.assertFalse(f.staged)                                    # the next release starts a new count


class TransferStops(unittest.TestCase):
    """trial.TransferWatch's record of the feed belt's stops, on a synthetic belt (no lumps, no physics)."""

    def setUp(self):
        self.f = Feeder(CFGV, DV, 0)
        self.w = trial.TransferWatch(CFGV, DV, SimpleNamespace(ngeom=0), [])
        self.data, self.belts = SimpleNamespace(ncon=0), SimpleNamespace(feed=dict(f=.3))
        self.f.start(0.)
        self.f._stop(1., 'beam')                                      # the stop command, the belt at creep

    def see(self, t, speed):
        self.belts.feed['f'] = speed
        self.w.observe(t, self.data, self.f, self.belts)

    def test_a_stop_that_reaches_rest_is_measured(self):
        for i, speed in enumerate((.3, .2, .1, 0.)):
            self.see(1. + i * .01, speed)
        r = self.w.report(self.f)
        v = DV['feeder']['speed_m_s']
        self.assertEqual(r['stops_interrupted'], [])
        self.assertEqual(len(r['stops']), 1)
        s = r['stops'][0]
        self.assertEqual((s['release'], s['why'], s['stop_time_s'], s['lump_slide_m']), (0, 'beam', .03, 0.))
        self.assertAlmostEqual(s['travel_m'], .6 * v * .01, places=4)
        self.assertEqual((r['summary']['stop_travel_max_m'], r['summary']['stops_interrupted']), (s['travel_m'], 0))

    def test_a_stop_cut_short_has_no_stop_distance(self):
        """Staging starts 10 ms after the stop command, before the belt is at rest, and runs it for 3 s. Until
        2026-10-04 the stop record stayed open and booked the whole run as the stop (seed 392: 0.756 m, 8.16 s)."""
        f = self.f
        self.see(1., .3)                                              # the record opens
        f.staging, f.goal = True, 1.
        for i in range(1, 300):
            self.see(1. + i * .01, 1.)
        f.staging, f.goal = False, 0.
        self.see(4., 0.)
        r = self.w.report(f)
        self.assertEqual(r['stops'], [])
        self.assertEqual([(s['release'], s['interrupted'], s['stop_time_s']) for s in r['stops_interrupted']],
                         [(0, 'staging', .01)])
        self.assertLess(r['stops_interrupted'][0]['travel_m'], .001)
        self.assertEqual((r['summary']['stop_travel_max_m'], r['summary']['lump_slide_max_m'],
                          r['summary']['stops_interrupted']), (None, None, 1))
        self.assertTrue(all(s['done'] for s in self.w.stops))         # the bench run's end condition still holds


if __name__ == '__main__':
    unittest.main()
