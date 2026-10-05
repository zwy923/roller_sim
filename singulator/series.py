"""Summaries of a sampled series (pure numpy)."""
import numpy as np


def load_stats(loads, dt, limit=None):
    """Load summary from a signed per-step series F (negative = resisting the motion).

    net_*      50 ms moving average of F itself: resistance and assistance cancel, i.e. the net effect.
    resist_*   computed on max(-F, 0) BEFORE averaging, and assist_* on max(F, 0), so a load that
               alternates in sign shows up in both instead of averaging away (a +/-2000 N alternation
               has zero net average but a 1000 N average resistance and assistance).
    *_peak_step_N     largest single-step value -- a contact impulse, not a load a part holds.
    *_over_10pct_limit_s  time the one-sided load exceeds 10 % of the drive limit (any value > 0 without one).
    The 50 ms averages are model diagnostics, not design loads for a drive or a cylinder.
    """
    a = np.asarray(loads, float)
    keys = ('net_avg50ms_resist_max_N', 'net_avg50ms_assist_max_N', 'resist_avg50ms_max_N', 'resist_peak_step_N',
            'resist_over_10pct_limit_s', 'assist_avg50ms_max_N', 'assist_peak_step_N', 'assist_over_10pct_limit_s')
    if not len(a):
        return dict.fromkeys(keys, 0.)
    w = max(1, int(round(.05 / dt)))

    def avg(x):
        c = np.concatenate([[0.], np.cumsum(x)])
        return (c[w:] - c[:-w]) / w if len(x) >= w else np.array([x.mean()])
    thr = .1 * limit if limit else 0.
    net, r, s = avg(a), np.maximum(-a, 0.), np.maximum(a, 0.)
    return dict(zip(keys, (round(float(max(0., -net.min())), 1), round(float(max(0., net.max())), 1),
                           round(float(avg(r).max()), 1), round(float(r.max()), 1), round(float((r > thr).sum() * dt), 3),
                           round(float(avg(s).max()), 1), round(float(s.max()), 1), round(float((s > thr).sum() * dt), 3))))
