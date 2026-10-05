"""Replay saved runs at the feed head: when would another stop rule have stopped the feed belt, and would the second
lump of a release have gone all the same? Pure post-processing of model.xml + trajectory.npz (s7_run.py --traj).

    python experiments/s8/s8_beam.py RUN_DIR [RUN_DIR ...] [--beam X,Z ...] [--stop-past M ...]

RUN_DIR holds run folders (one per batch: result.json, model.xml, trajectory.npz), at any depth.
  --beam X,Z       a drop beam X m past the head edge, Z m above the main belt (the line: 0.10,0.05);
  --stop-past M    stop once the release's first centroid is M m past the head edge (no beam).

Up to the moment a rule would have stopped the belt, the run under it is the run that was saved: nothing the
controller did differs before then. So the replay is exact up to that stop, and an estimate after it:
  * every release that put more than one lump over the edge is looked at from its first lump's crossing (t1);
  * 'needed' is the feed belt travel between t1 and the second lump's crossing (t2) in the saved run; 'given' is the
    travel the rule leaves after t1 -- up to its stop command, plus the ramp-down from creep;
  * the second lump was PUSHED if its centroid advanced about as far as the belt did from t1 to t2 (within
    DRAG_MARGIN): it rode the belt. Pushed and needed > given: the rule would have kept it back (avoided). Pushed and
    needed <= given: it goes all the same (kept). Not pushed -- it advanced further than the belt, so the lump in
    front pulled it -- is listed apart (dragged): with the belt stopped earlier the first lump also tips differently,
    and whether it still pulls the second one along is what only a run can say;
  * 'early': a beam cut before any centroid of the release was over the edge (a lump sagging over it while still on
    the feed belt). The line would stop there with nothing released: each one is a stop the rule adds.
The default rule is the line's own beam; its stop is checked against the one the run recorded.
"""
import argparse
import glob
import json
import os
from collections import Counter

import mujoco
import numpy as np

DEBOUNCE_S = .02         # sensing.beams.DEBOUNCE_S
SAMPLE_S = .01
DRAG_MARGIN = .005       # m: the second lump advanced more than the belt by this much from t1 to t2: pulled along


def quat_mat(q):
    """(n, 4) w x y z -> (n, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def cuts(V, x, z):
    """Does the convex hull V (m, 3) meet the beam line at (x, z)? (sensing.beams.cuts_line without scipy: the
    x-z projection of a convex hull contains (x, z) iff no edge direction of the projection separates it -- tested on
    the projected hull built by monotone chain.)"""
    P = V[:, [0, 2]]
    if not (P[:, 0].min() < x < P[:, 0].max() and P[:, 1].min() < z < P[:, 1].max()):
        return False
    pts = sorted(set(map(tuple, np.round(P, 9))))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    h = lo[:-1] + hi[:-1]
    return all(cross(h[i], h[(i + 1) % len(h)], (x, z)) >= 0 for i in range(len(h)))


class Run:
    def __init__(self, folder):
        self.name = os.path.basename(folder.rstrip('/'))
        self.r = json.load(open(os.path.join(folder, 'result.json'), encoding='utf-8'))
        tr = np.load(os.path.join(folder, 'trajectory.npz'))
        m = mujoco.MjModel.from_xml_path(os.path.join(folder, 'model.xml'))
        self.t, self.com = tr['t'], tr['com'].astype(float)
        names = list(tr['drive_names'])
        fd = self.r['feeder']['geometry']
        self.x1, self.beam = fd['x1'], fd['beam']
        self.v, self.vc, self.ramp = fd['speed_m_s'], fd['creep_speed_m_s'], self.r['config']['feeder_ramp_s']
        f = tr['drive_f'][:, names.index('feeder')].astype(float)
        self.travel = np.concatenate([[0.], np.cumsum(f[1:] * self.v * np.diff(self.t))])
        self.pos, self.quat = tr['pos'].astype(float), tr['quat'].astype(float)
        self.geoms = []
        for k in range(self.com.shape[1]):
            g = m.geom('bg%d' % k)
            mesh = m.geom_dataid[g.id]
            a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            gq = quat_mat(m.geom_quat[g.id][None])[0]
            self.geoms.append(m.geom_pos[g.id] + m.mesh_vert[a:a + n] @ gq.T)    # body-frame vertices
        self.tip = {L['lump']: L['t_tip'] for L in self.r['transfer']['lumps']}
        self._V = {}

    def i(self, t):
        return int(np.searchsorted(self.t, t - 1e-9))

    def V(self, k, i):
        key = (k, i)
        if key not in self._V:
            R = quat_mat(self.quat[i, k][None])[0]
            self._V[key] = self.pos[i, k] + self.geoms[k] @ R.T
        return self._V[key]

    def first_cut(self, ks, x, z, i0, i1):
        """First frame in i0..i1 at which a lump of ks cuts the beam (x, z), or None."""
        for i in range(i0, min(i1, len(self.t))):
            if any(cuts(self.V(k, i), x, z) for k in ks):
                return i
        return None


def releases(run):
    """The saved run's releases with something over the edge: (release, members by crossing time)."""
    for rel in run.r['feeder']['releases']:
        ms = [k for k in rel['members'] if run.tip.get(k) is not None]
        if ms:
            yield rel, sorted(ms, key=lambda k: run.tip[k])


def stop_time(run, rel, ms, rule):
    """When the rule commands the stop, and whether a beam was cut before the first centroid crossed."""
    t1 = run.tip[ms[0]]
    i0, i1 = run.i(rel['t_start_s']), run.i(t1 + 6.)
    kind, a, b = rule
    if kind == 'past':
        k0 = ms[0]
        for i in range(run.i(t1), i1):
            if run.com[i, k0, 0] > run.x1 + a:
                return run.t[i] + SAMPLE_S, False
        return None, False
    # lumps that can cut it: this release's and every lump not yet over the edge at the release's start
    ks = [k for k in range(run.com.shape[1]) if run.tip.get(k) is None or run.tip[k] >= rel['t_start_s'] - 1e-9]
    i = run.first_cut(ks, run.x1 + a, b, i0, i1)
    if i is None:
        return None, False
    return run.t[i] + DEBOUNCE_S + SAMPLE_S, run.t[i] < t1 - 1e-9


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dirs', nargs='+')
    p.add_argument('--beam', action='append', default=[], metavar='X,Z')
    p.add_argument('--stop-past', action='append', default=[], type=float, metavar='M')
    p.add_argument('--json', help='write every release of every rule here')
    a = p.parse_args()
    folders = sorted({os.path.dirname(f) for d in a.dirs for f in glob.glob(os.path.join(d, '**', 'trajectory.npz'),
                                                                            recursive=True)})
    rules = [('beam', .10, .05)] + [('beam',) + tuple(float(v) for v in s.split(',')) for s in a.beam] \
        + [('past', m, None) for m in a.stop_past]
    label = lambda r: 'beam %.3f m past, %.3f m up' % r[1:] if r[0] == 'beam' else 'stop %.3f m past' % r[1]
    out = {label(r): [] for r in rules}
    check = Counter()
    for folder in folders:
        run = Run(folder)
        for rel, ms in releases(run):
            t1 = run.tip[ms[0]]
            for rule in rules:
                ts, early = stop_time(run, rel, ms, rule)
                row = dict(run=run.name, release=rel['release'], n=len(ms), early=bool(early),
                           t_stop=None if ts is None else round(float(ts), 3), t1=t1,
                           saved_stop=rel['t_stop_s'], saved_why=rel['stop'])
                if rule == rules[0] and rel['stop'] == 'beam' and ts is not None:
                    check['beam'] += 1
                    check['same'] += abs(ts - rel['t_stop_s']) <= .015
                if len(ms) > 1 and ts is not None:
                    t2 = run.tip[ms[1]]
                    i1, i2 = run.i(t1), run.i(t2)
                    needed = run.travel[i2] - run.travel[i1]
                    adv = run.com[i2, ms[1], 0] - run.com[i1, ms[1], 0]
                    given = run.travel[min(run.i(ts), len(run.t) - 1)] - run.travel[i1] \
                        + run.vc * run.ramp * (run.vc / run.v) / 2
                    pushed = adv - needed <= DRAG_MARGIN
                    row.update(t2=t2, gap_s=round(t2 - t1, 3), needed=round(float(needed), 4),
                               given=round(float(given), 4), advance=round(float(adv), 4), fate='kept' if pushed and needed <= given else
                               'avoided' if pushed else 'dragged')
                out[label(rule)].append(row)
    print('%d runs. The line\'s own beam replayed against the recorded stops: %d of %d beam stops within 15 ms'
          % (len(folders), check['same'], check['beam']))
    print('%-34s %9s %6s %8s %6s %8s %7s %12s' % ('rule', 'releases', 'multi', 'avoided', 'kept', 'dragged',
                                                  'early', 'stop earlier'))
    for name, rows in out.items():
        multi = [x for x in rows if x['n'] > 1]
        fate = Counter(x.get('fate') for x in multi)
        base = {(x['run'], x['release']): x['t_stop'] for x in out[label(rules[0])]}
        dt = sorted(base[(x['run'], x['release'])] - x['t_stop'] for x in rows
                    if x['t_stop'] is not None and base.get((x['run'], x['release'])) is not None)
        print('%-34s %9d %6d %8d %6d %8d %7d %12s' % (
            name, len(rows), len(multi), fate['avoided'], fate['kept'], fate['dragged'],
            sum(x['early'] for x in rows), 'median %.2f s' % dt[len(dt) // 2] if dt else '-'))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(out, fh)


if __name__ == '__main__':
    main()
