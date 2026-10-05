"""The overhead cameras as the controllers see them: tracked objects, 50 ms late.

Until 2026-09-30 every controller read the true state: true centroids and hulls at 100 Hz. Since then (user:
把控制依据换成现实能获得的信号) the cameras are modelled. Per 10 ms frame, per tracked OBJECT the controllers get:
an estimated plan outline (convex polygon), its extents, the outline's area centroid (not the centre of mass: a
camera cannot see that), the top height, the velocity from the track, a recognition confidence (conf; conf_seen
is the same without the shape term: how surely the object is seen at all, whatever it holds), the estimated number
of lumps in it, a track id and its lineage (the ids it was merged or split from).

Frames reach the controllers machine.sensors.LATENCY_S late. Lumps closer than SEG_GAP form one blob -- the cameras
cannot separate them; a blob counts as more than one lump when a merge of two tracks made it or when its outline
is clearly not convex. A merge adds up the counts of the tracks the cameras were still following; a ghost -- a
track no blob has matched for GHOST_S, left behind when the blob it followed came apart -- adds nothing
(2026-10-04). A blob that came apart keeps its lineage in its pieces, also when neither piece lies near its
centroid (SPLIT_PAD, 2026-10-04). A track that vanishes where lumps do not leave the line is LOST: the vision is
then not healthy, and the controllers stop until every lost object is found again where it was lost.

The true state enters here only to form the measurements (the scene the cameras look at). Every object keeps the
lumps it was formed from in '_truth' for the verification written to result.json; no controller reads a key that
starts with '_'. With oracle=True (--sensing oracle, and the feed belt's view with --feeder-sensing oracle) the
same interface delivers the ideal view: one object per lump, its true centroid and hull, no latency.

Every number below is a placeholder for a prototype, not a measured property of a camera. The noise is white per
frame, equipment does not occlude, and the lumps are convex -- real coal is not, which flatters the blob-shape test.
"""
import math
from collections import deque

import numpy as np

from ..geom2d import box_gap, clip, hull2, inside, poly_area, poly_centroid
from ..machine import sensors

POS_SIGMA = .005       # m: outline position noise per frame, per axis
EDGE_SIGMA = .004      # m: outline size noise per frame (the outline grows or shrinks by about this)
HEIGHT_SIGMA = .005    # m: top height noise
SEG_GAP = .015         # m: lumps closer than this (3-D) form one blob ...
SEG_GAP_OUT = .025     # ... and stay one until they are this far apart (hysteresis: no flicker at the threshold)
VIS_MIN = .5           # a lump is detected when at least this fraction of its top points is in some view ...
VIS_FULL = .9          # ... and fully trusted from this fraction on
MATCH_GATE = .20       # m: a track and a blob are the same object within this (at 0.4 m/s a frame is 4 mm)
LOST_S = .20           # s: a track without a blob for this long is dropped (lost, unless at an exit)
GHOST_S = .015         # s: a track no blob has matched for longer than this (one frame missed) is a ghost: its count
                       # is not added in a merge
REACQUIRE = .30        # m: a lost object counts as found again when a new track appears this close to it
SPLIT_GAP = .05        # m: a blob that appears this close to a tracked one came off it (the track split)
SPLIT_PAD = .03        # m: a blob that appears with its centroid inside the last outline (grown by this) of a track
                       # that found no blob this frame is a piece of it
SHRINK = .75           # a merged blob whose plan area falls below this share of its area at the merge lost a lump
BIRTH_S = .10          # s: a new track is fully trusted only after this long
SOLID_MULTI = .90      # plan area / hull area below this: the blob holds more than one lump ...
SOLID_SURE = .96       # ... and between the two it is ambiguous (confidence halved)
FAST = 1.0             # m/s: faster than this (falling, tumbling) blurs: confidence halved
CONF_OK = .6           # confidence a controller needs before it acts on an object
VEL_WINDOW = .30       # s: velocity = least-squares slope of the centroid over this window
RASTER = .01           # m: raster for the area of a blob of three or more lumps


def solidity(hulls):
    """Plan area of the union of convex polygons over the area of their common convex hull: 1 for one
    convex lump, clearly less for two touching ones."""
    H = hull2(np.concatenate(hulls))
    aH = poly_area(H)
    if aH <= 0:
        return 1.
    if len(hulls) == 2:
        I = clip(hulls[0], hulls[1])
        return (poly_area(hulls[0]) + poly_area(hulls[1]) - (poly_area(I) if len(I) >= 3 else 0.)) / aH
    lo, hi = H.min(0), H.max(0)
    X, Y = np.meshgrid(np.arange(lo[0] + RASTER / 2, hi[0], RASTER), np.arange(lo[1] + RASTER / 2, hi[1], RASTER))
    inH = inside(H, X, Y)
    inU = np.zeros_like(inH)
    for P in hulls:
        inU |= inside(P, X, Y)
    n = int(inH.sum())
    return float(inU[inH].sum()) / n if n else 1.


def groups(ks, dist, together=frozenset()):
    """Lumps closer than SEG_GAP -- or SEG_GAP_OUT for pairs in `together` (one blob last frame) -- joined
    transitively: a list of lists of lump ids."""
    parent = {k: k for k in ks}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if root(a) != root(b) and dist(a, b) < (SEG_GAP_OUT if frozenset((a, b)) in together else SEG_GAP):
                parent[root(b)] = root(a)
    out = {}
    for k in ks:
        out.setdefault(root(k), []).append(k)
    return list(out.values())


class Frame(dict):
    """One vision frame as the controllers get it: {tid: object}, with .t (when it was true) and .health."""


class Vision:
    """Fused overhead cameras -> tracked objects (see the module docstring)."""

    def __init__(self, cameras, rng, exits, oracle=False):
        """cameras: machine.sensors.layout()['cameras']; rng: the noise stream; exits(x, y) -> True where a lump may
        leave the view normally (past the measuring belt: the separator and the bins)."""
        self.cams, self.rng, self.exits, self.oracle = cameras, rng, exits, oracle
        self.tracks, self.next_tid = {}, 0
        self.together = frozenset()                        # lump pairs in one blob last frame (hysteresis)
        self.queue = deque()
        self.lost, self.events = [], []
        self.outage = []                                   # injected (t0, t1): every camera dark
        self.counts = dict(births=0, merges=0, splits=0, lost=0, left_at_exit=0, reacquired=0, taken_off=0,
                           apart=0)
        self.err = []                                      # verification: |outline centroid - true centroid|
        self.latest = Frame()
        self.latest.t, self.latest.health = 0., dict(ok=True, why=None)

    @property
    def margin(self):
        """m: the position margin a controller puts on an edge these cameras judge: 3 x the outline position noise
        (none on the ideal view)."""
        return 0. if self.oracle else 3 * POS_SIGMA

    def dark(self, t):
        return any(t0 <= t < t1 for t0, t1 in self.outage)

    # ---- the scene -> blobs ------------------------------------------------------------------------
    def _visible(self, V, t):
        """Fraction of a lump's top points inside at least one camera's view, and those points."""
        if self.dark(t):
            return 0., V[:0]
        z = V[:, 2]
        top = V[z >= z.min() + .5 * (z.max() - z.min())]
        seen = np.zeros(len(top), bool)
        for c in self.cams:
            seen |= sensors.in_view(c, top)
        return float(seen.mean()), top[seen]

    def _blobs(self, t, lumps, dist):
        """lumps: {k: world vertices}; dist(i, j): 3-D gap between two lumps (capped). Returns the blobs."""
        vis = {k: self._visible(lumps[k], t) for k in sorted(lumps)}
        seen = [k for k in sorted(lumps) if vis[k][0] >= VIS_MIN]
        out = []
        gs = groups(seen, dist, self.together) if not self.oracle else [[k] for k in seen]
        self.together = frozenset(frozenset((a, b)) for g in gs for a in g for b in g if a < b)
        for g in gs:
            hulls = [hull2(lumps[k][:, :2]) for k in g]
            pts = hull2(np.concatenate([lumps[k][:, :2] if vis[k][0] >= 1. else vis[k][1][:, :2] for k in g]))
            out.append(dict(truth=tuple(g), pts=pts, solid=solidity(hulls) if len(g) > 1 else 1.,
                            vis=min(vis[k][0] for k in g), ztop=max(float(lumps[k][:, 2].max()) for k in g)))
        return out

    def _measure(self, b):
        P, solid, ztop = b['pts'], b['solid'], b['ztop']
        if not self.oracle:
            c = P.mean(0)
            size = max(float(np.linalg.norm(P - c, axis=1).mean()), 1e-3)
            P = c + (P - c) * (1. + self.rng.normal(0., EDGE_SIGMA) / size) + self.rng.normal(0., POS_SIGMA, 2)
            solid = min(1., solid + self.rng.normal(0., .01))
            ztop += self.rng.normal(0., HEIGHT_SIGMA)
        c = poly_centroid(P)
        return dict(pts=P, cx=float(c[0]), cy=float(c[1]), x0=float(P[:, 0].min()), x1=float(P[:, 0].max()),
                    y0=float(P[:, 1].min()), y1=float(P[:, 1].max()), ztop=float(ztop), solid=float(solid),
                    vis=b['vis'], _truth=b['truth'])

    # ---- tracking --------------------------------------------------------------------------------------
    @staticmethod
    def velocity(h):
        """Least-squares centroid velocity over a track history [(t, x, y)]; None until it spans VEL_WINDOW."""
        if len(h) < 3 or h[-1][0] - h[0][0] < VEL_WINDOW - 1e-9:
            return None
        T = np.array([p[0] for p in h])
        A = np.stack([T - T.mean(), np.ones_like(T)], 1)
        sol = np.linalg.lstsq(A, np.array([[p[1], p[2]] for p in h]), rcond=None)[0]
        return float(sol[0, 0]), float(sol[0, 1])

    def _predict(self, tr, t):
        h, v = tr['hist'], self.velocity(tr['hist'])
        dt = t - h[-1][0]
        return h[-1][1] + (v[0] * dt if v else 0.), h[-1][2] + (v[1] * dt if v else 0.)

    def _new(self, t, m, lineage=frozenset(), count=1, why='births', born=None):
        """born: a merge or split continues objects the cameras already knew: it keeps their age."""
        tid = self.next_tid
        self.next_tid += 1
        self.tracks[tid] = dict(tid=tid, lineage=frozenset(lineage) | {tid}, count=count,
                                born=t if born is None else born, seen=t,
                                hist=[(t, m['cx'], m['cy'])], m=m, area_ref=poly_area(m['pts']))
        self.counts[why] += 1
        if why == 'births':
            for L in self.lost:                              # a new track where one was lost: found again
                if not L.get('reacquired') and math.hypot(m['cx'] - L['at'][0], m['cy'] - L['at'][1]) < REACQUIRE:
                    L['reacquired'] = round(t, 3)
                    self.counts['reacquired'] += 1
                    break
        return tid

    def _update(self, tr, t, m):
        tr['hist'].append((t, m['cx'], m['cy']))
        tr['seen'], tr['m'] = t, m
        if tr['count'] > 1:                         # a blob of several lumps that shrank has lost one of them
            a = poly_area(m['pts'])
            if a < SHRINK * tr['area_ref']:
                tr['count'] -= 1
                tr['area_ref'] = a
                self.events.append(dict(t_s=round(t, 3), event='shrank', tid=tr['tid'], count=tr['count']))

    def frame(self, t, lumps, dist, com=None):
        """Advance one 10 ms frame. lumps: {k: world vertices} of every lump in the line (above the drop
        level, not taken off the line); com: {k: true centroid} -- the oracle's centroid, and the
        verification's reference. Returns what the controllers see now (sensors.LATENCY_S old)."""
        blobs = [self._measure(b) for b in self._blobs(t, lumps, dist)]
        if self.oracle:
            fr = Frame()
            for m in blobs:
                k = m['_truth'][0]
                c = com[k] if com is not None else (m['cx'], m['cy'])
                fr[k] = dict(m, cx=float(c[0]), cy=float(c[1]), tid=k, lineage=frozenset(m['_truth']), n_est=1,
                             conf=1., conf_seen=1., coasting=False, vx=None, vy=None, age_s=None)
            fr.t, fr.health = t, dict(ok=True, why=None)
            self.latest = fr
            return fr
        tracks = list(self.tracks.values())
        pred = {tr['tid']: self._predict(tr, t) for tr in tracks}
        cand = [[tr['tid'] for tr in tracks
                 if inside(m['pts'], *pred[tr['tid']], pad=.03)
                 or math.hypot(m['cx'] - pred[tr['tid']][0], m['cy'] - pred[tr['tid']][1]) < MATCH_GATE]
                for m in blobs]
        by_track = {}
        for i, near in enumerate(cand):
            for tid in near:
                by_track.setdefault(tid, []).append(i)
        done_b, done_t = set(), set()
        for i, near in enumerate(cand):
            if len(near) == 1 and by_track[near[0]] == [i]:                  # the same object
                self._update(self.tracks[near[0]], t, blobs[i])
                done_b.add(i)
                done_t.add(near[0])
            elif len(near) >= 2 and all(by_track[tid] == [i] for tid in near):   # tracks ran together: merge
                old = [self.tracks.pop(tid) for tid in near]
                # A ghost's lumps are already counted: the blob it followed came apart and its pieces went on under
                # other ids. Added again at every merge, the count of ONE lump lying next to others on the feed
                # belt reached 13 in 19 s (scatter 7001), and the station held it as 'multi' -- a single lump void
                live = [o for o in old if t - o['seen'] <= GHOST_S] or old
                self._new(t, blobs[i], frozenset().union(*(o['lineage'] for o in old)),
                          sum(o['count'] for o in live), 'merges', born=max(o['born'] for o in old))
                self.events.append(dict(t_s=round(t, 3), event='merge', tids=list(near)))
                done_b.add(i)
                done_t.update(near)
        for tid, idx in by_track.items():
            if tid in done_t or tid not in self.tracks:
                continue
            idx = [i for i in idx if i not in done_b]
            if len(idx) >= 2 and all(cand[i] == [tid] for i in idx):         # one track, several blobs: split
                old = self.tracks.pop(tid)
                kids = [self._new(t, blobs[i], old['lineage'], 1, 'splits', born=old['born']) for i in idx]
                self.events.append(dict(t_s=round(t, 3), event='split', tid=tid, into=kids))
                done_b.update(idx)
                done_t.add(tid)
        born = []
        for i, m in enumerate(blobs):                                         # anything left: nearest wins
            if i in done_b:
                continue
            free = [tid for tid in cand[i] if tid not in done_t and tid in self.tracks]
            if free:
                tid = min(free, key=lambda k: math.hypot(m['cx'] - pred[k][0], m['cy'] - pred[k][1]))
                self._update(self.tracks[tid], t, m)
                done_t.add(tid)
            else:
                born.append(m)
        apart = {}
        for m in born:
            # a blob appearing right next to a track that is still seen came off it: the track split. Both
            # halves get new ids that remember the parent, and the one that stays keeps what is left of its count
            par = [tr for tr in self.tracks.values() if tr['seen'] >= t - 1e-9 and tr['born'] < t - 1e-9
                   and box_gap(tr['m'], m) < SPLIT_GAP]
            if par:
                old = self.tracks.pop(min(par, key=lambda tr: box_gap(tr['m'], m))['tid'])
                stay = self._new(t, old['m'], old['lineage'], max(1, old['count'] - 1), 'splits', born=old['born'])
                self.tracks[stay]['hist'] = old['hist']
                kid = self._new(t, m, old['lineage'], 1, 'splits', born=old['born'])
                self.events.append(dict(t_s=round(t, 3), event='split', tid=old['tid'], into=[stay, kid]))
                continue
            # a blob appearing inside the last outline of a track that found no blob this frame is a piece of it.
            # Two lumps in a row: the outline centroid of their blob lies between them, more than MATCH_GATE from
            # either one's own, so when they part neither piece is matched to the blob's track. Until 2026-10-04
            # both were new objects without a past and the track lingered as a ghost: the station found an item's
            # object gone and an unknown one in its place, and held a single lump 'track_lost' (touching 7004)
            src = [tr for tid, tr in self.tracks.items() if tid not in done_t and tr['seen'] < t - 1e-9
                   and inside(tr['m']['pts'], m['cx'], m['cy'], pad=SPLIT_PAD)]
            if src:
                old = min(src, key=lambda tr: math.hypot(m['cx'] - tr['m']['cx'], m['cy'] - tr['m']['cy']))
                apart.setdefault(old['tid'], []).append(self._new(t, m, old['lineage'], 1, 'splits', born=old['born']))
            else:
                self._new(t, m)
        for tid, kids in apart.items():                                       # the blob's own track ends here
            del self.tracks[tid]
            self.counts['apart'] += 1
            self.events.append(dict(t_s=round(t, 3), event='split', tid=tid, into=kids, apart=True))
        for tid in [tid for tid in list(self.tracks) if tid not in done_t]:
            tr = self.tracks[tid]
            if t - tr['seen'] < LOST_S - 1e-9:
                continue                                                      # coasting
            del self.tracks[tid]
            m = tr['m']
            if self.exits(m['cx'], m['cy']) and not self.dark(t):
                self.counts['left_at_exit'] += 1
            else:
                self.counts['lost'] += 1
                self.lost.append(dict(t_s=round(t, 3), tid=tid, at=[round(m['cx'], 3), round(m['cy'], 3)],
                                      _truth=m['_truth']))
                self.events.append(dict(t_s=round(t, 3), event='lost', tid=tid))
        fr = Frame()
        for tr in self.tracks.values():
            h = tr['hist']
            while len(h) > 2 and h[1][0] < t - VEL_WINDOW - 1e-9:
                h.pop(0)
            m, v, age = tr['m'], self.velocity(h), t - tr['born']
            coasting = tr['seen'] < t - 1e-9
            seen = 0. if coasting else min(1., max(0., (m['vis'] - VIS_MIN) / (VIS_FULL - VIS_MIN)))
            seen *= min(1., age / BIRTH_S)
            if v and math.hypot(*v) > FAST:
                seen *= .5
            conf = seen * (.5 if SOLID_MULTI <= m['solid'] < SOLID_SURE else 1.)   # one lump or two: not sure
            fr[tr['tid']] = dict(m, tid=tr['tid'], lineage=tr['lineage'], conf=round(conf, 3),
                                 conf_seen=round(seen, 3), coasting=coasting,
                                 n_est=max(tr['count'], 2 if m['solid'] < SOLID_MULTI else 1),
                                 vx=None if v is None else v[0], vy=None if v is None else v[1], age_s=round(age, 3))
        fr.t, fr.health = t, self._health(t)
        if com is not None:
            self.audit(t, com)
        self.queue.append(fr)
        while len(self.queue) > 1 and self.queue[1].t <= t - sensors.LATENCY_S + 1e-9:
            self.queue.popleft()
        self.latest = self.queue[0]
        return self.latest

    def audit(self, t, com):
        """Verification only: outline centroid against the true centre of mass, per single-lump object seen
        this frame. com: {k: true centroid}."""
        for tr in self.tracks.values():
            m = tr['m']
            if tr['seen'] >= t - 1e-9 and len(m['_truth']) == 1 and m['_truth'][0] in com:
                c = com[m['_truth'][0]]
                self.err.append(math.hypot(m['cx'] - c[0], m['cy'] - c[1]))

    def _health(self, t):
        if self.dark(t):
            return dict(ok=False, why='camera_outage')
        open_ = [L for L in self.lost if not L.get('reacquired')]
        if open_:
            return dict(ok=False, why='track_lost', at=open_[0]['at'])
        return dict(ok=True, why=None)

    def taken_off(self, t, lumps):
        """The simulation has taken these lumps off the line (a held item): their tracks end without a lost
        alarm."""
        gone = [tid for tid, tr in self.tracks.items() if set(tr['m']['_truth']) & set(lumps)]
        for tid in gone:
            del self.tracks[tid]
        self.counts['taken_off'] += len(gone)
        self.events.append(dict(t_s=round(t, 3), event='taken_off', lumps=sorted(lumps), tracks=len(gone)))
        self.queue.clear()                                 # no frame from before may still show them

    def report(self):
        e = np.array(self.err)
        return dict(mode='oracle' if self.oracle else 'vision', counts=self.counts,
                    lost=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in L.items()} for L in self.lost],
                    events=self.events[-300:],
                    centroid_error_m=None if not len(e) else dict(
                        p50=round(float(np.median(e)), 4), p95=round(float(np.percentile(e, 95)), 4),
                        max=round(float(e.max()), 4), frames=int(len(e)),
                        note='outline area centroid (what the controllers get) against the true centre of mass'),
                    parameters=dict(latency_s=sensors.LATENCY_S, pos_sigma_m=POS_SIGMA, edge_sigma_m=EDGE_SIGMA,
                                    height_sigma_m=HEIGHT_SIGMA, seg_gap_m=SEG_GAP, solid_multi=SOLID_MULTI,
                                    conf_ok=CONF_OK, lost_after_s=LOST_S,
                                    note='placeholders; white noise per frame; no occlusion by equipment'))
