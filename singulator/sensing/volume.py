"""The volume measurement on the measuring belt: the volume, and whether the scan is usable.

What the station asks of a volume device:

    read(t, rest)   one sample of an item at rest since `rest`: None until the device has its verdict, then -- once
                    -- dict(valid, reasons, volume_m3, ...): valid False = the measurement cannot stand for one lump
    describe()      for result.json

Scanner is the placeholder device: three depth heads (machine/sensors.py) that deliver their verdict scan_s after
the belt came to rest. The volume it returns is still the ideal hull volume; what is modelled is the verdict:
exactly one object in the zone, wholly on the belt, seen by two heads, and a top surface one convex lump could have
-- two touching lumps leave a valley under the roof of their common hull deeper than SCAN_DEFECT. The lumps are
convex, which flatters the valley test: real coal is not. The volume detection that is being developed takes its
place by implementing read() and describe() (docs/ARCHITECTURE.md).
"""
import numpy as np
from scipy.spatial import ConvexHull

from ..machine import sensors
from . import vision

SCAN_RASTER = .005     # m: raster of the scanner's height map
SCAN_DEFECT = .03      # m: a valley this deep under the hull roof = more than one lump
SCAN_SIGMA = .003      # m: height-map noise of the scanner


def top_surface(V, X, Y):
    """Top of the convex hull of the points V over the grid (X, Y); nan outside its plan outline."""
    h = ConvexHull(V)
    A, b = h.equations[:, :3], h.equations[:, 3]
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    rhs = -(b[None, :] + x * A[None, :, 0] + y * A[None, :, 1])          # a_z * z <= rhs, facet by facet
    up, dn, side = A[:, 2] > 1e-9, A[:, 2] < -1e-9, np.abs(A[:, 2]) <= 1e-9
    top = (rhs[:, up] / A[up, 2]).min(1)
    bot = (rhs[:, dn] / A[dn, 2]).max(1) if dn.any() else np.full(len(x), -np.inf)
    ok = top >= bot - 1e-9
    if side.any():
        ok &= np.all(rhs[:, side] >= -1e-9, axis=1)
    return np.where(ok, top, np.nan).reshape(X.shape)


class Scanner:
    """The volume scanner over the measuring belt: the volume, and whether the scan is usable."""

    def __init__(self, heads, zone, rng, oracle=False, scan_s=1., look=None):
        """heads, zone: machine.sensors.layout()['scanner']; rng: the noise stream; scan_s: how long a scan takes
        from the belt at rest; look() -> (lumps, volume, dist, belt): what lies under the scanner now, as scan()
        takes it (the true scene: sensing.suite.Sensors supplies it)."""
        self.heads, self.zone, self.rng, self.oracle = heads, zone, rng, oracle
        self.scan_s, self.look = scan_s, look
        self.fail_on = set()                                 # injected: these scans (0, 1, ...) fail
        self.scans = []

    def read(self, t, rest):
        """The device as the station uses it: the verdict, scan_s after the belt came to rest."""
        if t < rest + self.scan_s - 1e-9:
            return None
        res = self.scan(t, *self.look())
        return {k: v for k, v in res.items() if not k.startswith('_')}

    def describe(self):
        return dict(volume='true hull volume (ideal value); usability judged by the scanner model')

    def scan(self, t, lumps, volume, dist, belt):
        """lumps: {k: world vertices} of every lump reaching into the scan zone (the scene); volume(k): true
        hull volume; dist(i, j): 3-D gap; belt: ((x0, y0), (x1, y1)) of the measuring belt between its skirts.
        Returns the verdict; '_truth' is for the verification only."""
        z0 = self.zone[0][2]
        (x0, y0), (x1, y1) = belt
        res = dict(t_s=round(t, 3), valid=True, reasons=[], volume_m3=None, objects=0, defect_m=None,
                   _truth=tuple(sorted(lumps)))
        if len(self.scans) in self.fail_on:
            res.update(valid=False, reasons=['scan_failed'])
            self.scans.append(res)
            return res
        self.scans.append(res)
        gs = vision.groups(sorted(lumps), dist)
        res['objects'] = len(gs)          # (a single scan: no hysteresis)
        if len(gs) != 1:
            res.update(valid=False, reasons=['objects_in_zone_%d' % len(gs)])
            return res
        V = np.concatenate([lumps[k] for k in gs[0]])
        if V[:, 0].min() < x0 or V[:, 0].max() > x1 or V[:, 1].min() < y0 or V[:, 1].max() > y1:
            res['valid'] = False
            res['reasons'].append('not_wholly_on_the_belt')
        top = V[V[:, 2] >= V[:, 2].min() + .3 * (V[:, 2].max() - V[:, 2].min())]
        if (sum(sensors.in_view(h, top).astype(int) for h in self.heads) < 2).any():
            res['valid'] = False
            res['reasons'].append('coverage')
        lo, hi = V[:, :2].min(0), V[:, :2].max(0)
        X, Y = np.meshgrid(np.arange(lo[0] + SCAN_RASTER / 2, hi[0], SCAN_RASTER),
                           np.arange(lo[1] + SCAN_RASTER / 2, hi[1], SCAN_RASTER))
        roof = top_surface(V, X, Y)
        surf = np.full(X.shape, z0)
        for k in gs[0]:
            surf = np.fmax(surf, np.nan_to_num(top_surface(lumps[k], X, Y), nan=z0))
        if not self.oracle:
            surf = surf + self.rng.normal(0., SCAN_SIGMA, surf.shape)
        inroof = np.isfinite(roof)
        res['defect_m'] = round(float(np.percentile((roof - surf)[inroof], 99.5)), 4) if inroof.any() else 0.
        if res['defect_m'] > SCAN_DEFECT:
            res['valid'] = False
            res['reasons'].append('not_one_lump')
        res['volume_m3'] = round(sum(volume(k) for k in gs[0]), 6)
        return res
