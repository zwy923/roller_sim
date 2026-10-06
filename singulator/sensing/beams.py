"""Through-beams: a light beam across a belt as the controllers read it.

A 20 ms debounce, injected faults, and three diagnoses made from the cameras and the timing alone: blocked longer
than a lump can take (long block: a jam), blocked while the cameras see the line clear (dirty or misaligned),
never blocked while the cameras see a lump across it (dead). oracle=True: the geometric line test without debounce.
Where each beam is and how long it may stay blocked: machine/sensors.py.
"""
import numpy as np
from scipy.spatial import ConvexHull

DEBOUNCE_S = .02       # s: a beam changes state only after the new level held this long
DIAG_S = .50           # s: blocked this long while the cameras see the line clear = dirty; a lump seen across
                       # the line this long without a block = dead


def cuts_line(V, x, z, box=None):
    """A beam across the belt at (x, z) meets the convex hull V exactly when (x, z) lies in the hull's
    projection on the x-z plane. box: V's ((min x, y, z), (max x, y, z)), when known."""
    (x0, _, z0), (x1, _, z1) = box if box is not None else (V.min(0), V.max(0))
    if not (x0 < x < x1 and z0 < z < z1):
        return False
    h = ConvexHull(V[:, [0, 2]])
    return bool(np.all(h.equations[:, :2] @ (x, z) + h.equations[:, 2] <= 0.))


class Beam:
    """One through-beam: the line test on the scene, debounce, faults, diagnoses."""

    def __init__(self, name, x, z, max_block_s, oracle=False):
        self.name, self.x, self.z, self.max_block_s, self.oracle = name, x, z, max_block_s, oracle
        self.blocked, self.raw, self.since, self.raw_since = False, False, 0., 0.
        self.fault = None                                       # None, 'dead' (never blocks) or 'dirty' (always)
        self.edges, self.alarm = [], None
        self.t_dirty = self.t_dead = self.last_t = None
        self.run_block = 0.                                     # s blocked while no block was planned

    def sample(self, t, lumps, boxes=None):
        """lumps: world vertices of every lump in the line; boxes: their bounding boxes, when known (cuts_line).
        Returns the (debounced) state."""
        raw = any(cuts_line(V, self.x, self.z, B) for V, B in zip(lumps, boxes or [None] * len(lumps)))
        if self.fault == 'dead':
            raw = False
        elif self.fault == 'dirty':
            raw = True
        if raw != self.raw:
            self.raw, self.raw_since = raw, t
        if self.raw != self.blocked and t - self.raw_since >= (0. if self.oracle else DEBOUNCE_S) - 1e-9:
            self.blocked, self.since = self.raw, t
            self.edges.append((round(t, 3), 'block' if self.blocked else 'clear'))
        return self.blocked

    def blocks_since(self, t0):
        return sum(1 for t, e in self.edges if e == 'block' and t >= t0 - 1e-9)

    def diagnose(self, t, seen_clear, seen_across, expected_block=False):
        """seen_clear: the cameras see nothing near the line; seen_across: they see a lump's body across it
        (not just a nose or a tail); expected_block: a block is planned now (a lump staged at the line, or the
        belt under it stopped), so it does not count toward a long block. Returns the alarm (None while
        healthy); every alarm stops the line."""
        dt = t - self.last_t if self.last_t is not None else 0.
        self.last_t = t
        if self.alarm is not None:
            return self.alarm
        if self.blocked:
            self.t_dead = None
            self.run_block = self.run_block + dt if not expected_block else self.run_block
            if seen_clear:
                self.t_dirty = self.t_dirty if self.t_dirty is not None else t
                if t - self.t_dirty >= DIAG_S - 1e-9:
                    self.alarm = dict(t_s=round(t, 3), beam=self.name, why='dirty',
                                      note='blocked while the cameras see the line clear: dirty or misaligned')
            else:
                self.t_dirty = None
                if self.run_block > self.max_block_s:
                    self.alarm = dict(t_s=round(t, 3), beam=self.name, why='long_block',
                                      note='something stays across the line longer than a lump takes to pass')
        else:
            self.t_dirty, self.run_block = None, 0.
            if seen_across:
                self.t_dead = self.t_dead if self.t_dead is not None else t
                if t - self.t_dead >= DIAG_S - 1e-9:
                    self.alarm = dict(t_s=round(t, 3), beam=self.name, why='dead',
                                      note='the cameras see a lump across the line, the beam does not')
            else:
                self.t_dead = None
        return self.alarm

    def report(self):
        return dict(name=self.name, x=round(float(self.x), 4), z=round(float(self.z), 4),
                    max_block_s=self.max_block_s, blocks=sum(e[1] == 'block' for e in self.edges),
                    fault=self.fault, alarm=self.alarm, edges=self.edges[-200:])
