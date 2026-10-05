"""The singulation section as the cameras see it: what the plough face's closing permission and the stall rule
read.

SectionWatch follows the objects in the section (plough start - 0.3 m .. lane exit): which are in it, which have
not advanced jam_speed * jam_window over jam_window (the stall rule, on the tracked outline centroid), and whether
a tail left the lane recently (the section is still emptying).
"""
import numpy as np


class SectionWatch:
    """Vision only: the singulation section (plough start - 0.3 m .. lane exit) as the cameras see it -- which
    objects are in it, which have not advanced jam_speed * jam_window over jam_window (the stall rule on the
    tracked outline centroid), and whether a tail left the lane recently."""

    def __init__(self, cfg, d, lo):
        self.cfg, self.lo, self.exit_x = cfg, lo, d['exit_x']
        self.hist, self.since, self.tail = {}, {}, {}

    def update(self, t, view, waiting):
        W = self.cfg['jam_window']
        section = {tid: o for tid, o in view.items() if o['x0'] < self.exit_x and o['x1'] > self.lo}
        for tid, o in view.items():
            h = self.hist.setdefault(tid, [])
            h.append((t, o['cx']))
            while len(h) > 2 and h[1][0] <= t - W + 1e-9:
                h.pop(0)
            if o['x0'] > self.exit_x and tid not in self.tail and o['x0'] < self.exit_x + .5:
                self.tail[tid] = t
        stuck = []
        for tid, o in section.items():
            if waiting(o):
                self.since.pop(tid, None)
                continue
            s = self.since.setdefault(tid, t)
            h = self.hist[tid]
            if t - s >= W - 1e-9 and h[0][0] <= t - W + .02 and h[-1][1] - h[0][1] < self.cfg['jam_speed'] * W:
                stuck.append(tid)
        for tid in [k for k in self.since if k not in section]:
            del self.since[tid]
        passing = any(tt > t - W for tt in self.tail.values())
        return section, stuck, passing

    @staticmethod
    def plans(section, margin):
        """Outlines grown by the cameras' position margin, for the face's closing-sweep check."""
        out = {}
        for tid, o in section.items():
            P = o['pts']
            c = P.mean(0)
            r = np.maximum(np.linalg.norm(P - c, axis=1, keepdims=True), 1e-9)
            out[tid] = P + (P - c) / r * margin
        return out
