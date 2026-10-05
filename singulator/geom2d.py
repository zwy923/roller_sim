"""Plane geometry on plan-view outlines: convex polygons as (n, 2) arrays, objects' bounding boxes.

Pure numpy; no state, no tunables. Used by the layouts (do two lumps overlap), the cameras' model (outline,
centroid, how convex), the face's closing sweep and the station (how close are two objects).
"""
import math

import numpy as np


def poly_area(P):
    x, y = P[:, 0], P[:, 1]
    return .5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def poly_centroid(P):
    """Area centroid of a polygon (its vertex mean when it has no area)."""
    x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    a = c.sum() / 2
    if abs(a) < 1e-12:
        return P.mean(0)
    return np.array([((x + xn) * c).sum() / (6 * a), ((y + yn) * c).sum() / (6 * a)])


def hull2(P):
    """Convex hull of plan points, counter-clockwise (Andrew's monotone chain; points on an edge are dropped).
    Until 2026-10-04 this was scipy's ConvexHull. On Windows scipy opens a temporary file for qhull's messages
    at every call; at 600 calls per simulated second that was most of a run's wall time (23 s of 33 s, three lumps
    for 8 s), and runs in parallel processes waited on each other for it (EXPERIMENTS.md S6)."""
    P = np.asarray(P, float)
    pts = sorted(set(map(tuple, P.tolist())))
    half = []
    for seq in (pts, pts[::-1]):                        # lower chain left to right, upper chain right to left
        h = []
        for p in seq:
            while len(h) >= 2:
                (ax, ay), (bx, by) = h[-2], h[-1]
                if (bx - ax) * (p[1] - ay) > (by - ay) * (p[0] - ax):
                    break                               # a left turn at b: b is on the hull so far
                h.pop()
            h.append(p)
        half.append(h[:-1])
    H = half[0] + half[1]
    return np.array(H) if len(H) >= 3 else P            # degenerate (collinear) point set: as it is


def inside(P, x, y, pad=0.):
    """Which points (x, y) (scalars or arrays) lie inside the counter-clockwise convex polygon P grown by pad."""
    e = np.roll(P, -1, 0) - P
    n = np.stack([e[:, 1], -e[:, 0]], 1)                # outward normals
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    q = np.stack([np.asarray(x, float), np.asarray(y, float)], -1)
    d = np.einsum('...ij,ij->...i', q[..., None, :] - P, n)
    return np.all(d <= pad, axis=-1)


def clip(P, Q):
    """Intersection of two convex counter-clockwise polygons (Sutherland-Hodgman)."""
    out = P
    for i in range(len(Q)):
        a, b = Q[i], Q[(i + 1) % len(Q)]
        if not len(out):
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        res = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                res.append(p)
            if sp * sq < 0:
                res.append(p + (q - p) * (sp / (sp - sq)))
        out = np.array(res)
    return out


def overlap(a, b, margin=.005):
    """Separating-axis test on two convex polygons, inflated by margin. True = they would overlap."""
    for poly in (a, b):
        e = np.roll(poly, -1, 0) - poly
        n = np.stack([-e[:, 1], e[:, 0]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        pa, pb = a @ n.T, b @ n.T
        if (pa.max(0) + margin < pb.min(0)).any() or (pb.max(0) + margin < pa.min(0)).any():
            return False
    return True


def overlap_many(P, H, margin):
    """Separating-axis test of the convex polygons P (N, k, 2) against one convex polygon H (m, 2).
    True where P[i] and H come within `margin` of each other."""
    def normals(E):
        n = np.stack([-E[..., 1], E[..., 0]], -1)
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    A = normals(np.roll(P, -1, axis=1) - P)                       # (N, k, 2) axes of P
    pP, pH = np.einsum('nkd,njd->nkj', P, A), np.einsum('md,njd->nmj', H, A)
    sep = ((pP.max(1) + margin < pH.min(1)) | (pH.max(1) + margin < pP.min(1))).any(-1)
    a = normals(np.roll(H, -1, axis=0) - H)                       # (m, 2) axes of H
    qP, qH = np.einsum('nkd,jd->nkj', P, a), H @ a.T
    sep |= ((qP.max(1) + margin < qH.min(0)) | (qH.max(0) + margin < qP.min(1))).any(-1)
    return ~sep


def box_gap(a, b):
    """Plan gap between two objects' bounding boxes (0 when they overlap). An object is anything with the keys
    x0, x1, y0, y1."""
    dx = max(0., max(a['x0'], b['x0']) - min(a['x1'], b['x1']))
    dy = max(0., max(a['y0'], b['y0']) - min(a['y1'], b['y1']))
    return math.hypot(dx, dy)
