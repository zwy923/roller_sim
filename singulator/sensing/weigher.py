"""The load cells under the measuring belt: the reading."""
import mujoco
import numpy as np


def weigher_reading(model, data, weigher, lump_of, load_only=False):
    """Load-cell reading of M: the downward contact force of everything touching M and its skirts, N.
    Also, for the verification only, which lumps load M and what else every lump touches (other lumps by
    id, 'equipment' off the weigh frame). load_only: the force alone (every physics step while weighing)."""
    Fz, on, other, w = 0., set(), {}, np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        k1, k2 = lump_of.get(g1), lump_of.get(g2)
        if g1 in weigher or g2 in weigher:
            k = k2 if g1 in weigher else k1
            if k is None:
                continue
            mujoco.mj_contactForce(model, data, i, w)
            f = con.frame.reshape(3, 3).T @ w[:3]            # force of geom1 on geom2, world frame
            Fz += f[2] if g1 in weigher else -f[2]
            on.add(k)
            continue
        if load_only:
            continue
        for k, ok in ((k1, k2), (k2, k1)):
            if k is not None:
                other.setdefault(k, set()).add(ok if ok is not None else 'equipment')
    return Fz if load_only else (Fz, on, other)
