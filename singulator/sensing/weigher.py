"""Weighing on the measuring belt: the load cells and the device that turns their signal into a mass.

Two things, kept apart so that either can be replaced:

  LoadCell   the signal. In the model: the downward contact force of every lump on the weigh frame (the measuring
             belt and its skirts), an ideal sum without load-cell dynamics, belt tension or vibration;
  Weigher    the device as the station uses it (the interface below). FixedTimeWeigher and SteadyWeigher are the two
             placeholder devices of --weigh-model; the belt scale that is still to be chosen takes their place by
             implementing the same interface (make_weigher, docs/ARCHITECTURE.md).

What the station asks of a weigher:

    reading              N: what the indicator shows this sample (not zeroed)
    track_zero()         the belt is seen empty this sample: follow the zero
    clear()              the scale was cleaned by hand: the zero starts again
    begin() -> tare      an item is coming to rest: a weighing starts. Returns the zero it will be read against (N)
    zero_ok(tare)        was the scale clear when that weighing began?
    read(t, rest, tare)  one sample of an item at rest since `rest`: None until the device has its result, then
                         dict(t_weigh_s, weigh_wait_s, steady, mass_kg, reading_std_N), with void='unsteady' when
                         the result must not be used
    describe(), limits() for result.json

and what the run feeds it: load_step(Fz) every physics step while the station weighs, sample(Fz) every 10 ms.
Nothing here is calibrated.
"""
import mujoco
import numpy as np

from ..config import SAMPLE_S

FILTER_S = .30        # s: the indicator's moving average of the load-cell samples ...
STEADY_S = .50        # s: ... is steady when over this long it stays within ...
STEADY_BAND = .01     # ... this fraction of its value (at least STEADY_MIN_N); 60 lumps on the second version all
STEADY_MIN_N = 2.     # N: settled within 0.8 s this way, 5 never did under the old 2 % std rule (rocking contacts)
WEIGH_MAX_S = 3.      # s: SteadyWeigher: at rest this long without a steady reading: void (unsteady)
TARE_TOL = 20.        # N: the empty scale must read within this of zero (about 2 kg) ...
ZERO_BAND = 3.        # N: ... it follows the empty reading only within this band (zero tracking), by 1/ZERO_N ...
ZERO_N = 50           # ... per sample
ZERO_OFF_S = 1.       # s: the empty reading outside the band this long: the scale is not clear (tare fault)
G = 9.81


class LoadCell:
    """The load cells of the measuring belt in the model."""

    def __init__(self, model, frame, lump_of):
        """frame: the geoms the load cells carry (machine.assembly.weigh_frame); lump_of: {geom id: lump}."""
        self.model, self.frame, self.lump_of = model, frame, lump_of
        self._w = np.zeros(6)

    def force(self, data):
        """The reading now, N: the downward contact force of every lump touching the weigh frame."""
        frame, lump_of, w = self.frame, self.lump_of, self._w
        Fz = 0.
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in frame or g2 in frame:
                if lump_of.get(g2 if g1 in frame else g1) is None:
                    continue
                mujoco.mj_contactForce(self.model, data, i, w)
                f = con.frame.reshape(3, 3).T @ w[:3]            # force of geom1 on geom2, world frame
                Fz += f[2] if g1 in frame else -f[2]
        return Fz


class Weigher:
    """What the two placeholder devices share: the sampling, zero tracking, and the indicator that judges whether
    the reading is steady. A device decides in verdict() when its result stands."""
    kind = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.reading = 0.                   # N: this sample's reading
        self.tare, self.zero_off = 0., 0.   # the zero, and how long the empty reading has been off it
        self.samples = []                   # net readings of the weighing in hand
        self._sum, self._n = 0., 0          # load of the physics steps since the last sample

    # ---- fed by the run ----------------------------------------------------------------------------------
    def load_step(self, Fz):
        """One physics step's load (N) while the station weighs: averaged into the next sample."""
        self._sum += Fz
        self._n += 1

    def sample(self, Fz):
        """Close a 10 ms sample: the mean of the physics steps since the last one, else the load now (Fz)."""
        self.reading = self._sum / self._n if self._n else Fz
        self._sum, self._n = 0., 0
        return self.reading

    # ---- used by the station -----------------------------------------------------------------------------
    def track_zero(self):
        """The belt is seen empty: zero tracking. The zero follows the reading only within ZERO_BAND of it; a
        reading off the zero for ZERO_OFF_S means something lies on the scale: re-zero there, and zero_ok() will
        refuse the next weighing."""
        if abs(self.reading - self.tare) <= ZERO_BAND:
            self.tare += (self.reading - self.tare) / ZERO_N
            self.zero_off = 0.
        else:
            self.zero_off += SAMPLE_S
            if self.zero_off >= ZERO_OFF_S - 1e-9:
                self.tare, self.zero_off = self.reading, 0.

    def clear(self):
        """The scale was cleaned by hand (a held item taken off): it is clear again."""
        self.tare, self.zero_off = 0., 0.

    def begin(self):
        """A weighing starts (the belt is stopping under an item). Returns the zero it is read against, N."""
        self.samples = []
        return round(self.tare, 2)

    def zero_ok(self, tare):
        return abs(tare) <= TARE_TOL

    def read(self, t, rest, tare):
        """One sample of the item at rest since `rest`; see the module docstring."""
        self.samples.append(self.reading - tare)
        nf, ns = int(round(FILTER_S / SAMPLE_S)), int(round(STEADY_S / SAMPLE_S))
        F = np.array(self.samples)
        steady = False
        if len(F) >= nf + ns - 1:
            ma = np.convolve(F, np.ones(nf) / nf, 'valid')[-ns:]
            steady = ma.max() - ma.min() <= max(STEADY_MIN_N, STEADY_BAND * abs(ma.mean()))
        void = self.verdict(t, rest, steady)
        if void is None:
            return None
        win = F[-ns:]                                     # the result: the mean of the last STEADY_S
        out = dict(t_weigh_s=round(t, 3), weigh_wait_s=round(t - rest, 3), steady=bool(steady),
                   mass_kg=round(float(win.mean()) / G, 3), reading_std_N=round(float(win.std()), 2))
        if void:
            out['void'] = void
        return out

    def verdict(self, t, rest, steady):
        """None: no result yet; '': the result stands; a void reason: there is one and it must not be used."""
        raise NotImplementedError

    # ---- for result.json ---------------------------------------------------------------------------------
    def describe(self):
        return dict(weigh_model=self.kind, filter_s=FILTER_S, steady_s=STEADY_S, steady_band=STEADY_BAND,
                    max_wait_s=WEIGH_MAX_S, skirts_on_weigh_frame=True)

    def limits(self):
        """The device's part of the station's void rules."""
        return dict(tare_tol_N=TARE_TOL, weigh_max_s=WEIGH_MAX_S)


class FixedTimeWeigher(Weigher):
    """--weigh-model fixed (the default since 2026-10-05, user: 称重和测体积是固定耗时 1 s 的概念装置): the reading
    scan_s after the belt came to rest stands, whatever it does. Whether it was steady by the indicator's rule is
    recorded, not judged."""
    kind = 'fixed'

    def verdict(self, t, rest, steady):
        return '' if t >= rest + self.cfg['scan_s'] - 1e-9 else None


class SteadyWeigher(Weigher):
    """--weigh-model steady: the result stands once the indicator is steady; at rest for WEIGH_MAX_S without a
    steady reading, the item is void ('unsteady')."""
    kind = 'steady'

    def verdict(self, t, rest, steady):
        if steady:
            return ''
        return 'unsteady' if t - rest >= WEIGH_MAX_S - 1e-9 else None


WEIGHERS = {cls.kind: cls for cls in (FixedTimeWeigher, SteadyWeigher)}


def make_weigher(cfg):
    """The weighing device of --weigh-model."""
    return WEIGHERS[cfg['weigh_model']](cfg)
