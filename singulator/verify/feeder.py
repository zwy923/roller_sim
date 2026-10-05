"""Which true lumps went over the feed belt's head edge, release by release.

With the ideal view (--feeder-sensing oracle, the default) the feed belt control knows each lump by its id, and its
own record says which ones a release let go ('members', 'after_stop'). With the cameras it has objects without
identity -- a release only knows that something went. FeederWitness then keeps the truth next to it: the lumps whose
true centroid passed the edge in each release ('lumps'), which of them after its stop ('lumps_after_stop'), and
which of those only once the belt had been run again for staging ('lumps_in_staging': carried over with no release
decided, not by the release's own stopping distance). Either way it gives the rest of the run one answer to "which
lumps have gone" (went, waiting) and completes the feeder's report. It feeds nothing back.
"""


class FeederWitness:
    def __init__(self, feeder, n):
        """n: lumps in the batch."""
        self.f, self.n = feeder, n
        self.gone = set()                               # camera view: true lumps over the edge

    def observe(self, t, com_x):
        """Every sample the feed belt control runs. com_x: {lump: true centroid x} of the lumps still on the
        machine."""
        f = self.f
        if not f.vision:
            return                                      # the controller's own record is by lump
        for k, x in com_x.items():
            if k not in self.gone and x > f.g['x1'] and f.releases:
                rel = f.releases[-1]
                self.gone.add(k)
                rel.setdefault('lumps', []).append(k)
                if rel['t_stop_s'] is not None:
                    rel.setdefault('lumps_after_stop', []).append(k)
                    if f.staged:
                        rel.setdefault('lumps_in_staging', []).append(k)

    @property
    def went(self):
        """The lumps that have gone over the edge so far."""
        return self.gone if self.f.vision else self.f.released

    def waiting(self, k, state, cx):
        """Is lump k waiting on the feed belt (for the stall bookkeeping of the lumps' record only): not gone yet,
        its true centroid behind the edge."""
        return state == 'on_belt' and k not in self.went and cx <= self.f.g['x1']

    def report(self, feeder_report):
        """Add which lumps went to the feeder's report (in place; returns it). Until 2026-10-04 a run on the
        cameras listed every lump as never released: the report read the ideal view's record."""
        f = self.f
        key = 'lumps' if f.vision else 'members'
        sizes = [len(r[key]) for r in f.releases if r.get(key)]
        went = [k for r in f.releases for k in r.get('lumps', [])] if f.vision else list(f.released)
        after_stop = (sum(len(r.get('lumps_after_stop', [])) for r in f.releases) if f.vision
                      else f.counts['after_stop'])
        feeder_report.update(released_order=went, never_released=[k for k in range(self.n) if k not in went])
        feeder_report['counts'].update(
            after_stop=after_stop, in_staging=sum(len(r.get('lumps_in_staging', [])) for r in f.releases),
            releases=len(sizes), single=sum(s == 1 for s in sizes),
            lumps_per_release={str(s): sizes.count(s) for s in sorted(set(sizes))})
        return feeder_report
