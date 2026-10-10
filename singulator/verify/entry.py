"""Truth-only lane-entrance audit. A geometric diagnostic, never evidence of independent weighing."""
from ..physics.lumps import exit_cut

MIN_GAP_S = .25  # engineering screen: 0.10 m at nominal 0.40 m/s; not an industrial acceptance standard


class EntryWatch:
    def __init__(self, d, lumps):
        self.x, self.y0, self.y1 = float(d['lane_out_start']), float(d['report']['lane_y_m']), float(d['lane_top'])
        self.lumps = lumps
        self.rec = {L.k: dict(lump=L.k, front_s=None, tail_s=None, overlap=False, outside_lane=False) for L in lumps}
        self.together_s, self.last_t = 0., 0.

    def observe(self, t):
        live = []
        for L in self.lumps:
            r = self.rec[L.k]
            if L.state in ('dropped', 'taken_off') or not hasattr(L, 'V'):
                continue
            x0, x1 = L.box[0][0], L.box[1][0]
            cut = exit_cut(L.V, L.edges, self.x, (x0, x1))
            if cut is not None:
                if r['front_s'] is None:
                    r['front_s'] = round(t, 4)
                live.append(L.k)
                if cut[0] < self.y0 - .02 or cut[1] > self.y1 + .02:
                    r['outside_lane'] = True
            if r['front_s'] is not None and r['tail_s'] is None and x0 > self.x:
                r['tail_s'] = round(t, 4)
        if len(live) > 1:
            self.together_s += t - self.last_t
            for k in live:
                self.rec[k]['overlap'] = True
        self.last_t = t

    def report(self):
        rows = [dict(r) for r in self.rec.values()]
        ordered = sorted((r for r in rows if r['front_s'] is not None), key=lambda r: (r['front_s'], r['lump']))
        previous = None
        for r in ordered:
            r['gap_before_s'] = (None if previous is None or previous['tail_s'] is None else
                                 round(r['front_s'] - previous['tail_s'], 4))
            r['gap_ok'] = previous is None or r['gap_before_s'] is not None and r['gap_before_s'] >= MIN_GAP_S
            previous = r
        for r in rows:
            r['qualified'] = bool(r['tail_s'] is not None and not r['overlap'] and not r['outside_lane']
                                  and r.get('gap_ok', False))
        complete = all(r['tail_s'] is not None for r in rows)
        return dict(plane_x_m=self.x, minimum_net_time_gap_s=MIN_GAP_S, sample_uncertainty_s=.02,
                    input=len(rows), completed=sum(r['tail_s'] is not None for r in rows),
                    qualified=sum(r['qualified'] for r in rows), overlapping=sum(r['overlap'] for r in rows),
                    together_s=round(self.together_s, 4),
                    clear_s=max(r['tail_s'] for r in rows) if complete else None, lumps=rows,
                    definition='whole block crosses lane entrance, no other block occupies that plane during '
                               'its crossing, cross-section within lane (+/-20 mm), at least 0.25 s after '
                               'previous tail; first block has no preceding-gap requirement; all inputs counted',
                    note='geometric entrance screening only, not independent measurement acceptance')
