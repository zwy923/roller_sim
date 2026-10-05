"""The station's measurements against the true state: what a real line could not know about itself.

The station (control/station.py) decides on sensor signals only. StationWitness stands next to it with the truth --
which lumps really loaded the weigh frame while an item was weighed, what else they touched, what they really
weigh and are made of, where they really landed -- and writes that into the station's record (the keys of an item
that start with 'truth', and material, ideal_route, route_as_ideal, bins, bins_match_route, t_landed_s), from which
the verification in result.json is counted:

    false_valid         an item sorted as valid that was not one true lump, weighed alone, within 2 %
    false_void          an item held although it was (scan failures apart: a scan can fail on a good lump)
    landed_unmeasured   lumps in a bin without a valid single-lump item discharged

The station calls the witness at fixed points of its sample (seen, landed, at_rest, decided, routed, revoked); with
control.station.NoWitness in its place it runs the same, and its record simply lacks the truth. Nothing here feeds
back: no decision of the station reads what the witness writes.
"""


def frame_contacts(data, frame, lump_of):
    """Who loads the weigh frame now, and what else every lump touches: (on, other). on: the lumps in contact with
    a geom of `frame`; other: {lump: {other lumps by id, 'equipment' for anything else off the frame}}.
    lump_of: {geom id: lump}."""
    on, other = set(), {}
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        k1, k2 = lump_of.get(g1), lump_of.get(g2)
        if g1 in frame or g2 in frame:
            k = k2 if g1 in frame else k1
            if k is not None:
                on.add(k)
            continue
        for k, ok in ((k1, k2), (k2, k1)):
            if k is not None:
                other.setdefault(k, set()).add(ok if ok is not None else 'equipment')
    return on, other


class StationWitness:
    def __init__(self, cfg, blocks):
        """blocks: the batch (lumps.make_blocks), with the compiled masses."""
        self.cfg, self.blocks = cfg, blocks
        self.station = None
        self.states, self.bins = [], []         # every lump's state and bin, this sample
        self.on, self.other = set(), {}         # frame_contacts() of this sample
        self.taken = set()                      # lumps the station has had in hand (seen past the head edge)

    def attach(self, station):
        """Stand next to this station: it calls the hooks below from now on."""
        self.station, station.witness = station, self
        return self

    def truth(self, states, bins, on, other):
        """Every sample, before the station observes: the lumps' states and bins, and frame_contacts()."""
        self.states, self.bins, self.on, self.other = states, bins, on, other

    def waiting(self, k, state, cx):
        """Is lump k waiting for the station (for the stall bookkeeping of the lumps' record only): one it has
        taken, or any lump upstream of the head edge while it holds the section."""
        st = self.station
        return state == 'on_belt' and (k in self.taken or (st.section_held or st.u < 1.)
                                       and cx <= st.g['buffer']['x0'])

    # ---- the hooks, in the order the station reaches them in a sample ------------------------------------
    def seen(self, objs):
        """The objects past the head edge this sample."""
        for o in objs.values():
            self.taken.update(o['_truth'])

    def landed(self, t, sent):
        """After the plate-path check: has everything of a sent item landed, and in which bin?"""
        for it in sent:
            lumps = it.get('truth_lumps', [])
            if 't_landed_s' not in it and lumps and self.states and all(self.states[k] in ('sorted', 'dropped')
                                                                       for k in lumps):
                bins = {str(k): self.bins[k] for k in lumps}
                it.update(t_landed_s=round(t, 3), bins=bins,
                          bins_match_route=all(b == it['route'] for b in bins.values()))
                self.station._event(t, 'landed', item=it['item'], bins=bins)

    def at_rest(self, it):
        """Every sample an item is at rest on the measuring belt: what physically loads the weigh frame, and what
        else it touches."""
        on, other = self.on, self.other
        closure, stack = set(on), list(on)
        while stack:                                  # lumps leaning on the lumps that load the frame
            for x in other.get(stack.pop(), ()):
                if not isinstance(x, str) and x not in closure:
                    closure.add(x)
                    stack.append(x)
        iso = closure <= set(on) and all('equipment' not in other.get(k, ()) for k in closure)
        it['_on'] = it.get('_on', set()) | closure
        it['truth_isolated'] = it.get('truth_isolated', True) and iso

    def decided(self, it):
        """The item has its mass and its scan: what it truly was."""
        thr = self.cfg['sort_density']
        lumps = sorted(it.pop('_on', set()))              # the lumps on (or leaning on) the weigh frame
        true = sum(self.blocks[k]['mass_kg'] for k in lumps)
        it.update(truth_lumps=lumps, truth_single=len(lumps) == 1, truth_mass_kg=round(true, 3),
                  mass_error_pct=round(100. * (it['mass_kg'] - true) / true, 3) if true else None,
                  truth_density_kg_m3=[round(self.blocks[k]['density'], 1) for k in lumps],
                  material=[self.blocks[k]['material'] for k in lumps],
                  ideal_route=['gangue' if self.blocks[k]['density'] >= thr else 'coal' for k in lumps])

    def routed(self, it):
        """The item was found good and has its route."""
        it['route_as_ideal'] = all(r == it['route'] for r in it['ideal_route'])

    def revoked(self, it):
        """A route was taken back (the item became void while it waited for the plate)."""
        it.pop('route_as_ideal', None)

    # ---- result.json -------------------------------------------------------------------------------------
    def report(self, station_report):
        """Add the verification to the station's report (in place; returns it)."""
        items = self.station.items
        done = [it for it in items if 't_decided_s' in it]
        valid = [it for it in done if not it['void']]
        # a valid item must be one true lump, weighed alone, within 2 % (true state); and no lump may land in a bin
        # without such an item (sorted unmeasured)
        measured = {k for it in valid if it.get('truth_single') and 't_discharge_s' in it for k in it['truth_lumps']}
        unmeasured = [k for k, st in enumerate(self.states) if st == 'sorted' and k not in measured]
        false_valid = [it['item'] for it in valid if not it['truth_single'] or not it.get('truth_isolated', True)
                       or it['mass_error_pct'] is None or abs(it['mass_error_pct']) > 2.]
        false_void = [it['item'] for it in done if it['void'] and it['truth_single'] and it.get('truth_isolated', True)
                      and it.get('mass_error_pct') is not None and abs(it['mass_error_pct']) <= 2.
                      and 'scan' not in it['reasons']]
        station_report['counts'].update(
            route_not_as_ideal=sum(not it['route_as_ideal'] for it in valid),
            bins_not_as_route=sum(it.get('bins_match_route') is False for it in items),
            landed=sum('t_landed_s' in it for it in items))
        station_report.update(
            verification=dict(false_valid=false_valid, false_void=false_void, landed_unmeasured=unmeasured,
                              rule='valid = one true lump, touching nothing off the weigh frame, weighed '
                                   'within 2 %; false_void excludes scan failures (a scan can fail on a '
                                   'good lump); landed_unmeasured = lumps in a bin without a valid '
                                   'single-lump item discharged'),
            mass_error_pct_abs_max=max((abs(it['mass_error_pct']) for it in valid
                                        if it['mass_error_pct'] is not None), default=None))
        return station_report
