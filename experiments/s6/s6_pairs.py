"""What became of two lumps that left the feed belt in one release (EXPERIMENTS.md S6): pure Python.

    python experiments/s6/s6_pairs.py OUT_DIR [OUT_DIR ...] [--only a,b] [--compare a:b ...] [--alias a=b ...]
                                       [--seeds FIRST-LAST] [--list]

Reads the full results (<tag>.result.json.gz). For every release of exactly two lumps: how long both were across the
main belt's head edge at once (outcome.together_pairs_s; 0 = one after the other), the time between their fronts
there, and whether each was then measured alone (apart), both were held as one void item (double), or something
else happened to one of them (other).

Per configuration: the pairs, how many ended apart / as a double, and the gap the speed difference should leave
between the two (gap_m below) -- the buffer belt parts a pair when the second lump reaches it late enough for the
first to have run GROUP_GAP ahead. The tables: by overlap time and by that gap, per buffer belt speed (--only picks
the configurations pooled in them; the same pair seen under two configurations counts twice).
--compare a:b: the same pair (same batch, same two lumps released together) under configuration a against b --
the difference of the gap, with a bootstrap interval. This asks what a structure does to the stagger of a pair,
on every pair, not only on the few that end as a double.
Several OUT_DIRs are read as one; --alias a=b reads the folder of configuration a under the name b (a re-run kept
under another name); --seeds keeps only the batches of these seeds.
"""
import argparse
import glob
import gzip
import json
import os
import random
from collections import OrderedDict

GROUP_GAP = .10         # m: singulator.station.GROUP_GAP -- lumps this far apart on the buffer belt are two items
BINS = ((0., 0., '先后过（不重叠）'), (1e-9, .2, '重叠 ≤ 0.2 s'), (.2, .4, '0.2–0.4 s'), (.4, .6, '0.4–0.6 s'),
        (.6, 1e9, '> 0.6 s'))
GAPS = ((-1e9, -.10, '< −0.10 m'), (-.10, 0., '−0.10–0 m'), (0., GROUP_GAP, '0–0.10 m'),
        (GROUP_GAP, .20, '0.10–0.20 m'), (.20, 1e9, '≥ 0.20 m'))


def gap_m(v, vb, fa, ta, fb, tb):
    """The clear gap (m, along the line) the buffer belt's speed leaves between two lumps: a is the one whose front
    crosses the head edge first. f, t: when a lump's front and tail cross the edge (s); v, vb: main and buffer belt
    speed. On the main belt the second one's front is v * (fb - fa) behind the first one's front, the first one is
    v * (ta - fa) long: the gap before the edge is v * (fb - ta) (negative: side by side over that length). Each
    lump takes the buffer belt's speed when its centroid tips over the edge, about halfway between its front and
    its tail crossing; the first runs at vb for that much longer and gains (vb - v) times it."""
    lead = (fb + tb) / 2 - (fa + ta) / 2            # the second one's centroid crosses this much later
    return v * (fb - ta) + (vb - v) * lead


def pairs(r):
    """One record per two-lump release of a full result."""
    o, st, B, c = r['outcome'], r['station'], r['blocks'], r['config']
    alone = {k for it in st['items'] if it.get('void') is False and it.get('truth_single') for k in it['truth_lumps']}
    double = [set(it['truth_lumps']) for it in st['items'] if it.get('void') and it.get('truth_single') is False]
    out = []
    for x in r['transfer']['releases']:
        if len(x['lumps']) != 2:
            continue
        # the first of the two: the one whose front crosses the head edge first; fronts in the same 10 ms sample: the
        # one whose tail crosses first (its centroid tips over first, it is the one the buffer belt takes first)
        when = lambda k: tuple(y for key in ('front_at_plane_s', 'tail_past_plane_s')
                               for y in (B[k].get(key) is None, B[k].get(key) or 0.))
        a, b = sorted(x['lumps'], key=when)
        T = [B[k].get(key) for k in (a, b) for key in ('front_at_plane_s', 'tail_past_plane_s')]
        kind = ('apart' if a in alone and b in alone else 'double' if any({a, b} <= d for d in double) else 'other')
        key = '%d-%d' % (min(a, b), max(a, b))
        out.append(dict(lumps=(min(a, b), max(a, b)), first=a,
                        overlap_s=float((o.get('together_pairs_s') or {}).get(key, 0.)),
                        stagger_s=None if None in (T[0], T[2]) else round(T[2] - T[0], 2), kind=kind,
                        gap_m=None if None in T else round(gap_m(c['v_belt'], c['buffer_speed'], *T), 3),
                        tip_spread_s=x.get('tip_spread_s'), late=bool(x.get('after_stop') or x.get('in_staging')),
                        touched=tuple(bool(B[k].get('touched_plough')) for k in (min(a, b), max(a, b))),
                        families=(B[a]['family'], B[b]['family'])))
    return out


def boot(xs, n=4000, seed=20261004):
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return m, means[int(.025 * n)], means[int(.975 * n) - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', nargs='+')
    p.add_argument('--only', help='comma-separated configurations pooled in the two tables (default: all)')
    p.add_argument('--compare', action='append', default=[], metavar='A:B', help='the same pairs under A and B')
    p.add_argument('--alias', action='append', default=[], metavar='A=B', help='read configuration A as B')
    p.add_argument('--seeds', metavar='FIRST-LAST', help='only the batches of these seeds')
    p.add_argument('--list', action='store_true', help='every pair, one per line')
    a = p.parse_args()
    only = set(a.only.split(',')) if a.only else None
    alias = dict(x.split('=') for x in a.alias)
    seeds = tuple(int(x) for x in a.seeds.split('-')) if a.seeds else None
    cfgs, speed, three = OrderedDict(), {}, 0           # config -> {(tag, lumps): record}; config -> buffer speed
    for fp in sorted(fp for out in a.out for fp in glob.glob(os.path.join(out, '*', '*.result.json.gz'))):
        cfg = os.path.basename(os.path.dirname(fp))
        cfg = alias.get(cfg, cfg)
        if seeds and not seeds[0] <= int(os.path.basename(fp).split('.')[0].rsplit('_', 1)[1]) <= seeds[1]:
            continue
        with gzip.open(fp, 'rt', encoding='utf-8') as fh:
            r = json.load(fh)
        speed[cfg] = r['config']['buffer_speed']
        three += sum(len(x['lumps']) > 2 for x in r['transfer']['releases']) * (not only or cfg in only)
        for q in pairs(r):
            q.update(config=cfg, tag=os.path.basename(fp).split('.')[0])
            cfgs.setdefault(cfg, OrderedDict())[(q['tag'], q['lumps'])] = q
            if a.list and (not only or cfg in only):
                print(cfg, q['tag'], q['lumps'], 'overlap %.2f' % q['overlap_s'], 'stagger', q['stagger_s'],
                      'gap', q['gap_m'], q['kind'], 'tips', q['tip_spread_s'], 'late' if q['late'] else '',
                      q['families'], 'touched the face', q['touched'])
    n = lambda Q, kind: sum(q['kind'] == kind for q in Q)
    print('| 配置 | 缓冲带 m/s | 两块一起放下（对） | 各自单独测成 | 一起作废（真双块） | 其他 | 算出的间距 < 0.10 m（对） '
          '| 按间距判对 |')
    print('|---|---|---|---|---|---|---|---|')
    for cfg, P in cfgs.items():
        Q = list(P.values())
        G = [q for q in Q if q['gap_m'] is not None and q['kind'] != 'other']
        hit = sum((q['gap_m'] >= GROUP_GAP) == (q['kind'] == 'apart') for q in G)
        print('| %s | %.1f | %d | %d | %d | %d | %d | %d / %d |' % (
            cfg, speed[cfg], len(Q), n(Q, 'apart'), n(Q, 'double'), n(Q, 'other'),
            sum(q['gap_m'] is not None and q['gap_m'] < GROUP_GAP for q in Q), hit, len(G)))
    by = OrderedDict()
    for cfg, P in cfgs.items():
        if not only or cfg in only:
            by.setdefault(speed[cfg], []).extend(P.values())
    for head, bins, value in (('过主带机头时', BINS, lambda q: q['overlap_s']), ('按速差算出的间距', GAPS, lambda q: q['gap_m'])):
        print('\n| 缓冲带速度 | %s | 对数 | 各自单独测成 | 一起作废（真双块） | 其他 |' % head)
        print('|---|---|---|---|---|---|')
        for vb in sorted(by):
            for lo, hi, name in bins:
                Q = [q for q in by[vb] if value(q) is not None
                     and (lo <= value(q) <= hi if bins is BINS else lo <= value(q) < hi)]
                if Q:
                    print('| %.1f m/s | %s | %d | %d | %d | %d |' % (vb, name, len(Q), n(Q, 'apart'), n(Q, 'double'),
                                                                   n(Q, 'other')))
    print('\n三块以上一起放下的放料：%d 次（不在表里）' % three)
    if a.compare:
        print('\n| 比较（同一对料：a 对 b） | 对数 | 间距之差均值 m (95 % CI) | 中位 m | 间距 < 0.10 m 的对：a / b '
              '| 真双块：a / b |')
        print('|---|---|---|---|---|---|')
        for item in a.compare:
            x, y = item.split(':')
            keys = [k for k in cfgs.get(x, {}) if k in cfgs.get(y, {})
                    and cfgs[x][k]['gap_m'] is not None and cfgs[y][k]['gap_m'] is not None]
            if not keys:
                continue
            d = sorted(cfgs[x][k]['gap_m'] - cfgs[y][k]['gap_m'] for k in keys)
            m, lo, hi = boot(d)
            med = d[len(d) // 2] if len(d) % 2 else (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2
            print('| %s 对 %s | %d | %+.3f (%+.3f, %+.3f) | %+.3f | %d / %d | %d / %d |' % (
                x, y, len(keys), m, lo, hi, med,
                sum(cfgs[x][k]['gap_m'] < GROUP_GAP for k in keys), sum(cfgs[y][k]['gap_m'] < GROUP_GAP for k in keys),
                sum(cfgs[x][k]['kind'] == 'double' for k in keys), sum(cfgs[y][k]['kind'] == 'double' for k in keys)))


if __name__ == '__main__':
    main()
