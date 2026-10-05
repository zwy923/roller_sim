"""Baseline and the feed head changes side by side, on the batches every configuration ran (S8): pure Python.

    python experiments/s8/s8_compare.py OUT_DIR [OUT_DIR ...] [--configs baseline,stop_past] [--seeds FIRST-LAST]
                                        [--list]

Reads <config>/<tag>.result.json.gz (experiments/s7/s7_run.py) and audits every batch with singulator.audit. Only the
tags that every listed configuration has a result for are compared (paired: same batch, same lumps, same layout).

  measured        lumps measured alone / lumps put in (the S7 measure; 95 % interval resampling batches);
  not measured    by cause: multi (two or more on the scale together), off (touching off the scale), other;
  manual          batches that needed a person (a void item held on the scale);
  releases        feed belt releases that put a lump over the edge; multi: those that put more than one over;
                  after_stop: lumps that went after the stop was commanded; level: multi-releases whose second lump
                  went within 0.10 s of the first (centroids level);
  stops           why releases ended (beam, went, beam_missed, ...); jogs; empty: releases that put nothing over;
  S1 alarm        batches the drop beam's own diagnosis stopped (dead / long block);
  clear           line clear time, mean over batches, and the paired mean difference to the first configuration;
  better / worse  batches with every lump measured alone in this configuration and not in the first, and back.
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from singulator.audit import BAD_END, audit  # noqa: E402

LEVEL_S = .10


def boot(pairs, n=4000, seed=20261006):
    rng = random.Random(seed)
    vals = sorted(sum(a for a, _ in S) / max(1, sum(b for _, b in S))
                  for S in ([rng.choice(pairs) for _ in pairs] for _ in range(n)))
    return vals[int(.025 * n)], vals[int(.975 * n) - 1]


def batch(r):
    L, b = audit(r)
    fd, tr = r['feeder'], r['transfer']
    tip = {x['lump']: x['t_tip'] for x in tr['lumps']}
    rels = [x for x in fd['releases'] if x['members']]
    multi = [x for x in rels if len(x['members']) > 1]
    level = sum(1 for x in multi if sorted(tip[k] for k in x['members'])[1] - min(tip[k] for k in x['members'])
                <= LEVEL_S + 1e-9)
    beam = r['perception']['beams']['beam_feed']
    return dict(lumps=L, cls=b['cls'], manual=b['manual'], unfinished=b['unfinished'], numerics=b['numerics_bad'],
                all=b['all_measured'], clear=r['outcome'].get('line_clear_s'), releases=len(rels),
                multi=len(multi), level=level, in_multi=sum(len(x['members']) for x in multi),
                after_stop=sum(len(x['after_stop']) for x in fd['releases']),
                empty=sum(1 for x in fd['releases'] if not x['members']),
                stops=Counter(x['stop'] for x in fd['releases']), jogs=sum(x['jogs'] for x in fd['releases']),
                s1=(beam.get('alarm') or {}).get('why'), holds=b['holds'],
                multi_tags=[sorted(x['members']) for x in multi])


def load(outs, configs, seeds):
    data = OrderedDict((c, {}) for c in configs)
    for out in outs:
        for c in configs:
            for fp in glob.glob(os.path.join(out, c, '*.result.json.gz')):
                tag = os.path.basename(fp)[:-len('.result.json.gz')]
                if seeds and not seeds[0] <= int(tag.rsplit('_', 1)[1]) <= seeds[1]:
                    continue
                with gzip.open(fp, 'rt', encoding='utf-8') as fh:
                    data[c][tag] = batch(json.load(fh))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', nargs='+')
    p.add_argument('--configs', required=True)
    p.add_argument('--seeds', metavar='FIRST-LAST')
    p.add_argument('--list', action='store_true', help='every lump not measured alone, per configuration')
    a = p.parse_args()
    configs = a.configs.split(',')
    seeds = tuple(int(x) for x in a.seeds.split('-')) if a.seeds else None
    data = load(a.out, configs, seeds)
    tags = sorted(set.intersection(*(set(d) for d in data.values())))
    print('%d batches with a result in every configuration (%s)' % (len(tags), ', '.join(
        '%s %d' % (c, len(d)) for c, d in data.items())))
    first = data[configs[0]]
    head = ('config', 'measured', 'rate (95 %)', 'multi', 'off', 'other', 'manual', 'unfin.', 'numer.', 'releases',
            'multi-rel', 'level', 'after_stop', 'jogs', 'empty', 'S1 alarm', 'clear s', 'Δclear', 'better', 'worse')
    print('| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head))
    for c, d in data.items():
        B = [d[t] for t in tags]
        F = Counter(f['fate'] for b in B for f in b['lumps'].values())
        n, ok = sum(F.values()), F['measured']
        lo, hi = boot([(sum(f['fate'] == 'measured' for f in b['lumps'].values()), len(b['lumps'])) for b in B])
        clear = [b['clear'] for b in B if b['clear'] is not None]
        dclear = [d[t]['clear'] - first[t]['clear'] for t in tags
                  if d[t]['clear'] is not None and first[t]['clear'] is not None]
        rel = sum(b['releases'] for b in B)
        row = (c, '%d / %d' % (ok, n), '%.1f %% (%.1f–%.1f)' % (100 * ok / n, 100 * lo, 100 * hi), F['multi'],
               F['touching'], n - ok - F['multi'] - F['touching'], sum(b['manual'] for b in B),
               sum(b['unfinished'] for b in B), sum(b['numerics'] for b in B), rel,
               '%d (%.1f %%)' % (sum(b['multi'] for b in B), 100 * sum(b['multi'] for b in B) / max(1, rel)),
               sum(b['level'] for b in B), sum(b['after_stop'] for b in B), sum(b['jogs'] for b in B),
               sum(b['empty'] for b in B), sum(bool(b['s1']) for b in B),
               '%.1f' % (sum(clear) / len(clear)) if clear else '-',
               '%+.1f' % (sum(dclear) / len(dclear)) if dclear else '-',
               sum(1 for t in tags if d[t]['all'] and not first[t]['all']),
               sum(1 for t in tags if first[t]['all'] and not d[t]['all']))
        print('| ' + ' | '.join(str(x) for x in row) + ' |')
    for c, d in data.items():
        B = [d[t] for t in tags]
        stops = sum((b['stops'] for b in B), Counter())
        ends = Counter(b['cls'] for b in B if b['cls'] in BAD_END)
        print('%s: stops %s; bad ends %s; S1 alarms %s' % (c, dict(stops), dict(ends) or 'none',
                                                          dict(Counter(b['s1'] for b in B if b['s1'])) or 'none'))
        if a.list:
            for t in tags:
                bad = {k: f['fate'] for k, f in d[t]['lumps'].items() if f['fate'] != 'measured'}
                if bad:
                    print('   %s %s  (multi-releases %s)' % (t, bad, d[t]['multi_tags']))


if __name__ == '__main__':
    main()
