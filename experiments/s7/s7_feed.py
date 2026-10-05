"""What the feed head did in the baseline batches (EXPERIMENTS.md S7): pure Python.

    python experiments/s7/s7_feed.py OUT_DIR [OUT_DIR ...] [--config baseline]

Per release (one run of the feed belt): how many lumps went, how far apart in time their centroids passed the head
edge, how long the first one took to cut the drop beam, and what became of the lumps that went together.
"""
import argparse
import glob
import gzip
import json
import os
from collections import Counter

from s7_analyze import audit, zh


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', nargs='+')
    p.add_argument('--config', default='baseline')
    a = p.parse_args()
    stops, sizes, tip, slow, empty, pairs = Counter(), Counter(), [], [], Counter(), []
    nb = jogs = alarms = early = 0
    for fp in sorted(fp for out in a.out for fp in glob.glob(os.path.join(out, a.config, '*.result.json.gz'))):
        with gzip.open(fp, 'rt', encoding='utf-8') as fh:
            r = json.load(fh)
        nb += 1
        tag = os.path.basename(fp).split('.')[0]
        L, _ = audit(r)
        fd = r['feeder']
        went = {}                                           # lump -> when its centroid passed the head edge
        for e in fd['events']:
            if e['event'] == 'released':
                for k in e['blocks']:
                    went.setdefault(k, e['t_s'])
        blocks = [t for t, e in r['perception']['beams']['beam_feed']['edges'] if e == 'block']
        alarms += bool(r['perception']['beams']['beam_feed']['alarm'])
        n_empty = 0
        for rel in fd['releases']:
            stops[rel['stop']] += 1
            jogs += rel['jogs']
            if not rel['members']:
                n_empty += 1
                continue
            sizes[len(rel['members'])] += 1
            cut = [t for t in blocks if t >= rel['t_first_s'] - 1e-9]
            if rel['stop'] == 'beam' and rel['t_stop_s'] < rel['t_first_s']:
                early += 1                                  # the beam was cut before any centroid was over the edge
            elif cut:
                (tip if rel['stop'] == 'beam' else slow).append(round(cut[0] - rel['t_first_s'], 2))
            if len(rel['members']) > 1:
                ts = sorted(went[k] for k in rel['members'])
                pairs.append(dict(tag=tag, lumps=rel['members'], gap_s=round(ts[1] - ts[0], 2),
                                  after_stop=bool(rel['after_stop']), stop=rel['stop'],
                                  fates=[L[k]['fate'] for k in rel['members']]))
        empty[n_empty] += 1
    n = sum(sizes.values())
    multi = [x for x in pairs]
    lost = [x for x in multi if any(f != 'measured' for f in x['fates'])]
    level = [x for x in multi if x['gap_s'] <= .10]
    print('%d 批，%d 次放料（另有 %d 次没有料过边的空放料，出在 %d 批里）' % (
        nb, n, sum(k * v for k, v in empty.items()), sum(v for k, v in empty.items() if k)))
    print('一次放下的块数：%s；一次放下不止一块的 %d 次（%.1f %%）' % (dict(sorted(sizes.items())), len(multi),
                                                         100 * len(multi) / max(1, n)))
    print('  其中第二块在头一块过边后 0.10 s 以内过边（3 mm 慢走行程，质心齐平）：%d 次；更晚：%d 次（中位 %.2f s，最晚 %.2f s）' % (
        len(level), len(multi) - len(level), q([x['gap_s'] for x in multi if x['gap_s'] > .10], .5),
        max([x['gap_s'] for x in multi] or [float('nan')])))
    print('  到秤上没拆开（这几块没有独立测成）的 %d 次：%s' % (
        len(lost), ['%s %s 间隔 %.2f s %s' % (zh(x['tag']), x['lumps'], x['gap_s'], x['fates']) for x in lost]))
    print('  没拆开的 %d 次里质心齐平（≤ 0.10 s）的 %d 次；拆开了的 %d 次里质心齐平的 %d 次' % (
        len(lost), sum(x['gap_s'] <= .10 for x in lost), len(multi) - len(lost),
        sum(x['gap_s'] <= .10 for x in multi if x not in lost)))
    print('停带原因：%s；点动 %d 次；S1 失效报警 %d 批' % (dict(stops), jogs, alarms))
    print('头一块质心过边到 S1 被挡：光束停带的 %d 次，中位 %.2f s，p90 %.2f s，最长 %.2f s；'
          '1 s 兜底停带的 %d 次，之后光束在质心过边后 %.2f–%.2f s 被挡（中位 %.2f s）；'
          '质心还没过边光束就被挡的 %d 次' % (
              len(tip), q(tip, .5), q(tip, .9), max(tip or [float('nan')]), len(slow), min(slow or [float('nan')]),
              max(slow or [float('nan')]), q(slow, .5), early))


if __name__ == '__main__':
    main()
