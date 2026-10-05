"""Tables for EXPERIMENTS.md S6 from the batch summaries of s6_run.py (pure Python: no numpy needed).

    python experiments/s6/s6_analyze.py OUT_DIR [OUT_DIR ...] [--base base] [--pair a:b ...] [--only a,b]
                                         [--seeds FIRST-LAST] [--screened] [--alias a=b ...] [--json FILE]
                                         [--collect FILE]

Per configuration: how many batches needed a person (a void item held on the scale, a jam, a fault, a lump off the
line) and why -- the true state says whether a void item really was more than one lump --, what the feed head let
go together and what became of it, how steady the feed belt control was, the time to clear the line; and the paired
difference to the base configuration on the batches both ran (exact McNemar test on 'the batch needed a person',
bootstrap interval on the time and on the lumps measured); --pair a:b adds the comparison of a against b. Several
OUT_DIRs are read as one (the same batch of a configuration in two of them is an error). --screened leaves out the
batches that did not pass the model's own numerical screen (deepest penetration over the limit), to see what the
numbers owe to them. --alias a=b reads configuration a under the name b (a re-run kept under another name, to pool
it with the batches of b from another OUT_DIR).
"""
import argparse
import glob
import json
import math
import os
import random
from collections import Counter, OrderedDict

LAYOUTS = ('scatter', 'aligned', 'touching', 'oblique')
LAYOUT_ZH = dict(scatter='随机', aligned='并齐', touching='相贴', oblique='斜放')
BAD_END = ('jammed', 'station_fault', 'face_fault', 'incomplete', 'numerical_failure')


# ---- statistics --------------------------------------------------------------------------------------
def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k, n, alpha=.05):
    """Exact two-sided interval for a proportion k / n."""
    if n == 0:
        return (float('nan'), float('nan'))

    def solve(f, lo, hi):
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0. if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < alpha / 2, 0., 1.)
    upper = 1. if k == n else solve(lambda p: binom_cdf(k, n, p) > alpha / 2, 0., 1.)
    return lower, upper


def mcnemar(b, c):
    """Exact two-sided McNemar test on the discordant pairs (b: only the first, c: only the second)."""
    n = b + c
    if n == 0:
        return 1.
    return min(1., 2 * binom_cdf(min(b, c), n, .5))


def boot_mean(xs, n=4000, seed=20261004):
    """Mean and its 95 % bootstrap interval (resampling batches)."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return m, means[int(.025 * n)], means[int(.975 * n) - 1]


def quantile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def median(xs):
    return quantile(xs, .5)


# ---- one batch ---------------------------------------------------------------------------------------
def void_kind(it):
    """Why a void item stopped the line, by the true state: 'double' (more than one lump on the scale),
    'leaning' (one lump, touching something off the scale), or a single lump weighed alone -- then by the
    controller's own reasons."""
    if it.get('truth_single') is False:
        return 'double'
    if it.get('truth_isolated') is False:
        return 'leaning'
    return 'single:' + '+'.join(sorted(it.get('reasons') or ['?']))


MECH_ZH = OrderedDict([
    ('double', '真多块（两块一起上秤）'), ('leaning', '单块但靠着秤外的东西'),
    ('count', '跟踪记数虚高（单块被记成多块）'), ('lineage', '跟踪丢失血缘（单块）'),
    ('rocking', '圆料停稳后还在晃，读数超时'), ('nose', '下一块料停在 S3 后鼻子伸过接缝（扫描区里两个对象）'),
    ('dragged', '下一块料停在 S3 后伸过接缝，被计量带带到机头'), ('near', '别的料靠到了 5 cm 以内'),
    ('joint', '料停稳后轮廓出了称重区（滚回接缝或滚到前端）'),
    ('handover', 'S2 遮挡次数和相机看到的对象数对不上'), ('other', '其他')])


def mechanism(it):
    """What physically (or in the tracker) made a void item void: a key of MECH_ZH."""
    if it.get('truth_single') is False:
        return 'double'
    if it.get('truth_isolated') is False:
        return 'leaning'
    why, info = set(it.get('reasons') or []), it.get('reason_info') or {}
    detail = lambda k: (info.get(k) or {}).get('detail', '')
    if 'track_lost' in why:
        return 'lineage'
    if 'outside' in why and detail('outside').startswith('front at the measuring belt head'):
        return 'dragged'
    if 'outside' in why and detail('outside').startswith('another object'):
        return 'near'
    if 'scan' in why and 'objects_in_zone' in detail('scan'):
        return 'nose'
    if 'outside' in why:
        return 'joint'
    if 'multi' in why and (it.get('n_obj_max') or 1) <= 1 and detail('multi').startswith('objects'):
        return 'count'
    if 'unsteady' in why:
        return 'rocking'
    if 'handover' in why:
        return 'handover'
    return 'other'


def batch(s):
    """Derived record of one batch summary."""
    if 'error' in s:
        return dict(error=s['error'])
    voids = [it for it in s['items'] if it.get('void')]
    kinds = [void_kind(it) for it in voids]
    rel = s['releases']
    of = {k: i for i, r in enumerate(rel) for k in r['lumps']}             # lump -> the release it went in
    multi = [r for r in rel if len(r['lumps']) > 1]
    alone = {k for it in s['items'] if it.get('void') is False and it.get('truth_single')
             for k in it['truth_lumps']}                                   # lumps measured as one valid item each
    # a void item of several lumps: did they leave the feed belt in one release, or one catch the other up later?
    doubles = [it for it in voids if it.get('truth_single') is False]
    same = sum(len({of.get(k) for k in it['truth_lumps']}) == 1 for it in doubles)
    bad_end = s['classification'] in BAD_END
    return dict(layout=s['layout'], seed=s['seed'], lumps=s['count'], cls=s['classification'],
                bad_end=bad_end, dropped=s['dropped'],
                stop=s['holds'] > 0, holds=s['holds'], kinds=kinds,
                person=s['holds'] > 0 or bad_end or s['dropped'] > 0,
                reasons=['%s|%s' % ('+'.join(sorted(it.get('reasons') or ['?'])),
                                    'double' if it.get('truth_single') is False else 'single') for it in voids],
                mech=[mechanism(it) for it in voids],
                stop_double=any(k in ('double', 'leaning') for k in kinds),
                stop_single=any(k.startswith('single') for k in kinds),
                doubles=len(doubles), doubles_same_release=same,
                taken_off=len(s['taken_off']), valid=s['valid'], landed=s['landed'], alone=len(alone),
                false_valid=len(s['false_valid']), unmeasured=len(s['landed_unmeasured']),
                misrouted=s['route_not_as_ideal'], void_discharged=s['void_discharged'],
                retracts=s['unjam'], fault=s['fault'], jam=(s['jam_stop'] or {}).get('reason'),
                releases=len(rel), multi=len(multi),
                multi_apart=sum(all(k in alone for k in r['lumps']) for r in multi),
                in_staging=sum(len(r.get('in_staging') or []) for r in rel),
                after_stop=sum(len(r.get('after_stop') or []) for r in rel),
                funnel2_s=s['funnel']['two_or_more_s'], funnel_max=s['funnel']['max_lumps'],
                abreast_s=s['together_at_cut_s'], clear_s=s['line_clear_s'], end_s=s['end_s'],
                unsure=s['feeder_counts'].get('unsure_waits', 0), stagings=s['feeder_counts'].get('stagings', 0),
                jogs=s['feeder_counts'].get('jogs', 0), beam_missed=s['feeder_counts'].get('beam_missed', 0),
                section_holds=s['station_counts']['section_holds'], splits=s['station_counts']['splits'],
                buffer_stops=s['station_counts']['buffer_stops'], held_s=s['upstream_held_s'],
                weigh_err=s['mass_error_max_pct'], numerics_ok=s['numerics_ok'], wall_s=s['wall_s'],
                merges=s['vision'].get('merges'), track_splits=s['vision'].get('splits'), lost=s['vision'].get('lost'),
                n_est_over=sum((it.get('n_est_max') or 1) > len(it['truth_lumps']) for it in s['items']
                               if it.get('truth_lumps')),
                sha=s.get('source_sha'), platform=s.get('platform'),
                signature=(s['classification'], s['holds'], s['line_clear_s'],
                           tuple(tuple(r['lumps']) for r in rel)))


def load(outs, only=None, seeds=None, screened=False, alias=None):
    """{config: {tag: derived record}} and the raw summaries, over every OUT_DIR."""
    data, raw = OrderedDict(), OrderedDict()
    for out in outs:
        for fp in sorted(glob.glob(os.path.join(out, '*', '*.json'))):
            with open(fp, encoding='utf-8') as fh:
                s = json.load(fh)
            s['config'] = (alias or {}).get(s['config'], s['config'])
            if only and s['config'] not in only:
                continue
            if seeds and not seeds[0] <= int(s['tag'].rsplit('_', 1)[1]) <= seeds[1]:
                continue
            if screened and s.get('numerics_ok') is False:
                continue
            if s['tag'] in data.get(s['config'], {}):
                raise SystemExit('%s/%s is in more than one OUT_DIR' % (s['config'], s['tag']))
            data.setdefault(s['config'], {})[s['tag']] = batch(s)
            raw.setdefault(s['config'], {})[s['tag']] = s
    return data, raw


# ---- per configuration -------------------------------------------------------------------------------
def aggregate(B):
    B = [b for b in B if 'error' not in b]
    n = len(B)
    if not n:
        return dict(batches=0)
    lumps = sum(b['lumps'] for b in B)
    total = lambda key: sum(b[key] for b in B)
    stops, person = total('stop'), total('person')
    kinds = Counter(k for b in B for k in b['kinds'])
    rel, multi = total('releases'), total('multi')
    clear = [b['clear_s'] for b in B if b['clear_s'] is not None]
    m, lo, hi = boot_mean(clear)
    return dict(batches=n, lumps=lumps,
                person_batches=person, person_rate=person / n, person_ci=clopper_pearson(person, n),
                stop_batches=stops, stop_rate=stops / n, stop_ci=clopper_pearson(stops, n),
                holds=total('holds'), void_kinds=dict(kinds),
                void_reasons=dict(Counter(r for b in B for r in b['reasons'])),
                mech=dict(Counter(m for b in B for m in b['mech'])),
                stop_double_batches=total('stop_double'), stop_single_batches=total('stop_single'),
                void_double=kinds.get('double', 0) + kinds.get('leaning', 0),
                void_single=sum(v for k, v in kinds.items() if k.startswith('single')),
                doubles=total('doubles'), doubles_same_release=total('doubles_same_release'),
                bad_end_batches=total('bad_end'), bad_ends=dict(Counter(b['cls'] for b in B if b['bad_end'])),
                faults=dict(Counter(b['fault'] or b['jam'] for b in B if b['fault'] or b['jam'])),
                dropped_lumps=total('dropped'),
                retract_batches=sum(b['retracts'] > 0 for b in B), retracts=total('retracts'),
                taken_off=total('taken_off'), valid=total('valid'), alone=total('alone'),
                alone_share=total('alone') / lumps,
                false_valid=total('false_valid'), unmeasured=total('unmeasured'),
                misrouted=total('misrouted'), void_discharged=total('void_discharged'),
                releases=rel, multi=multi, multi_rate=multi / rel if rel else None,
                multi_ci=clopper_pearson(multi, rel) if rel else None, multi_apart=total('multi_apart'),
                in_staging=total('in_staging'), after_stop=total('after_stop'),
                funnel2_batches=sum(b['funnel2_s'] > 0 for b in B), funnel2_s=total('funnel2_s'),
                abreast_batches=sum(b['abreast_s'] > 0 for b in B),
                clear_mean=m, clear_ci=(lo, hi), clear_median=median(clear), clear_p90=quantile(clear, .9),
                clear_max=max(clear, default=None), clear_n=len(clear),
                unsure_s=total('unsure') * .01, stagings=total('stagings'), jogs=total('jogs'),
                beam_missed=total('beam_missed'), section_holds=total('section_holds'), splits=total('splits'),
                buffer_stops=total('buffer_stops'), held_s=total('held_s'), n_est_over=total('n_est_over'),
                weigh_err_max=max((b['weigh_err'] for b in B if b['weigh_err'] is not None), default=None),
                numerics_bad=sum(not b['numerics_ok'] for b in B),
                merges=sum(b['merges'] or 0 for b in B), wall_mean_s=total('wall_s') / n,
                sha=sorted({str(b['sha'])[:12] for b in B}), platform=sorted({str(b['platform']) for b in B}))


def paired(A, Bc):
    """Configuration A against the base Bc on the batches both ran."""
    tags = sorted(t for t in A if t in Bc and 'error' not in A[t] and 'error' not in Bc[t])
    if not tags:
        return dict(pairs=0)
    only = lambda key, X, Y: sum(bool(X[t][key]) and not Y[t][key] for t in tags)
    both = [t for t in tags if A[t]['clear_s'] is not None and Bc[t]['clear_s'] is not None]
    d = [A[t]['clear_s'] - Bc[t]['clear_s'] for t in both]
    m, lo, hi = boot_mean(d)
    da = [A[t]['alone'] - Bc[t]['alone'] for t in tags]
    am, alo, ahi = boot_mean(da)
    return dict(pairs=len(tags),
                person_a=sum(A[t]['person'] for t in tags), person_base=sum(Bc[t]['person'] for t in tags),
                person_only_a=only('person', A, Bc), person_only_base=only('person', Bc, A),
                person_p=mcnemar(only('person', A, Bc), only('person', Bc, A)),
                stop_a=sum(A[t]['stop'] for t in tags), stop_base=sum(Bc[t]['stop'] for t in tags),
                stop_only_a=only('stop', A, Bc), stop_only_base=only('stop', Bc, A),
                stop_p=mcnemar(only('stop', A, Bc), only('stop', Bc, A)),
                clear_diff_mean=m, clear_diff_ci=(lo, hi), clear_diff_median=median(d), clear_pairs=len(d),
                clear_diff_abs_median=median([abs(x) for x in d]),
                alone_diff_sum=sum(da), alone_diff_mean=am, alone_diff_ci=(alo, ahi),
                identical_batches=sum(A[t]['signature'] == Bc[t]['signature'] for t in tags),
                same_releases=sum(A[t]['signature'][3] == Bc[t]['signature'][3] for t in tags),
                multi_a=sum(A[t]['multi'] for t in tags), multi_base=sum(Bc[t]['multi'] for t in tags),
                bad_end_a=sum(A[t]['bad_end'] for t in tags), bad_end_base=sum(Bc[t]['bad_end'] for t in tags),
                retract_a=sum(A[t]['retracts'] > 0 for t in tags),
                retract_base=sum(Bc[t]['retracts'] > 0 for t in tags))


# ---- output ------------------------------------------------------------------------------------------
def pct(x, d=1):
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else '%.*f %%' % (d, 100 * x)


def num(x, fmt='%.1f'):
    return '—' if x is None else fmt % x


def table(head, rows):
    return ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def paired_rows(data, pairs):
    rows, out = [], {}
    for a, b in pairs:
        p = out['%s:%s' % (a, b)] = paired(data[a], data[b])
        if not p['pairs']:
            continue
        ci = p['clear_diff_ci'] if p['clear_diff_mean'] is not None else (None, None)
        rows.append(['%s 对 %s' % (a, b), '%d' % p['pairs'], '%d / %d' % (p['person_a'], p['person_base']),
                     '%d / %d' % (p['person_only_a'], p['person_only_base']), '%.3f' % p['person_p'],
                     '%s (%s, %s)' % (num(p['alone_diff_mean'], '%+.3f'), num(p['alone_diff_ci'][0], '%+.3f'),
                                      num(p['alone_diff_ci'][1], '%+.3f')),
                     '%s (%s, %s)' % (num(p['clear_diff_mean'], '%+.2f'), num(ci[0], '%+.2f'), num(ci[1], '%+.2f')),
                     num(p['clear_diff_median'], '%+.2f'), '%d / %d' % (p['bad_end_a'], p['bad_end_base']),
                     '%d / %d' % (p['retract_a'], p['retract_base']),
                     '%d / %d' % (p['identical_batches'], p['same_releases'])])
    return rows, out


def report(data, base, pairs=()):
    cfgs = [c for c in data if aggregate(list(data[c].values()))['batches']]
    agg = {c: aggregate(list(data[c].values())) for c in cfgs}
    L = ['各配置（需人工 = 有作废件停在秤上、卡堵、故障、掉料之一）：', '']
    L += table(['配置', '批 / 块', '需人工批 (95 % CI)', '作废停机批：真多块 / 单块作废', '卡堵或故障批', '犁面回退批',
                '单块有效测量', '清线 s：均值 / 中位 / p90 / 最大', '错误放行 / 未测落仓', '超数值筛查限（批）'],
               [[c, '%d / %d' % (a['batches'], a['lumps']),
                 '%d = %s (%s–%s)' % (a['person_batches'], pct(a['person_rate']), pct(a['person_ci'][0]),
                                      pct(a['person_ci'][1])),
                 '%d：%d / %d' % (a['stop_batches'], a['stop_double_batches'], a['stop_single_batches']),
                 '%d' % a['bad_end_batches'], '%d' % a['retract_batches'],
                 '%d = %s' % (a['alone'], pct(a['alone_share'])),
                 ' / '.join(num(a[k]) for k in ('clear_mean', 'clear_median', 'clear_p90', 'clear_max')),
                 '%d / %d' % (a['false_valid'], a['unmeasured']), '%d' % a['numerics_bad']] for c, a in agg.items()])
    L += ['', '给料机头和上游：', '']
    L += table(['配置', '放料次数', '一次下多块 (95 % CI)', '其中每块都单独测成', '停带后才下 / 预送带下（块）',
                '预送起动 / 每次放料', '因看不准暂停 s / 批', '点动 / 漏检', '漏斗里同时两块（批）', '上游被计量段按住 s / 批',
                '对象记数多于真实块数（件）'],
               [[c, '%d' % a['releases'],
                 '%d = %s (%s–%s)' % (a['multi'], pct(a['multi_rate']), pct(a['multi_ci'][0]), pct(a['multi_ci'][1])),
                 '%d / %d' % (a['multi_apart'], a['multi']), '%d / %d' % (a['after_stop'], a['in_staging']),
                 '%d / %.1f' % (a['stagings'], a['stagings'] / max(1, a['releases'])),
                 '%.2f' % (a['unsure_s'] / a['batches']), '%d / %d' % (a['jogs'], a['beam_missed']),
                 '%d' % a['funnel2_batches'], '%.2f' % (a['held_s'] / a['batches']), '%d' % a['n_est_over']]
                for c, a in agg.items()])
    todo = [(c, base) for c in cfgs if c != base and base in data]
    todo += [(a, b) for a, b in pairs if a in data and b in data and (a, b) not in todo]
    rows, pair = paired_rows(data, todo)
    if rows:
        L += ['', '同批配对（前者对后者）：', '']
        L += table(['比较', '对数', '需人工批：前者 / 后者', '只在前者 / 只在后者', 'McNemar p',
                    '每批单块有效测量数之差 (95 % CI)', '清线时间差均值 s (95 % CI)', '差的中位 s', '卡堵或故障批', '回退批',
                    '结果完全相同 / 放料分组相同（批）'], rows)
    L += ['', '作废件（控制器给的理由|真值）与异常结局：']
    for c, a in agg.items():
        L.append('- %s: %s；真多块 %d 件里同一次放料下来的 %d 件；结局异常 %s；故障 %s；掉料 %d 块' % (
            c, a['void_reasons'] or '无', a['doubles'], a['doubles_same_release'], a['bad_ends'] or '无',
            a['faults'] or '无', a['dropped_lumps']))
    used = [m for m in MECH_ZH if any(a['mech'].get(m) for a in agg.values())]
    if used:
        L += ['', '作废件按成因（件）：', '']
        L += table(['配置'] + [MECH_ZH[m] for m in used] + ['合计'],
                   [[c] + ['%d' % a['mech'].get(m, 0) for m in used] + ['%d' % sum(a['mech'].values())]
                    for c, a in agg.items()])
    L += ['', '按布料的需人工批：']
    for c in cfgs:
        by = []
        for lay in LAYOUTS:
            B = [b for b in data[c].values() if 'error' not in b and b['layout'] == lay]
            if B:
                by.append('%s %d/%d' % (LAYOUT_ZH[lay], sum(b['person'] for b in B), len(B)))
        L.append('- %s: %s' % (c, '，'.join(by)))
    L += ['', '代码与平台：' + '；'.join('%s %s %s' % (c, ','.join(a['sha']), ','.join(a['platform']))
                                    for c, a in agg.items())]
    errs = {c: [t for t, b in data[c].items() if 'error' in b] for c in data}
    if any(errs.values()):
        L += ['', '运行出错的批：%s' % {c: e for c, e in errs.items() if e}]
    return '\n'.join(L), agg, pair


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', nargs='+')
    p.add_argument('--base', default='base')
    p.add_argument('--pair', action='append', default=[], metavar='A:B', help='also compare A against B; repeatable')
    p.add_argument('--only', help='comma-separated configurations to read (default: all)')
    p.add_argument('--seeds', metavar='FIRST-LAST', help='only the batches of these seeds')
    p.add_argument('--screened', action='store_true', help='leave out the batches over the penetration limit')
    p.add_argument('--alias', action='append', default=[], metavar='A=B', help='read configuration A as B')
    p.add_argument('--json', help='write the aggregates here')
    p.add_argument('--collect', help='write every batch summary into this one file')
    a = p.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split('-')) if a.seeds else None
    data, raw = load(a.out, set(a.only.split(',')) if a.only else None, seeds, a.screened,
                     dict(x.split('=') for x in a.alias))
    text, agg, pair = report(data, a.base, [tuple(x.split(':')) for x in a.pair])
    print(text)
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(dict(base=a.base, aggregate=agg, paired=pair), fh, ensure_ascii=False, indent=1)
    if a.collect:
        with open(a.collect, 'w', encoding='utf-8') as fh:
            json.dump(raw, fh, ensure_ascii=False)


if __name__ == '__main__':
    main()
