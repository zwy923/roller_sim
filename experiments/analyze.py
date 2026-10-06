"""Can the line measure every lump on its own? The audit of a batch run (S7 on, 2026-10-05): pure Python.

    python experiments/analyze.py OUT_DIR [OUT_DIR ...] [--only a,b] [--seeds FIRST-LAST] [--alias a=b ...]
                                  [--list] [--json FILE]

Reads what experiments/run.py writes: <config>/<tag>.json (a summary; only 'error' when the batch did not run)
and <config>/<tag>.result.json.gz (the full result). The audit is imported from singulator.audit, also used by
simulate.run and the single-run CLI. It is keyed by the true lump put on the feed belt, never by a camera's object number, and the true state
is used to judge the result only.

    measured alone   the lump has one item; that item is valid (not void, not revoked); it truly held this one lump,
    ---------------  touching nothing off the scale; its scan was valid. A missing field is never a pass
    lumps put in     all of them: a lump whose item was void and taken off the line stays in the count

A valid item that was NOT one lump alone (false valid), a lump in a bin that no valid single-lump item sent there,
a lump tied to two items, a wrong route and a wrong bin are listed on their own: an average must not hide them.
Later checks are counted beside the measure and do not enter it (user, 2026-10-05: the measuring device is a concept
device that takes a fixed second, its reading is not under test): the reading was not steady when it was taken; it
was more than 2 % off the true mass; 'whole line' = the lump was measured alone, discharged by the route its true
density asks for, and lies in that bin.

Every lump that was not measured alone gets one cause, from the true state first:
    multi        its item truly held more than one lump                                       (多块一起上秤)
    touching     one lump, but leaning on something off the scale, or lying across the joint  (接触秤外)
    near         one lump, touching nothing, but another lump reached into the measuring zone
                 or came within the isolation gap of it                                       (邻块贴近)
    misjudged    one lump, alone and clear, and the controller still made it void              (识别误判)
    collateral   never judged: on the measuring belt or across the joint when a void item was taken off
    dropped      fell off the line
    left         still on the line when the run ended (jam, fault, time limit)
A batch keeps every flag that applies: no result / over the numerical screening limit / verification failed /
unfinished / needs a person / every lump measured alone / passed (every lump through the whole line, nothing else).
Several OUT_DIRs are read as one; --alias a=b reads the folder of configuration a under the name b.
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys
from pathlib import Path
from collections import Counter, OrderedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator.audit import BAD_END, audit, void_cause  # one implementation for saved and live results

LAYOUT_ZH = OrderedDict(scatter='随机', aligned='并齐', touching='相贴', oblique='斜放')
CAUSE_ZH = OrderedDict([('multi', '多块一起上秤'), ('touching', '接触秤外'), ('near', '邻块贴近（未接触）'),
                        ('misjudged', '识别误判'), ('collateral', '连带移除'), ('dropped', '掉料'),
                        ('left', '没走完')])
OTHER_ZH = OrderedDict([('false_valid', '错误有效测量'), ('unmeasured_landing', '未测落仓'),
                        ('undetermined', '不可判定')])

def boot_ratio(pairs, n=4000, seed=20261005):
    """95 % interval of sum(a) / sum(b) over batches, resampling batches (lumps of one batch are not independent)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        S = [rng.choice(pairs) for _ in pairs]
        vals.append(sum(a for a, _ in S) / max(1, sum(b for _, b in S)))
    vals.sort()
    return vals[int(.025 * n)], vals[int(.975 * n) - 1]


def load(outs, only=None, seeds=None, alias=None):
    """{config: {tag: dict(lumps=..., **batch flags)}}; a batch without a result is dict(no_result=its error)."""
    data = OrderedDict()
    for fp in sorted(fp for out in outs for fp in glob.glob(os.path.join(out, '*', '*.json'))):
        cfg = os.path.basename(os.path.dirname(fp))
        cfg = (alias or {}).get(cfg, cfg)
        tag = os.path.basename(fp)[:-len('.json')]
        if (only and cfg not in only) or (seeds and not seeds[0] <= int(tag.rsplit('_', 1)[1]) <= seeds[1]):
            continue
        if tag in data.get(cfg, {}):
            raise SystemExit('%s/%s is in more than one OUT_DIR' % (cfg, tag))
        full = fp[:-len('.json')] + '.result.json.gz'
        if not os.path.exists(full):
            with open(fp, encoding='utf-8') as fh:
                why = json.load(fh).get('error') or 'no full result'
            data.setdefault(cfg, OrderedDict())[tag] = dict(no_result=str(why).strip().splitlines()[-1][:200])
            continue
        with gzip.open(full, 'rt', encoding='utf-8') as fh:
            L, batch = audit(json.load(fh))
        data.setdefault(cfg, OrderedDict())[tag] = dict(batch, lumps=L)
    return data


def zh(tag):
    lay, seed = tag.rsplit('_', 1)
    return '%s %s' % (LAYOUT_ZH.get(lay, lay), seed)


def table(head, rows):
    return ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def lumps(T):
    return [f for b in T.values() for f in b['lumps'].values()]


def rate(T):
    """(measured alone, lumps put in, 95 % interval) over the batches that have a result."""
    T = {t: b for t, b in T.items() if 'lumps' in b}
    n, ok = len(lumps(T)), sum(f['fate'] == 'measured' for f in lumps(T))
    lo, hi = boot_ratio([(sum(f['fate'] == 'measured' for f in b['lumps'].values()), len(b['lumps']))
                         for b in T.values()]) if T else (float('nan'),) * 2
    return ok, n, lo, hi


def pct(ok, n, lo, hi):
    return '%.1f %% (%.1f–%.1f)' % (100 * ok / n, 100 * lo, 100 * hi) if n else '—'


HEAD1 = ['', '批', '投入块数', '独立测量成功', '成功率（95 % 区间，按批重抽）'] + list(OTHER_ZH.values())[:2] + list(
    CAUSE_ZH.values()) + [OTHER_ZH['undetermined']]
HEAD2 = ['', '独立测量成功', '读数时还没稳（只记录）', '质量差 > 2 %', '整线成功（排出且按真实密度落对仓）', '错分', '错仓',
         '一块料对上两件', '模型自带核验项之和']
HEAD3 = ['', '全部任务', '无结果', '数值筛查超限（其中只是落仓砸地）', '全批独立测成', '全批自动完成', '需人工处理的批',
         '没走完的批', '核验失败的批', '作废停机次数', '移出的块数', '犁面自动撤离次数', '清线均值 s']


def row1(name, T):
    T = {t: b for t, b in T.items() if 'lumps' in b}
    F = Counter(f['fate'] for f in lumps(T))
    ok, n, lo, hi = rate(T)
    return [name, '%d' % len(T), '%d' % n, '%d' % ok, pct(ok, n, lo, hi), '%d' % F['false_valid'],
            '%d' % F['unmeasured_landing']] + ['%d' % F[c] for c in CAUSE_ZH] + ['%d' % F['undetermined']]


def row2(name, T):
    M = [f for f in lumps({t: b for t, b in T.items() if 'lumps' in b})]
    ok = [f for f in M if f['fate'] == 'measured']
    return [name, '%d' % len(ok), '%d' % sum(f['unsteady'] for f in ok), '%d' % sum(f['mass_off'] for f in ok),
            '%d' % sum(f['whole_line'] for f in ok),
            '%d' % sum(bool(f.get('route_wrong')) for f in M), '%d' % sum(bool(f.get('bin_wrong')) for f in M),
            '%d' % sum(bool(f.get('linked_items')) for f in M),
            '%d' % sum(b['model_verification'] for b in T.values() if 'lumps' in b)]


def row3(name, T):
    R = [b for b in T.values() if 'lumps' in b]
    clear = [b['clear_s'] for b in R if b['clear_s'] is not None]
    return [name, '%d' % len(T), '%d' % (len(T) - len(R)),
            '%d（%d）' % (sum(b['numerics_bad'] for b in R), sum(b['numerics_floor_only'] for b in R)),
            '%d' % sum(b['all_measured'] for b in R), '%d' % sum(b['passed'] for b in R),
            '%d' % sum(b['manual'] for b in R), '%d' % sum(b['unfinished'] for b in R),
            '%d' % sum(b['verification_failed'] for b in R), '%d' % sum(b['holds'] for b in R),
            '%d' % sum(b['taken_off'] for b in R), '%d' % sum(b['unjam'] for b in R),
            '%.1f' % (sum(clear) / len(clear)) if clear else '—']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', nargs='+')
    p.add_argument('--only', help='comma-separated configurations (default: all)')
    p.add_argument('--seeds', metavar='FIRST-LAST')
    p.add_argument('--alias', action='append', default=[], metavar='A=B', help='read configuration A as B')
    p.add_argument('--list', action='store_true', help='every lump that was not measured alone, one per line')
    p.add_argument('--json', help='write every lump of every batch here')
    a = p.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split('-')) if a.seeds else None
    data = load(a.out, set(a.only.split(',')) if a.only else None, seeds, dict(x.split('=') for x in a.alias))
    L = ['独立测量成功 = 这块料有有效且未撤销的测量，真值是只有它一块、没有接触秤外的东西，'
         '扫描有效。分母是投入的全部料块，作废移走的也在里面。', '']
    L += table(['配置'] + HEAD1[1:], [row1(c, T) for c, T in data.items()])
    L += ['', '后续检查（不进主指标）：', ''] + table(['配置'] + HEAD2[1:], [row2(c, T) for c, T in data.items()])
    L += ['', '按批：', ''] + table(['配置'] + HEAD3[1:], [row3(c, T) for c, T in data.items()])
    for c, T in data.items():
        R = {t: b for t, b in T.items() if 'lumps' in b}
        L += ['', '%s，按布料：' % c, '']
        L += table(['布料'] + HEAD1[1:], [row1(LAYOUT_ZH[lay], {t: b for t, b in R.items() if t.startswith(lay + '_')})
                                         for lay in LAYOUT_ZH if any(t.startswith(lay + '_') for t in R)])
        clean = {t: b for t, b in R.items() if not b['numerics_bad']}
        ends = Counter(b['cls'] for b in R.values() if b['cls'] in BAD_END)
        L += ['', '去掉数值筛查超限的批：%d 批，%s' % (len(clean), '%d / %d = %s' % (rate(clean)[:2] + (pct(*rate(clean)),))),
              '结局异常的批：%s；无结果的批：%s' % (dict(ends) or '无', ['%s（%s）' % (zh(t), b['no_result'])
                                                          for t, b in T.items() if 'lumps' not in b] or '无'),
              '代码 %s；出口 %s m；给料读 %s，计量段读 %s，计量装置 %s' % tuple(
                  sorted({str(b[k]) for b in R.values()}) for k in ('sha', 'lane_w', 'feeder', 'sensing', 'weigh'))]
        if a.list:
            L += ['', '%s，没有独立测成的料块：' % c]
            for t, b in R.items():
                for k, f in b['lumps'].items():
                    if f['fate'] != 'measured':
                        more = {k2: v for k2, v in f.items()
                                if k2 not in ('fate', 'family', 'state') and v not in (None, [], {})}
                        name = CAUSE_ZH.get(f['fate']) or OTHER_ZH.get(f['fate'], f['fate'])
                        L.append('- %s 第 %d 块（%s）：%s %s' % (zh(t), k, f['family'], name,
                                                             json.dumps(more, ensure_ascii=False)))
            off = [(t, k, f) for t, b in R.items() for k, f in b['lumps'].items()
                   if f['fate'] == 'measured' and (not f['whole_line'] or f['mass_off']) or f.get('linked_items')]
            if off:
                L += ['', '%s，独立测成、但后续检查没过的（质量差 > 2 %%、没排出、错分、错仓、对上两件）：' % c]
                L += ['- %s 第 %d 块（%s）：读数稳 %s，质量差 %s %%，去向 %s，落 %s，应落 %s%s' % (
                    zh(t), k, f['family'], f.get('steady'), f.get('mass_error_pct'), f.get('route'), f.get('bin'),
                    f.get('ideal'), '，对上的件 %s' % f['linked_items'] if f.get('linked_items') else '')
                    for t, k, f in off]
    print('\n'.join(L))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False)


if __name__ == '__main__':
    main()
