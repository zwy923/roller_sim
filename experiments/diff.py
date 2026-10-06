"""Were two runs of the same batches the same, batch by batch?

    python experiments/diff.py DIR_A/CONFIG DIR_B/CONFIG

Reads the batch summaries run.py writes (<tag>.json) and compares, exactly, every field that says what happened in
a batch (KEYS: the outcome, each item's judgement and times, each release, the counts) -- not the wall time, the
arguments or the hash of the code. For the same job list on two versions of the code, or run twice on one machine.
Different machines or library versions do not give the same batch (docs/TODO.md, section 7). Exit status 1 if any
batch differs.
"""
import glob
import json
import os
import sys

KEYS = ('classification', 'end_s', 'line_clear_s', 'holds', 'void', 'valid', 'void_reasons', 'n_items', 'landed',
        'dropped', 'unjam', 'route_not_as_ideal', 'void_discharged', 'false_valid', 'false_void', 'landed_unmeasured',
        'fault', 'jam_stop', 'releases', 'feeder_counts', 'station_counts', 'pen_mm', 'numerics_ok',
        'mass_error_max_pct', 'tips', 'vision', 'items', 'taken_off', 'beam_alarms')


def load(folder):
    return {os.path.basename(f): json.load(open(f, encoding='utf-8'))
            for f in sorted(glob.glob(os.path.join(folder, '*.json')))}


def compare(a, b):
    """(identical, [(tag, differing fields, a's summary, b's summary)], batches with no result on both sides)."""
    same, differ, no_result = 0, [], 0
    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        if 'error' in x and 'error' in y:              # the model refused the batch both times
            no_result += 1
        elif 'error' in x or 'error' in y:
            differ.append((name[:-5], ['error'], x, y))
        else:
            fields = [k for k in KEYS if x.get(k) != y.get(k)]
            if fields:
                differ.append((name[:-5], fields, x, y))
            else:
                same += 1
    return same, differ, no_result


def main(a_dir, b_dir):
    a, b = load(a_dir), load(b_dir)
    same, differ, no_result = compare(a, b)
    code = lambda runs: sorted({(s.get('source_sha') or '?')[:12] for s in runs.values()})
    print('%s vs %s: %d batches in both, %d identical on every compared field, %d differ, %d with no result in both; '
          'only in the first %d, only in the second %d'
          % (a_dir, b_dir, len(set(a) & set(b)), same, len(differ), no_result, len(set(a) - set(b)),
             len(set(b) - set(a))))
    print('code', code(a), code(b))
    for tag, fields, x, y in differ:
        print('  DIFF %s: %s' % (tag, ', '.join(fields[:6]) + (' ...' if len(fields) > 6 else '')))
        for k in ('classification', 'holds', 'line_clear_s', 'fault'):
            if x.get(k) != y.get(k):
                print('       %s: %s | %s' % (k, x.get(k), y.get(k)))
    return 1 if differ else 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
