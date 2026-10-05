"""Paired timestep sensitivity screen; retain raw runs and never certify calibration from this.

python checks/timestep_checks.py --seeds 1 309 392 --counts 3 4 5 --duration 90
The same solref, shapes, layout and 10 ms controller interval are retained at all timesteps.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator.config import parse_config
from singulator.simulate import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 309, 392])
    p.add_argument('--counts', nargs='+', type=int, default=[3, 4, 5])
    p.add_argument('--duration', type=float, default=90.)
    p.add_argument('--out', type=Path, default=Path('output/timestep'))
    args = p.parse_args()
    if len(args.counts) != len(args.seeds):
        p.error('one count per seed is required')
    rows, comparisons = [], []
    args.out.mkdir(parents=True, exist_ok=True)
    for preset in ('line',):
        for seed, count in zip(args.seeds, args.counts):
            group = []
            for dt in (.0005, .00025, .000125):
                cfg = parse_config(['--seed', str(seed), '--count', str(count),
                                    '--duration', str(args.duration), '--dt', str(dt), '--no-video'])
                cfg['out_dir'] = args.out / ('%s_%d_n%d_dt%g' % (preset, seed, count, dt))
                r = run(cfg)
                o, n = r['outcome'], r['numerics']
                row = dict(preset=preset, seed=seed, count=count, dt=dt, solref=cfg['solref'],
                           classification=o['classification'], clear_s=o['clear_s'], tail_passed=o['tail_passed'],
                           min_gap_s=o['min_net_time_gap_s'], unjam_pulses=o['unjam_pulses'],
                           numerical_ok=n['ok'], penetration_m=n['max_penetration_m'],
                           solver_at_limit_steps=n['solver_at_limit_steps'], warnings=n['warnings'],
                           input_shapes=[{k:b[k] for k in ('axes', 'density', 'material', 'family')} for b in r['blocks']],
                           result=str(cfg['out_dir']/'result.json'))
                rows.append(row)
                group.append(row)
                print('%s seed=%d n=%d dt=%g: %s clear=%s pen=%.2f mm numerical_ok=%s' %
                      (preset, seed, count, dt, row['classification'], row['clear_s'], row['penetration_m']*1000,
                       row['numerical_ok']), flush=True)
                (args.out/'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            valid = all(r['numerical_ok'] for r in group)
            classes_same = len({r['classification'] for r in group}) == 1
            clears = [r['clear_s'] for r in group if r['clear_s'] is not None]
            spread = max(clears)-min(clears) if len(clears) == 3 else None
            # This is a declared engineering screen, NOT a significance test or accuracy guarantee.
            stable = valid and classes_same and spread is not None and spread <= max(.1, .01*max(clears))
            comparisons.append(dict(preset=preset, seed=seed, count=count, identical_input_shapes=
                                    all(r['input_shapes'] == group[0]['input_shapes'] for r in group),
                                    classifications_agree=classes_same, all_numerically_ok=valid,
                                    clear_time_spread_s=spread, stable_under_screen=stable))
    (args.out/'summary.json').write_text(json.dumps(dict(
        note='Three timesteps, fixed contact softness. Stable means numerics pass, classification agrees and '
             'clear-time spread <= max(0.1 s, 1%). Not proof of convergence, physical accuracy or equipment reliability.',
        comparisons=comparisons), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
