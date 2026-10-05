"""Run one batch: python plough.py [--seed N] [--duration S] [--layout L] [--bench] [--no-video] ...

The model lives in the singulator package; see README.md. Outputs go to runs/<name>_<time>/.
"""
from singulator.config import parse_config
from singulator.simulate import run
from singulator.audit import format_summary


def main():
    cfg = parse_config()
    out = run(cfg)
    o, g = out['outcome'], out['geometry']
    face = (('curve %g->%g deg' % (g['curve_top_deg'], g['curve_exit_deg'])) if g['face_shape'] == 'curve'
            else ('skew %.0f deg' % g['skew_deg']))
    st_g = g['station']
    print('%s  |  %s  face %.2f m  lane %.2f x %.2f m  | feed step %.2f m, station step %.2f m, buffer %.2f m at %.2f m/s'
          ' (main %.2f m/s)'
          % (cfg['out_dir'].name, face, g['diagonal_length_m'], g['lane_width_m'], g['lane_length_m'],
             g['feeder']['step_m'], st_g['step_m'], st_g['buffer']['length_m'], st_g['buffer_speed_m_s'], cfg['v_belt']))
    print(format_summary(out['measurement_audit']))
    print('singulation diagnostic: %s  tail past the head edge %d/%d  together %.2f s (riding %.2f)  lane yaw median %s deg  retracts %d'
          ' | belt resistance 50 ms avg %.0f N | pen %.1f mm | %s | %.0f s'
          % (o['classification'], o['tail_passed'], o['blocks_total'], o['together_at_cut_s'], o['riding_over_s'],
             o['lane_yaw_misalignment_median_deg'], o['unjam_pulses'], out['drives']['belt']['resist_avg50ms_max_N'],
             1000 * out['numerics']['max_penetration_m'], 'ok' if out['numerics']['ok'] else 'NUMERICS BAD',
             out['numerics']['wall_clock_s']))
    fr = out['feeder']
    print('feed belt: %d releases, lumps per release %s, jogs %d, never released %s | feed head (%s): %s'
          % (fr['counts']['releases'], fr['counts']['lumps_per_release'], fr['counts']['jogs'], fr['never_released'],
             out['transfer']['case'], out['transfer']['summary']))
    print('funnel (plough start .. lane entry): at most %d lumps at once, two or more for %.2f s'
          % (o['funnel']['max_lumps'], o['funnel']['two_or_more_s']))
    st, n = out['station'], out['station']['counts']
    print('station: %d items (%d measured: %d valid, %d void %s: held, lumps %s taken off), routed %d coal'
          ' / %d gangue, landed %d, not as ideal %d | void discharged %d | verification: false valid %s, false'
          ' void %s | weigh error max %s %% | section held %.1f s, frozen %.1f s | line clear %s s%s'
          % (n['items'], n['measured'], n['valid'], n['void'], n['void_reasons'] or '',
             sorted({a['lump'] for a in out['taken_off']}), n['coal'], n['gangue'], n['landed'], n['route_not_as_ideal'],
             n['void_discharged'], st['verification']['false_valid'], st['verification']['false_void'],
             st['mass_error_pct_abs_max'], st['upstream_held_s'], st['frozen_s'], o['line_clear_s'],
             '' if not st['fault'] else ' | STOP %s' % st['fault']['reason']))
    for it in st['items']:
        if 't_decided_s' in it:
            print('  item %d: %.2f kg (true %.2f) %s %s -> %s | truth lumps %s'
                  % (it['item'], it['mass_kg'], it['truth_mass_kg'],
                     '%.1f L' % (1000 * it['volume_m3']) if it.get('volume_m3') else 'scan void',
                     '%.0f kg/m3' % it['density_kg_m3'] if it.get('density_kg_m3') else '',
                     'HELD (%s)' % ', '.join(it['reasons']) if it['void'] else it['route'], it['truth_lumps']))
    pc = out['perception']
    print('sensing %s: vision %s, centroid error %s | beams %s'
          % (pc['sensing'], pc['vision']['counts'], pc['vision']['centroid_error_m'],
             {k: (b['blocks'], (b['alarm'] or {}).get('why')) for k, b in pc['beams'].items()}))


if __name__ == '__main__':
    main()
