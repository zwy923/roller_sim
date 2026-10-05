"""Shared per-lump measurement audit, evaluated only after simulation (no control feedback).

The fixed-time concept device succeeds when a valid, unrevoked item is one true lump,
isolated from off-scale contacts, with a valid scan. Reading stability, mass error and
correct sorting are later checks, not requirements of the primary measurement rate.
This module is pure Python so saved results can be audited without MuJoCo.
"""
from collections import Counter

BAD_END = ('jammed', 'station_fault', 'face_fault', 'incomplete', 'numerical_failure')
END_STATES = ('sorted', 'dropped', 'taken_off')     # a lump in any other state is still on the line
MASS_TOL = 2.           # %: the model's own verification line for a reading (station.verification); not a promise


def void_cause(it):
    """Why a void item could not stand for one lump: the true state first, the controller's reasons last."""
    why, info, scan = it.get('reasons') or [], it.get('reason_info') or {}, it.get('scan') or {}
    detail = lambda k: (info.get(k) or {}).get('detail') or ''
    if it.get('truth_single') is False:
        return 'multi'
    if it.get('truth_isolated') is False:
        return 'touching'
    if 'outside' in why and detail('outside').startswith('front at the measuring belt head'):
        return 'touching'               # carried to the measuring belt's head with its rear still on the buffer belt
    if 'not_wholly_on_the_belt' in (scan.get('reasons') or []):
        return 'touching'               # the scanner's geometry is the true one: part of it is off the measuring belt
    if (scan.get('objects') or 1) >= 2:
        return 'near'                   # ... and so is its count of what reaches into the zone
    if 'outside' in why and detail('outside').startswith('another object within'):
        return 'near'                   # the isolation gap: it may lean on it; the true state says it does not
    return 'misjudged'


def audit(r):
    """One full result -> (lumps, batch). lumps: {k: dict(fate=..., ...)} for every lump put in; batch: its flags."""
    B, st = r['blocks'], r['station']
    items, thr = st['items'], r['config']['sort_density']
    of = {}
    for it in items:                                # the items that held a lump (truth_lumps: what really lay on M)
        for k in it.get('truth_lumps') or []:
            of.setdefault(k, []).append(it)
    with_item = {e['lump']: e.get('items') or [] for e in r.get('taken_off') or []}
    by_id = {it['item']: it for it in items}
    L = {}
    for k, b in enumerate(B):
        mine, f = of.get(k, []), dict(family=b.get('family'), state=b['state'])
        ideal = 'gangue' if b['density'] >= thr else 'coal'
        valid = [it for it in mine if it.get('void') is False and not it.get('revoked')]
        if len(mine) > 1:
            f['linked_items'] = [it['item'] for it in mine]         # tied to more than one item: listed, never summed
            f['valid_twice'] = len(valid) > 1                       # ... and one lump is one success at most
        if valid:
            it = valid[0]
            known = all(isinstance(v, bool) for v in (it.get('truth_single'), it.get('truth_isolated'),
                                                      (it.get('scan') or {}).get('valid')))
            alone = (known and it['truth_single'] and it['truth_isolated'] and it['scan']['valid']
                     and it.get('truth_lumps') == [k])
            err = it.get('mass_error_pct')
            sent, landed = it.get('t_discharge_s') is not None, b['state'] == 'sorted'
            f.update(fate='measured' if alone else 'false_valid' if known else 'undetermined', item=it['item'],
                     steady=it.get('steady'), mass_error_pct=err, sent=sent, landed=landed, route=it.get('route'),
                     bin=b.get('bin'), ideal=ideal, unsteady=bool(alone and it.get('steady') is not True),
                     mass_off=bool(alone and (err is None or abs(err) > MASS_TOL)),
                     route_wrong=bool(alone and sent and it.get('route') != ideal),
                     bin_wrong=bool(alone and landed and b.get('bin') != it.get('route')),
                     truth_lumps=it.get('truth_lumps'), truth_isolated=it.get('truth_isolated'))
            f['whole_line'] = bool(alone and sent and landed and it.get('route') == ideal and b.get('bin') == ideal)
        elif mine:
            it = mine[0]
            f.update(fate=void_cause(it), item=it['item'], reasons=it.get('reasons'),
                     detail={k2: (v or {}).get('detail') for k2, v in (it.get('reason_info') or {}).items()},
                     scan=(it.get('scan') or {}).get('reasons'), truth_lumps=it.get('truth_lumps'),
                     truth_isolated=it.get('truth_isolated'))
            if b['state'] == 'sorted':
                f.update(fate='unmeasured_landing', bin=b.get('bin'), void_item=it['item'])
        elif b['state'] == 'taken_off':
            held = [by_id[i] for i in with_item.get(k, []) if i in by_id and by_id[i].get('void')]
            f.update(fate='collateral', with_item=[h['item'] for h in held],
                     with_cause=[void_cause(h) for h in held], with_reasons=[h.get('reasons') for h in held])
        elif b['state'] == 'sorted':
            f.update(fate='unmeasured_landing', bin=b.get('bin'))
        else:
            f.update(fate='dropped' if b['state'] == 'dropped' else 'left')
        L[k] = f
    fate = Counter(f['fate'] for f in L.values())
    ver, cnt, num = st.get('verification') or {}, st.get('counts') or {}, r['numerics']
    model = (len(ver.get('false_valid') or []) + len(ver.get('landed_unmeasured') or [])
             + (cnt.get('void_discharged') or 0))
    cls = r['outcome']['classification']
    failed = bool(fate['false_valid'] or fate['unmeasured_landing'] or fate['undetermined'] or model
                  or any(f.get('valid_twice') or f.get('route_wrong') or f.get('bin_wrong') for f in L.values()))
    unfinished = bool(cls in BAD_END or any(b['state'] not in END_STATES for b in B))
    manual = bool(any(fate[c] for c in ('multi', 'touching', 'near', 'misjudged', 'collateral', 'dropped')))
    whole = sum(bool(f.get('whole_line')) for f in L.values())
    batch = dict(cls=cls, clear_s=r['outcome'].get('line_clear_s'), numerics_bad=not num['ok'],
                 # over the limit only where a lump lands on a bin floor: nothing of the line (S6). Results written
                 # since 2026-10-06 screen the machine only (physics/numerics.py): this flag is for older ones
                 numerics_floor_only=bool(not num['ok'] and num.get('failure') is None and not num.get('warnings')
                                          and 'floor' in ((num.get('worst_contact') or {}).get('geoms') or [])),
                 verification_failed=failed, model_verification=model, unfinished=unfinished, manual=manual,
                 all_measured=fate['measured'] == len(B),
                 passed=bool(whole == len(B) and num['ok'] and not (failed or unfinished or manual)),
                 holds=cnt.get('holds') or 0, taken_off=len(r.get('taken_off') or []),
                 unjam=r['outcome'].get('unjam_pulses') or 0,
                 sha=(r.get('provenance') or {}).get('source_sha256', '')[:12],
                 feeder=(r.get('perception') or {}).get('feeder_sensing'), sensing=r['config'].get('sensing'),
                 weigh=r['config'].get('weigh_model', 'steady'), lane_w=r['config'].get('lane_w'))
    return L, batch


def measurement_report(result):
    """JSON-ready audit; the denominator includes every input lump, even taken-off ones.

    Batch flags retain the S7 definitions. Status is an overview, not a replacement for
    those flags or the measurement fraction: later-check failures stay visible even
    when every lump was measured independently. Old station verification is kept too.
    """
    lumps, batch = audit(result)
    fate = Counter(f['fate'] for f in lumps.values())
    measured = [f for f in lumps.values() if f['fate'] == 'measured']
    total = len(lumps)
    status = ('numerical_invalid' if batch['numerics_bad'] else
              'verification_failed' if batch['verification_failed'] else
              'incomplete' if batch['unfinished'] else
              'manual_required' if batch['manual'] else
              'passed' if batch['passed'] else 'undetermined')
    return dict(
        schema_version=1,
        metric='independent_measurement',
        definition='valid unrevoked item; one true lump; isolated from off-scale contacts; valid scan. '
                   'Denominator: all input lumps, including taken-off lumps. Stability, mass error and '
                   'correct sorting are later checks.',
        counts=dict(input=total, measured=len(measured),
                    success_rate=len(measured) / total if total else None,
                    not_measured=total - len(measured),
                    failure_causes={k: v for k, v in sorted(fate.items()) if k != 'measured'}),
        verification=dict(false_valid=fate['false_valid'],
                          unmeasured_landing=fate['unmeasured_landing'],
                          undetermined=fate['undetermined'],
                          duplicate_valid=sum(bool(f.get('valid_twice')) for f in lumps.values()),
                          void_discharged=result['station'].get('counts', {}).get('void_discharged', 0),
                          model_verification=batch['model_verification']),
        later_checks=dict(unsteady=sum(f['unsteady'] for f in measured),
                          mass_off=sum(f['mass_off'] for f in measured),
                          whole_line=sum(f['whole_line'] for f in measured),
                          route_wrong=sum(bool(f.get('route_wrong')) for f in lumps.values()),
                          bin_wrong=sum(bool(f.get('bin_wrong')) for f in lumps.values())),
        batch=dict(batch, status=status),
        lumps=[dict(lump=k, **f) for k, f in lumps.items()])


def format_summary(report):
    """Primary CLI readout, followed by safety flags and the separate later checks."""
    n, b = report['counts'], report['batch']
    v, later = report['verification'], report['later_checks']
    rate = 'n/a' if n['success_rate'] is None else '%.1f%%' % (100 * n['success_rate'])
    return ('independent measurement: %d/%d (%s) | batch %s | holds %d, taken off %d | failures %s\n'
            'audit: false valid %d, unmeasured landing %d, void discharged %d, duplicate valid %d, '
            'undetermined %d | numerics %s | model verification flags %d\n'
            'later checks: unsteady %d, mass error >2%% or missing %d, whole line %d/%d, '
            'wrong route %d, wrong bin %d'
            % (n['measured'], n['input'], rate, b['status'], b['holds'], b['taken_off'],
               n['failure_causes'] or 'none', v['false_valid'], v['unmeasured_landing'],
               v['void_discharged'], v['duplicate_valid'], v['undetermined'],
               'BAD' if b['numerics_bad'] else 'ok', v['model_verification'],
               later['unsteady'], later['mass_off'], later['whole_line'], n['input'],
               later['route_wrong'], later['bin_wrong']))

