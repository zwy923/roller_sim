"""What a run leaves behind: the result (result.json), and the files.

assemble(line, wall_s) puts the result together from the finished line -- every part reports for itself (the
controllers, the witnesses, the drives, the sensors); what is judged here is the outcome of the singulation part:
how the batch passed the measuring plane (the main belt's head edge), and the one-word classification.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from ..audit import measurement_report
from ..lumps import batch_material_summary
from ..machine.assembly import CATS
from ..physics.lumps import END_STATES

# stops of the line that the station raised (outcome.classification 'station_fault')
STATION_FAULTS = ('separator_fault', 'separator_path_timeout', 'beam_long_block', 'beam_dead', 'beam_dirty',
                  'vision_lost')
FACE_FAULTS = ('face_reset_fault', 'face_motion_fault')


def provenance(xml, sensing='vision'):
    """What produced a result: the package source and the compiled model, hashed, and what the model leaves out."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return dict(source_sha256=h.hexdigest(),
                model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                calibrated=False, convergence_verified=False,
                limitations=['rigid convex unbreakable lumps; shape and material distributions uncalibrated',
                             ('modelled sensors (singulator/sensing/): camera outlines with white placeholder noise '
                              'and 50 ms latency, no occlusion by equipment; debounced beams; scanner validity on '
                              'convex lumps' if sensing == 'vision' else
                              'ideal full-outline sensing at 100 Hz; no occlusion or processing latency'),
                             'held belt surfaces; no pulley transfer, belt compliance or thermal trips',
                             'weighing = ideal sum of contact forces on the weigh belt and its skirts (no load-cell '
                             'dynamics, belt tension or vibration); volume = the true hull volume'])


def classify(numerics, stop, left_on_belt, dropped, together_s, retracts):
    """The outcome in one word, the first that applies."""
    if numerics.get('failure'):
        return 'numerical_failure'
    if stop:
        return ('face_fault' if stop['reason'] in FACE_FAULTS
                else 'station_fault' if stop['reason'] in STATION_FAULTS else 'jammed')
    if left_on_belt:
        return 'incomplete'
    if dropped:
        return 'dropped'
    if together_s > 0:
        return 'abreast_at_cut'
    return 'single_file_after_unjam' if retracts else 'single_file'


def assemble(line, wall_s):
    """The result of a finished run (sim.line.Line), as result.json holds it."""
    cfg, d, lumps, blocks, tr, face = line.cfg, line.d, line.lumps, line.blocks, line.trace, line.face
    numerics = dict(line.diagnostics.report(line.data), initial_penetration_m=line.initial_pen,
                    wall_clock_s=round(wall_s, 1))
    for L, b in zip(lumps, blocks):
        b.update(L.summary())
        b.pop('vertices', None)
    crossings, together = tr.crossings, tr.together
    crossings.sort()
    gaps = [round(lumps[kb].enter - lumps[ka].pass_t, 3) if lumps[ka].pass_t is not None else None
            for (ta, ka), (tb, kb) in zip(crossings, crossings[1:])]
    known_gaps = [g for g in gaps if g is not None]
    # a lump counts as through when its TAIL has passed the measuring plane; the front reaching it is
    # recorded separately and is not a clear batch
    front = [L for L in lumps if L.enter is not None]
    tail = [L for L in lumps if L.pass_t is not None]
    left_on_belt = [L.k for L in lumps if L.state not in END_STATES]        # still on the machine
    dropped = [L.k for L in lumps if L.state == 'dropped']
    yaws = [L.face_yaw for L in lumps if L.face_yaw is not None]
    lane_yaws = [L.lane_yaw for L in lumps if L.lane_yaw is not None]
    stop = line.supervisor.stop
    # the whole line is clear when the last lump has landed (in a separator bin, or dropped) or was taken off
    done = all(L.state in END_STATES for L in lumps)
    out = dict(
        model='plough_singulator_v1', mujoco=mujoco.__version__,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items() if not k.startswith('_')},
        # the hardware, with what the controllers and the measuring devices add to its description
        geometry=dict(d['report'], feeder=dict(line.feeder.geometry(), lumps_start_on_it=True),
                      station=line.station.geometry(), feed_layer_from_m=round(line.feed_rear, 3),
                      feed_layer_to_m=round(line.feed_front, 3)),
        outcome=dict(
            classification=classify(numerics, stop, left_on_belt, dropped, together['s'], face.events),
            blocks_total=len(lumps), front_arrived=len(front), tail_passed=len(tail), dropped=len(dropped),
            left_on_belt=left_on_belt,
            clear_s=(round(max(L.pass_t for L in tail), 2) if len(tail) == len(lumps) else None),
            first_pass_yield=round(len(tail) / len(lumps), 3),
            order=[k for _, k in crossings],
            together_at_cut_s=round(together['s'], 3),
            together_pairs_s={k: round(v, 3) for k, v in together['pairs'].items()},
            max_total_cut_width_m=round(together['max_total_n'], 3),
            riding_over_s=round(tr.riding, 3),
            net_time_gaps_s=gaps, min_net_time_gap_s=min(known_gaps) if known_gaps else None,
            unknown_time_gaps=sum(g is None for g in gaps), gap_sampling_uncertainty_s=.02,
            negative_gaps=sum(g < 0 for g in known_gaps),
            exit_yaw_misalignment_deg=yaws,
            exit_yaw_misalignment_median_deg=round(float(np.median(yaws)), 1) if yaws else None,
            lane_yaw_misalignment_deg=lane_yaws,
            lane_yaw_misalignment_median_deg=round(float(np.median(lane_yaws)), 1) if lane_yaws else None,
            touched_plough=sum(1 for L in lumps if L.touched_plough),
            end_time_s=round(float(line.data.time), 2), jam_stop=stop,
            unjam_pulses=len(face.events), unjam_events=face.events, **face.report(),
            funnel=dict(tr.funnel, two_or_more_s=round(tr.funnel['two_or_more_s'], 2)),
            numerical_screen_passed=numerics['ok'],
            stop_model='run terminates on alarm; no post-alarm braking/coasting simulation',
            line_clear_s=(round(max((L.drop_t for L in lumps if L.drop_t is not None), default=0.), 2)
                          if done else None),
            sorted=sum(L.state == 'sorted' for L in lumps), taken_off=sum(L.state == 'taken_off' for L in lumps)),
        drives=line.belts.report(line.dt, face.drive),
        numerics=numerics,
        blocks=blocks, feed_material=batch_material_summary(blocks),
        provenance=provenance(line.xml, sensing=cfg['sensing']),
        feeder=line.feed_witness.report(line.feeder.report()),
        transfer=dict(line.transfer.report(line.feeder), layout=line.layout),
        station=line.witness.report(line.station.report()),
        perception=line.sensors.report(),
        taken_off=line.takeoff.log,
        scenario=dict(kind=cfg['scenario'], **line.scenario.info))
    out['measurement_audit'] = measurement_report(out)
    return out


def write(cfg, out, xml, traj, drive_names):
    """result.json, model.xml (the model that was actually compiled: the derived report alone cannot be
    re-inspected or re-run) and trajectory.npz into cfg['out_dir']."""
    cfg['out_dir'].mkdir(parents=True, exist_ok=True)
    (cfg['out_dir'] / 'result.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    (cfg['out_dir'] / 'model.xml').write_text(xml, encoding='utf-8')
    f32 = lambda key: np.array(traj[key], np.float32)
    np.savez_compressed(cfg['out_dir'] / 'trajectory.npz', t=np.array(traj['t']), front_x=f32('front_x'),
                        pos=f32('pos'), quat=f32('quat'), com=f32('com'), touch=np.array(traj['touch'], np.uint16),
                        touch_categories=np.array(CATS), face_deg=f32('face_deg'),
                        face_contact_N=f32('face_contact_N'), drive_f=f32('drive_f'),
                        drive_names=np.array(drive_names), weigh_N=f32('weigh_N'), sep_deg=f32('sep_deg'))
