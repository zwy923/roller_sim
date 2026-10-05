"""One batch through the machine: compile the model, step it, observe the lumps, judge, write the outputs.

Outputs in cfg['out_dir']: result.json (config, derived geometry, outcome, drive loads, retract events,
per-lump record, 'feeder' (releases, stops), 'transfer' (the feed head record, singulator/trial.py),
'station' (every item: weighing, volume, route, bin, void reasons, the verification against the true state),
'perception' (vision, beams, scanner, injected faults), 'taken_off' (lumps of held items)), model.xml (the
model actually compiled), trajectory.npz (every 10 ms: body pose, centroid, front edge, contact categories,
face angle, lump-face force, drive speed factors, the measuring belt's load-cell reading 'weigh_N' -- per-step
mean while weighing, else the instant -- and the separator plate angle 'sep_deg'), video.mp4 unless --no-video.

What the controllers read: with --sensing vision (the default) the station, the plough face and the alarms read
only modelled sensor signals (singulator/perception.py, line.py); the feed belt controller reads the lumps' true
centroids (--feeder-sensing oracle, the default since 2026-10-05) or the cameras too (--feeder-sensing vision).
With --sensing oracle everything reads the true state, for comparison.
"""
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from .assembly import build_xml
from .audit import measurement_report
from .drives import Conveyors
from .face import FaceRetract
from .feeder import Feeder
from .line import Scenario, SectionWatch, Sensors, TakeOff
from . import trial
from .lumps import block_world_vertices, make_blocks, plan_rows, plan_scatter, set_mass_properties, batch_material_summary
from .machine import derive
from .physics import Diagnostics, provenance, validate
from .station import Station, weigher_reading
from .tracking import CATS, Lump, categorise, lane_discharging, stalled
from .video import Recorder, overlay


STATION_FAULTS = ('separator_fault', 'separator_path_timeout', 'beam_long_block', 'beam_dead', 'beam_dirty',
                  'vision_lost')


def _face_force(model, data, cat):
    """Total lump-face normal force."""
    face_N, wrench = 0., np.zeros(6)
    lump, face = CATS.index('block'), CATS.index('plough')
    for ci in range(data.ncon):
        con = data.contact[ci]
        c1, c2 = int(cat[con.geom1]), int(cat[con.geom2])
        if {c1, c2} != {lump, face}:
            continue
        mujoco.mj_contactForce(model, data, ci, wrench)
        face_N += float(wrench[0])
    return face_N


def run(cfg):
    cfg = validate(cfg)
    rng = np.random.default_rng(cfg['seed'])
    blocks = make_blocks(cfg, rng)
    d = derive(cfg)
    if cfg['solref'] < 2 * cfg['dt']:
        raise ValueError('--solref %.4g s is below 2*dt (%.4g s): the soft contact stops being resolved'
                         ' (the S model lost runs to exactly this)' % (cfg['solref'], 2 * cfg['dt']))
    xml = build_xml(cfg, d, blocks)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    cat = categorise(model)
    lumps = [Lump(model, k) for k in range(len(blocks))]
    for L, b in zip(lumps, blocks):
        set_mass_properties(b, model.body_mass[L.body])
    geom_lump = {L.geom: L.k for L in lumps}
    plan = plan_rows if cfg['layout'] == 'rows' else plan_scatter
    layer, feed_rear, feed_front = plan(cfg, blocks, rng, d)
    for k, x, y, z, yaw in layer:
        lumps[k].place(data, x, y, z, yaw)
    belts = Conveyors(cfg, d, model)
    belts.command(data)                        # place every slat before the first contact evaluation
    station = Station(cfg, d, blocks)
    station.bind(model)
    station.place(model, data)                 # separator plate down (coal), linkage closed
    mujoco.mj_forward(model, data)
    watch = None
    vision = cfg['sensing'] == 'vision'
    scenario = Scenario(cfg, d, model, data, lumps, blocks)
    scenario.setup()                           # touching: two lumps laid touching in the lane
    sensors = Sensors(cfg, d, model, data, lumps, blocks)
    takeoff = TakeOff(d, model, data, lumps)
    layout = trial.arrange(cfg, d, model, data, lumps, rng)     # a bench layout; scatter / rows / flat: as placed
    transfer = trial.TransferWatch(cfg, d, model, lumps)
    # the whole batch lies on the feed belt; the first release starts at once
    feeder = Feeder(cfg, d, len(lumps))
    feeder.start(0.)
    dt = model.opt.timestep
    face_joint = None
    if cfg['unjam']:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'facej')
        face_joint = (model.jnt_qposadr[j], model.jnt_dofadr[j])
    actuator = model.actuator('face_drive').id if face_joint and cfg['face_drive_model'] == 'dynamic' else None
    face = FaceRetract(cfg, d, dt, face_joint, actuator=actuator)
    diagnostics = Diagnostics(model, cfg)
    # Layout overlap is an invalid initial condition, not an impact generated by the machine.
    initial_pen = max((max(0., -float(c.dist)) for c in data.contact[:data.ncon]), default=0.)
    if initial_pen > .001:
        raise ValueError('initial equipment/lump overlap %.6f m exceeds 1 mm' % initial_pen)

    stride = max(1, round(.01 / dt))                # observation every 10 ms
    dt_sample, slow = stride * dt, .02 * stride * dt
    n_win = int(round(cfg['jam_window'] / (stride * dt)))
    section_lo = d['P0'][0] - .3                    # singulation section: plough start - 0.3 m .. lane exit
    if vision:
        watch = SectionWatch(cfg, d, section_lo)
    together, riding, jam_stop = dict(s=0., pairs={}, max_total_n=0.), 0., None
    # the plough funnel, plough start to lane entry: how many lumps reach into it at once
    funnel = dict(x_from_m=round(float(d['P0'][0]), 4), x_to_m=round(float(d['lane_out_start']), 4), max_lumps=0,
                  two_or_more_s=0.)
    crossings = []
    traj = dict(t=[], front_x=[], pos=[], quat=[], com=[], touch=[], face_deg=[], face_contact_N=[],
                drive_f=[], weigh_N=[], sep_deg=[])

    def waiting(L):
        """Planned waiting, for the stall bookkeeping only: queued on the feed belt, or held (or taken) by the
        station."""
        if feeder.waiting(L.k, L.state, L.last_pose[0]):
            return 'feeder'
        if station.waiting(L.k, L.state, L.last_pose[0]):
            return 'station'
        return False

    rec = Recorder(cfg, d, model) if cfg['video'] else None
    wall = time.time()
    for step in range(int(math.ceil(cfg['duration'] / dt))):
        t = data.time
        f_target = min(1., t / cfg['ramp']) if cfg['ramp'] > 0 else 1.
        belts.upstream, belts.buffer_target, belts.measure_target = station.step(data, dt)
        belts.feed_target = 0. if feeder.paused else feeder.drive_target(dt)
        belts.command(data)
        face.command(t, data)
        mujoco.mj_step(model, data)
        diagnostics.observe(data, t)           # forces/contacts that produced this integration step
        if diagnostics.failure:
            break
        face.record(t, data)
        belts.after_step(data, t, f_target, dt)
        if station.weighing:
            station.load_step(weigher_reading(model, data, station.weigher, geom_lump, load_only=True))
        if (step + 1) % stride and step + 1 < int(math.ceil(cfg['duration'] / dt)):
            continue

        # ---- 10 ms observation --------------------------------------------------------------------
        # mj_step leaves derived poses and contacts at the START of the step, but qpos/time have
        # advanced. Restore the held conveyor surfaces, then synchronise all observation fields.
        # This is independent of video and cannot be triggered only by rendering.
        belts.command(data)
        mujoco.mj_forward(model, data)
        t = float(data.time)
        touch = {L.k: set() for L in lumps}
        for c in range(data.ncon):
            g1, g2 = data.contact.geom1[c], data.contact.geom2[c]
            for ga, gb in ((g1, g2), (g2, g1)):
                if ga in geom_lump:
                    touch[geom_lump[ga]].add(CATS[cat[gb]])
        live = []
        for L in lumps:
            if L.state == 'taken_off':
                continue                       # a held item's lump, off the line
            cut, first = L.observe(data, t, touch[L.k], cfg, d, waiting, dt_sample, slow)
            if first:
                crossings.append((t, L.k))
            if cut is not None:
                live.append((L.k, cut))
        if len(live) >= 2:
            together['s'] += dt_sample
            together['max_total_n'] = max(together['max_total_n'], sum(c[1] - c[0] for _, c in live))
            for i in range(len(live)):
                for j in range(i + 1, len(live)):
                    (ka, ca), (kb, cb) = live[i], live[j]
                    key = '%d-%d' % (min(ka, kb), max(ka, kb))
                    together['pairs'][key] = together['pairs'].get(key, 0.) + dt_sample
                    if min(ca[1], cb[1]) - max(ca[0], cb[0]) > .05:
                        riding += dt_sample
        in_section = [L.k for L in lumps if L.state == 'on_belt'
                      and L.last_pose[3] < d['exit_x'] and L.last_pose[4] > section_lo]
        n_funnel = sum(1 for L in lumps if L.state == 'on_belt' and L.last_pose[4] > funnel['x_from_m']
                       and L.last_pose[3] < funnel['x_to_m'])
        funnel['max_lumps'] = max(funnel['max_lumps'], n_funnel)
        funnel['two_or_more_s'] += dt_sample * (n_funnel >= 2)
        frame = sensors.frame(t)
        # closing permission: the section is clear, and no lump outline reaches into what the face will sweep
        # on its way home (checked again while it closes). Vision: the objects the cameras see there, their
        # outlines grown by the position margin
        if vision:
            v_section, v_stuck, v_passing = watch.update(
                t, frame, lambda o: station.section_held and o['cx'] <= d['station']['buffer']['x0'])
            face.zone_busy = bool(v_section)
            face.check_sweep(SectionWatch.plans(v_section))
        else:
            face.zone_busy = bool(in_section)
            face.check_sweep({L.k: L.plan for L in lumps if L.state == 'on_belt'})
        face_N = _face_force(model, data, cat)
        face.contact(face_N, dt_sample)
        reading = weigher_reading(model, data, station.weigher, geom_lump)
        station.observe(t, frame, sensors.beams, station.angle(data), sensors.scan,
                        audit=dict(reading=reading, states=[L.state for L in lumps], bins=[L.bin for L in lumps]))
        takeoff.act(t, station, sensors)
        scenario.act(t, station)
        if station.section_held or station.inhibit_release:
            feeder.pause(t)                    # the station holds the section, a void item, an alarm: no release
        else:
            feeder.resume(t)
            feeder.observe(t, sensors.feed_view, sensors.beams['beam_feed'].blocked, face.at_home)
            if feeder.vision:
                feeder.audit(t, {L.k: L.last_pose[0] for L in lumps if L.state == 'on_belt'})
            if feeder.miss_streak >= 2 and sensors.beams['beam_feed'].alarm is None:
                sensors.beams['beam_feed'].alarm = dict(t_s=round(t, 3), beam='beam_feed', why='dead', stop=True,
                                                        note='two releases in a row went without the beam')
        transfer.observe(t, data, feeder, belts)
        if cfg['bench'] and all(r['gap_before_m'] is not None for r in transfer.rec.values()) and all(
                s.get('done') for s in transfer.stops):
            break                              # the bench: every lump lies on the main belt, its questions are answered
        traj['t'].append(t)
        traj['front_x'].append([L.s_hist[-1][1] for L in lumps])
        traj['pos'].append([data.xpos[L.body].copy() for L in lumps])
        traj['quat'].append([data.xquat[L.body].copy() for L in lumps])
        traj['com'].append([data.xipos[L.body].copy() for L in lumps])
        traj['touch'].append([sum(1 << CATS.index(c) for c in touch[L.k]) for L in lumps])
        traj['face_deg'].append(math.degrees(face.theta))
        traj['face_contact_N'].append(face_N)
        traj['drive_f'].append(belts.factors())
        traj['weigh_N'].append(station.cell)
        traj['sep_deg'].append(station.plate_deg)

        # ---- stall -> retract, or stop for the operator ---------------------------------------------
        if vision:
            stuck, passing, in_section = v_stuck, v_passing, list(v_section)
        else:
            stuck = stalled(lumps, in_section, waiting, t, cfg, n_win)
            passing = lane_discharging(lumps, t, cfg['jam_window'])
        if face.hold_timed_out(t):
            # retracted and still not clear: the machine has nothing left to try. Stop with the face
            # open -- closing now would ram the lumps with the actuator
            jam_stop = dict(t_s=round(t, 2), reason='retract_hold_timeout', blocks_left=len(in_section),
                            stalled_blocks=stuck, in_closing_sweep=face.in_sweep, unjam_pulses_tried=len(face.events),
                            action='face stays retracted; stop conveying; operator clears the section')
            break
        if station.fault:
            # the separator did not get into position, or the station has had a lump in hand without progress
            jam_stop = dict(station.fault, blocks_left=len(in_section), unjam_pulses_tried=len(face.events),
                            action='stop the line; operator clears the station')
            break
        if face.fault:
            # the face did not get home: no further action on this batch
            jam_stop = dict(face.fault, reason='face_motion_fault', cause=face.fault['reason'], blocks_left=len(in_section),
                            unjam_pulses_tried=len(face.events),
                            action='terminate simulation; physical stopping transient not simulated')
            break
        if stuck and not passing and face.phase == 'idle' and t > cfg['ramp'] + 3. and t > face.next_ok:
            if cfg['unjam'] and len(face.events) < cfg['unjam_max']:
                face.start(t, blocks_left=len(in_section), stalled_blocks=stuck,
                           centroid_x=[round(frame[k]['cx'] if vision else lumps[k].c_hist[-1][1], 2) for k in stuck])
            else:
                jam_stop = dict(t_s=round(t, 2), reason='unjam_exhausted' if cfg['unjam'] else 'no_unjam',
                                blocks_left=len(in_section), stalled_blocks=stuck,
                                unjam_pulses_tried=len(face.events))
                break
        if t > cfg['ramp'] + 2. and all(L.state in d['end_states'] for L in lumps):
            break
        if rec and rec.due(t):
            # with the feed belt the camera follows the released lumps, not the ones queued on it
            went = feeder.lumps_went if feeder.vision else feeder.released
            live_k = [L.k for L in lumps if L.state not in d['end_states'] and L.k in went]
            rec.frame(data, [lumps[k].s_hist[-1][1] for k in (live_k or range(len(lumps)))],
                      overlay(cfg, t, sum(1 for L in lumps if L.pass_t is not None), len(lumps), together['s'],
                              belts.main['f'], face, feeder, station))
    if rec:
        rec.close()

    out = _result(cfg, d, data, blocks, lumps, crossings, together, riding, jam_stop, face, belts,
                  dict(diagnostics.report(data), initial_penetration_m=initial_pen,
                       wall_clock_s=round(time.time() - wall, 1)),
                  feed_rear, feed_front, dt)
    out['outcome']['funnel'] = dict(funnel, two_or_more_s=round(funnel['two_or_more_s'], 2))
    out['provenance'] = provenance(xml, sensing=cfg['sensing'])
    out['outcome']['numerical_screen_passed'] = out['numerics']['ok']
    out['outcome']['stop_model'] = 'run terminates on alarm; no post-alarm braking/coasting simulation'
    out['feeder'] = feeder.report()
    out['transfer'] = dict(transfer.report(feeder), layout=layout)
    out['station'] = station.report()
    out['perception'] = sensors.report()
    out['taken_off'] = takeoff.log
    out['scenario'] = dict(kind=cfg.get('scenario', 'none'), **scenario.info)
    # the whole line is clear when the last lump has landed (in a separator bin, or dropped) or was taken off
    done = all(L.state in d['end_states'] for L in lumps)
    out['outcome'].update(line_clear_s=round(max((L.drop_t for L in lumps if L.drop_t is not None), default=0.), 2)
                          if done else None, sorted=sum(L.state == 'sorted' for L in lumps),
                          taken_off=sum(L.state == 'taken_off' for L in lumps))
    out['measurement_audit'] = measurement_report(out)
    _write(cfg, out, xml, traj, belts.names())
    return out


def _result(cfg, d, data, blocks, lumps, crossings, together, riding, jam_stop, face, belts,
            numerics, feed_rear, feed_front, dt):
    for L, b in zip(lumps, blocks):
        b.update(L.summary())
        b.pop('vertices', None)
    crossings.sort()
    gaps = [round(lumps[kb].enter - lumps[ka].pass_t, 3) if lumps[ka].pass_t is not None else None
            for (ta, ka), (tb, kb) in zip(crossings, crossings[1:])]
    known_gaps = [g for g in gaps if g is not None]
    # a lump counts as through when its TAIL has passed the measuring plane; the front reaching it is
    # recorded separately and is not a clear batch
    front = [L for L in lumps if L.enter is not None]
    tail = [L for L in lumps if L.pass_t is not None]
    left_on_belt = [L.k for L in lumps if L.state not in d['end_states']]   # still on the machine
    dropped = [L.k for L in lumps if L.state == 'dropped']
    yaws = [L.face_yaw for L in lumps if L.face_yaw is not None]
    lane_yaws = [L.lane_yaw for L in lumps if L.lane_yaw is not None]
    cls = ('numerical_failure' if numerics.get('failure')
           else ('face_fault' if jam_stop['reason'] in ('face_reset_fault', 'face_motion_fault')
                 else 'station_fault' if jam_stop['reason'] in STATION_FAULTS else 'jammed') if jam_stop
           else 'incomplete' if left_on_belt
           else 'dropped' if dropped
           else 'abreast_at_cut' if together['s'] > 0
           else 'single_file_after_unjam' if face.events else 'single_file')
    return dict(model='plough_singulator_v1', mujoco=mujoco.__version__,
                finished_utc=datetime.now(timezone.utc).isoformat(),
                config={k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items() if not k.startswith('_')},
                geometry=dict(d['report'], feed_layer_from_m=round(feed_rear, 3), feed_layer_to_m=round(feed_front, 3)),
                outcome=dict(classification=cls, blocks_total=len(lumps), front_arrived=len(front),
                             tail_passed=len(tail), dropped=len(dropped), left_on_belt=left_on_belt,
                             clear_s=(round(max(L.pass_t for L in tail), 2) if len(tail) == len(lumps) else None),
                             first_pass_yield=round(len(tail) / len(lumps), 3),
                             order=[k for _, k in crossings],
                             together_at_cut_s=round(together['s'], 3),
                             together_pairs_s={k: round(v, 3) for k, v in together['pairs'].items()},
                             max_total_cut_width_m=round(together['max_total_n'], 3),
                             riding_over_s=round(riding, 3),
                             net_time_gaps_s=gaps, min_net_time_gap_s=min(known_gaps) if known_gaps else None,
                             unknown_time_gaps=sum(g is None for g in gaps), gap_sampling_uncertainty_s=.02,
                             negative_gaps=sum(g < 0 for g in known_gaps),
                             exit_yaw_misalignment_deg=yaws,
                             exit_yaw_misalignment_median_deg=round(float(np.median(yaws)), 1) if yaws else None,
                             lane_yaw_misalignment_deg=lane_yaws,
                             lane_yaw_misalignment_median_deg=round(float(np.median(lane_yaws)), 1) if lane_yaws else None,
                             touched_plough=sum(1 for L in lumps if L.touched_plough),
                             end_time_s=round(float(data.time), 2), jam_stop=jam_stop,
                             unjam_pulses=len(face.events), unjam_events=face.events, **face.report()),
                drives=belts.report(dt, face.drive),
                numerics=numerics,
                blocks=blocks, feed_material=batch_material_summary(blocks))


def _write(cfg, out, xml, traj, drive_names):
    cfg['out_dir'].mkdir(parents=True, exist_ok=True)
    (cfg['out_dir'] / 'result.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    # the model that was actually compiled: the derived report alone cannot be re-inspected (or re-run)
    (cfg['out_dir'] / 'model.xml').write_text(xml, encoding='utf-8')
    # body pose (pos + quat, which with model.xml reconstructs each lump exactly), centroid, front edge,
    # touched categories as a bit mask over touch_categories, face angle, total lump-face normal force,
    # drive speed factors, load-cell reading and plate angle
    f32 = lambda key: np.array(traj[key], np.float32)
    np.savez_compressed(cfg['out_dir'] / 'trajectory.npz', t=np.array(traj['t']), front_x=f32('front_x'),
                        pos=f32('pos'), quat=f32('quat'), com=f32('com'), touch=np.array(traj['touch'], np.uint16),
                        touch_categories=np.array(CATS), face_deg=f32('face_deg'),
                        face_contact_N=f32('face_contact_N'), drive_f=f32('drive_f'),
                        drive_names=np.array(drive_names), weigh_N=f32('weigh_N'), sep_deg=f32('sep_deg'))
