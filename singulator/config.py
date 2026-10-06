"""Every parameter of the line, defined once.

One line (user, 2026-09-30: 把当前的完整实验结构当作主线): a separately driven feed belt a step above the main
belt lets the batch go lump by lump, the plough face and the 0.60 m lane put the lumps in single file, and behind
the main belt's head edge a buffer belt, a measuring belt (weight and volume) and the flip separator sort each lump
by its measured density. The defaults below are that line -- the baseline fixed on 2026-10-05; a flag changes one
thing about it. Neither geometric checks nor model clearance establish real-machine reliability.

PARAMS is the table: name, default, type (or choices), help, grouped by what they belong to. From it

    parse_config(argv)        builds the command line (python plough.py --help lists the groups);
    default_config(**over)    gives code the same configuration (checks, experiments, a notebook);
    validate(cfg)             is applied by both.

There are no other defaults: code reads cfg['name'] and never falls back to a value of its own. A configuration is
a plain dict (it is written to result.json as it is); validate() refuses one with a missing or an unknown name.

Tunables -- the margins and thresholds that live as constants next to the code they belong to -- are not
parameters. --set changes one for a run and records it (singulator/tuning.py).
"""
import argparse
import math
from collections import namedtuple
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]     # project folder
SAMPLE_S = .01      # s: the sensors, the controllers and the records run on this sample; dt has to divide it

P = namedtuple('P', 'name default kind help', defaults=('',))
FLAG, LIST = 'flag', 'list'     # kinds beside a type or a tuple of choices: --name / --no-name; repeatable --name

LAYOUTS = ('scatter', 'aligned', 'touching', 'flat', 'oblique')
SCENARIOS = ('none', 'touching', 'touching_lane', 'off_scale_neighbour', 'off_scale_bridge')

PARAMS = {
    'the batch': [
        P('seed', 1, int),
        P('count', 5, int, 'lumps per batch (3-5 in service)'),
        P('size_min', .30, float,
          'screen size class of the feed, m: the square aperture the lump passed, which bounds its INTERMEDIATE '
          'axis. The long axis follows from the shape family (size/b, up to 2.2x) and is limited by --size-long-max'),
        P('size_max', .50, float),
        P('size_long_max', .50, float,
          'cap on the LONG axis, m (user, 2026-09-20): stands for the upstream picking or breaking that keeps the '
          'longest slivers out of a 300-500 mm class. It caps the extent along the body x axis, NOT the largest '
          'caliper (Feret) size: at 0.50 m about 6 %% of lumps still span more than 0.55 m in plan (max ~0.63 m), '
          'and two lumps can wedge in the lane whatever their size. Pass a large value for an uncapped screen class'),
        P('gangue_fraction', .3, float, 'gangue-rich category probability by COUNT, not batch gangue mass fraction'),
        P('middlings_fraction', .25, float,
          'intergrown category probability by COUNT; coal probability = 1 - gangue - middlings'),
        P('coal_density_min', 1250., float, 'effective dry endmember particle density, kg/m3'),
        P('coal_density_max', 1450., float, 'effective dry endmember particle density, kg/m3'),
        P('gangue_density_min', 2250., float, 'effective dry endmember particle density, kg/m3'),
        P('gangue_density_max', 2700., float, 'effective dry endmember particle density, kg/m3'),
        P('layout', 'scatter', LAYOUTS,
          "how the batch lies on the feed belt: 'scatter' (default: random x, y and yaw, one layer, lumps may "
          "touch), or a bench layout (singulator/sim/layouts.py): 'aligned' (并齐), 'touching' (相贴), 'flat' "
          "(扁平), 'oblique' (斜放)"),
        P('feed_len', 1.2, float,
          'length of the scatter patch, m; it grows in 0.2 m steps until the batch fits in one layer, so a small '
          'value just means "as dense as it will pack"'),
        P('feed_drop_height', .006, float, 'the batch is laid this far above the feed belt, m'),
    ],
    'feed belt': [
        P('feeder_speed', .10, float, 'feed belt speed on approach, m/s; below the main belt speed'),
        P('feeder_creep_speed', .03, float, 'feed belt speed once the lead is in the creep zone, m/s'),
        P('feeder_step', .10, float, 'the feed belt top is this far above the main belt top, m'),
        P('feeder_len', 2.2, float),
        P('feeder_gap', .80, float, 'main belt between the feed belt head edge and the plough start, m'),
        P('feeder_ramp_s', .30, float, 'start / stop ramp of the feed belt drive, s'),
        P('feeder_stall_s', .50, float, 'a released lump with no progress over this long hangs on the edge, s'),
        P('feeder_force_max', 3000., float, 'feed belt drive (and brake) force limit, N'),
        P('feeder_head_d', 0., float,
          'head drum diameter of the feed belt, m; 0 = the idealised sharp head edge (default)'),
        P('feeder_tail_d', 0., float,
          'tail pulley diameter of the main belt under the feed belt head, m; 0 = a square plate end'),
        P('feeder_handover', 0., float,
          "horizontal distance from the head drum's top tangent to where the main belt runs flat, m (the hand-over "
          "position; at least what the drum and pulley sizes need, see machine.feed_belt.min_handover). With "
          "--feeder-step these make the transfer adjustable"),
        P('feeder_release', 'predict', ('lane', 'predict'),
          "when the feed belt lets the next lump go: 'predict' (default) stages the next lump at the creep zone and "
          "lets it go as soon as the ones before are predicted to be out of the funnel when it gets there; 'lane' "
          "once every released lump is in the straight lane"),
        P('feeder_sensing', 'oracle', ('oracle', 'vision'),
          "what the feed belt controller reads (with --sensing vision): 'oracle' (default since 2026-10-05, user: "
          "给料直接读质心) = one object per lump with its true centroid and outline; 'vision' = the cameras' objects "
          "like the rest of the line (lumps lying against each other are one object, led by its front edge). On the "
          "camera path the drop beam stops the feed belt; with the true centroids --feeder-stop says what does"),
        P('feeder_stop', 'centroid', ('beam', 'centroid'),
          "what ends a release once a lump is over the head edge: 'beam' = the feed belt creeps on until the lump "
          "tips into the drop beam S1; 'centroid' = it stops once the lump's centroid is control.feeder.STOP_PAST "
          "(3 mm) past the edge and the lump tips on its own (S8, docs/TODO.md section 1). 'centroid' needs the true "
          "centroid: on the camera path (--feeder-sensing vision) the beam ends the release whatever this says"),
    ],
    'main belt, plough face, lane': [
        P('belt_w', 1.2, float, 'clear belt width the stream arrives on'),
        P('v_belt', .40, float, 'main belt speed, m/s'),
        P('plough_x', .50, float, 'x of the high end of the face, at the belt edge'),
        P('face_kind', 'rollers', ('plate', 'rollers'),
          "'rollers' (default): the plough face is a row of free-spinning vertical rollers (D 90 mm at 0.11 m pitch, "
          "0.50 m tall) the lumps ROLL against, which removes the sliding friction that queues lumps, spins them "
          "and anchors an arch on that abutment; 'plate': a steel plate they SLIDE along. The lane's outer wall and "
          "the skirts stay plates"),
        P('curve_top_deg', 55., float,
          'the face turns from this angle to the belt travel at the belt edge (shearing abreast pairs apart on the '
          'steep top) to --curve-exit-deg at the lane. A plate face must stay below atan(1/friction-steel) here -- '
          '65.8 deg on steel -- or lumps lock on it instead of sliding along it'),
        P('curve_exit_deg', 20., float,
          'angle of the face at the lane, i.e. the direction lumps enter the lane with'),
        P('lane_y', .05, float, 'y of the low skirt, the lane datum'),
        P('lane_w', .60, float,
          'lane clear width = the outlet (0.60 m, user 2026-10-05; 0.65 m from 2026-09-23 until then). The feed head '
          'lets about one release in ten go as two lumps, and a pair that fits through the lane abreast reaches the '
          'scale together: 11.4 %% of pairs fit at 0.65 m, 2.9 %% at 0.60 m, 1.4 %% at 0.58 m (EXPERIMENTS.md S5); '
          'the narrower the lane, the less room a lump coming at an angle has (S5: one jam at the lane mouth in 40 '
          'batches at 0.58 m)'),
        P('lane_len', .9, float, 'straight lane between the bend and the exit plane'),
        P('side_belt', True, FLAG,
          'a driven side belt in place of the static low-side skirt over the funnel and lane (default); '
          '--no-side-belt: the static skirt'),
        P('side_belt_ratio', 1.5, float, 'side-belt surface speed / main belt speed; below 1.0 it is a brake'),
        P('side_belt_back', .92, float,
          'how far upstream of the bend the side belt starts (0.92 m: over the whole funnel -- an arch needs two '
          'abutments, the roller face removed the upper one and the side belt the lower one)'),
    ],
    'face retract': [
        P('face_swing_deg', -25., float,
          'swing angle of the face about its hinge at the bend, CCW positive in the top view (+x flow right, +y '
          'up); negative opens the funnel'),
        P('face_swing_s', 1., float, 'time to swing out, and again to swing back'),
        P('face_hold_s', .3, float, 'minimum time the face stays retracted before it may close'),
        P('face_hold_max_s', 20., float,
          'the face waits, retracted, until the section is clear and no lump reaches into the area it sweeps on its '
          'way home, and only then closes (pausing if a lump enters that area while it closes). Not allowed to '
          'close after this long: the run stops with the face retracted (jammed / retract_hold_timeout); it never '
          'closes onto material'),
        P('face_force_max', 10000., float,
          'face servo torque limit divided by the chord, N; assumed, not cylinder sizing. The hinge includes '
          'carrier/roller inertia and soft joint stops'),
        P('face_kp', 30000., float, 'face servo position gain, N m/rad (assumed)'),
        P('face_kv', 3000., float, 'face servo velocity gain, N m s/rad (assumed)'),
        P('face_carrier_mass', 20., float, 'mass of the beam that carries the face, kg (assumed)'),
        P('face_position_tol', .002, float, 'the face is in position within this, rad'),
        P('face_velocity_tol', .01, float, '... and at rest below this, rad/s'),
        P('face_bearing_drag', 0., float, 'constant bearing drag torque of each face roller, N m'),
        P('unjam_max', 4, int,
          'retract actions per batch before a further stall counts as a jam (0: the first detected stall ends the '
          'run)'),
        P('jam_window', 4., float,
          'a lump in the section (plough start - 0.3 m .. lane exit) whose centroid advances less than jam-speed x '
          'jam-window over this window, with no tail leaving the lane, is a stall'),
        P('jam_speed', .02, float),
    ],
    'buffer belt, measuring belt, separator': [
        P('station_w', .70, float, 'width of the buffer and measuring belts and of the separator plate, m'),
        P('station_step', .10, float, 'their tops are this far below the main belt top, m'),
        P('buffer_len', 1.5, float),
        P('buffer_speed', .80, float, 'buffer belt speed, m/s: faster than the main belt, it parts lumps'),
        P('measure_len', .90, float),
        P('station_speed', .40, float, 'measuring belt speed, m/s'),
        P('scan_s', 1., float, 'the measuring device takes this long from the belt at rest, s'),
        P('weigh_model', 'fixed', ('fixed', 'steady'),
          "the weighing device on the measuring belt (singulator/sensing/weigher.py): 'fixed' (default since "
          "2026-10-05, user: 称重和测体积是固定耗时 1 s 的概念装置) = the mass is read scan_s after the belt is at "
          "rest, whatever the reading does; 'steady' = the indicator has to be steady first, and an item whose "
          "reading is not steady within 3 s is void ('unsteady')"),
        P('weigh_stop', 'rear', ('rear', 'centre'),
          "where the measuring belt stops an item: 'rear' (default) = as soon as it is wholly on the belt (its rear "
          "5-12 cm past the joint); 'centre' = in the middle of the weigh zone, so a round lump that rocks after the "
          "stop does not roll back to the joint"),
        P('buffer_approach', 'full', ('full', 'slow'),
          "how the buffer belt brings its lead item to the joint: 'full' (default) = at its own speed; 'slow' = at "
          "the measuring belt's speed over the last 0.45 m (it stops shorter at S3 and is not thrown onto the slower "
          "belt)"),
        P('sort_density', 1800., float, 'gangue at or above this measured density, kg/m3 (placeholder)'),
        P('separator_swing_s', 2., float, 'time the plate takes between its two positions, s'),
        P('separator_friction', .20, float, 'lump on the plate deck: 0.20 = lined with UHMW-PE'),
        P('separator_wall', .25, float, 'side walls on the plate, m high (0 = none)'),
    ],
    'sensing, faults, scenarios': [
        P('sensing', 'vision', ('oracle', 'vision'),
          "what the controllers read: 'vision' (default) = modelled sensor signals (estimated outlines, velocity "
          "and confidence from the cameras, debounced beams with diagnoses, the scanner's verdict; "
          "singulator/sensing/); 'oracle' = the same controllers on ideal sensors (the true state), for comparison"),
        P('fault', [], LIST,
          "inject a sensor fault, repeatable: scan_fail:N (the N-th scan, from 0, fails), beam_dirty:NAME@T (reads "
          "blocked from T s), beam_dead:NAME@T (never blocks from T s), vision_off:T0-T1 (every camera dark). NAME: "
          "beam_feed, beam_in, beam_stop, beam_gangue"),
        P('scenario', 'none', SCENARIOS,
          "acceptance scenarios (2026-09-30): 'touching' = two lumps laid touching end to end on the buffer belt at "
          "t = 0, they reach the measuring belt together; 'touching_lane' = the same pair laid in the lane; "
          "'off_scale_neighbour' = while the first item is weighed a lump is laid across the buffer / measuring "
          "belt joint against it; 'off_scale_bridge' = the lump being weighed is pushed back across the joint. Scan "
          "failure: --fault scan_fail:0"),
        P('bench', False, FLAG,
          'the feed head bench trial (designs/transfer_trial/): the run ends once every lump lies on the main belt '
          'below the feed belt head'),
    ],
    'contact and drives (uncalibrated)': [
        P('friction_belt', .55, float, 'lump on the rubber belt cover (coal or rock on rubber, 0.45-0.70)'),
        P('friction_steel', .45, float,
          'lump on steel: a plate face (--face-kind plate), the lane wall and the skirts (coal on mild steel, '
          '0.35-0.60); 0.20 stands for a UHMW-PE lining'),
        P('friction_block', .60, float, 'lump on lump (coal on coal / rock on rock sliding, 0.50-0.80)'),
        P('dampratio', 1., float,
          'soft-contact damping ratio, uncalibrated; not a measured restitution coefficient'),
        P('contact_condim', 3, int, '3 = sliding friction; 4 adds torsional, 6 rolling friction'),
        P('torsional_friction', 0., float, 'm; needs --contact-condim 4 or 6'),
        P('rolling_friction', 0., float, 'm; needs --contact-condim 6'),
        P('belt_force_max', 3000., float,
          'belt drive force limit at belt speed; the plough drag is mu*(mass on the face)*g'),
        P('side_belt_force_max', 15000., float, 'side belt drive force limit, N (1.5 kN stalled in most jammed batches)'),
        P('belt_mass', 200., float, 'reflected mass of the main belt drive, kg'),
        P('motor_slip', .05, float,
          'speed droop at the force limit; note the droop line gain is force-max/slip, so raising the limit also '
          'stiffens the drive (1.5 -> 15 kN at 5%% slip is a 10x stiffer drive)'),
    ],
    'run and numerics': [
        P('duration', 150., float, 'time limit of the batch, s'),
        P('ramp', .4, float, 'start-up ramp of the main and side belts, s'),
        P('dt', .00025, float,
          'physics step (s); default 0.25 ms after paired refinement exposed 0.5 ms sensitivity'),
        P('solref', .002, float, 'contact time constant; must stay >= 2*dt'),
        P('solver_iterations', 100, int),
        P('solver_tolerance', 1e-8, float),
        P('penetration_limit', .005, float, 'numerical screen: deeper contact penetration than this fails it, m'),
        P('set', [], LIST,
          "change a tunable (a module constant of the package) for this run, repeatable: MODULE.NAME=VALUE, e.g. "
          "--set control.station.APPROACH=.65 (singulator/tuning.py). The result records it"),
    ],
    'output': [
        P('name', 'pl', str, 'the outputs go to runs/<name>_<time>/'),
        P('video', True, FLAG, 'render video.mp4 (default); --no-video: results only'),
        P('fps', 30, int),
        P('video_speed', 1., float, 'playback speed of video.mp4 against the simulation (1 = real time)'),
        P('width', 1280, int, 'video frame, pixels'),
        P('height', 1000, int),
        P('top_h', 340, int, 'height of the plan view strip at the bottom of the frame, pixels'),
    ],
}
NAMES = {p.name: p for group in PARAMS.values() for p in group}


def _parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\nPARAMS')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for title, group in PARAMS.items():
        g = ap.add_argument_group(title)
        for p in group:
            flag = '--' + p.name.replace('_', '-')
            if p.kind == FLAG:
                g.add_argument(flag, dest=p.name, action=argparse.BooleanOptionalAction, default=p.default,
                               help=p.help or None)
            elif p.kind == LIST:
                g.add_argument(flag, dest=p.name, action='append', default=None, help=p.help or None)
            elif isinstance(p.kind, tuple):
                g.add_argument(flag, dest=p.name, choices=p.kind, default=p.default,
                               help=(p.help + ' ' if p.help else '') + '(default: %s)' % p.default)
            else:
                g.add_argument(flag, dest=p.name, type=p.kind, default=p.default,
                               help=(p.help + ' ' if p.help else '') + '(default: %s)' % (p.default,))
    return ap


def _finish(cfg):
    cfg = {name: (list(cfg[name] or []) if p.kind == LIST else cfg[name]) for name, p in NAMES.items()} | {
        k: v for k, v in cfg.items() if k not in NAMES}
    cfg.setdefault('out_dir', ROOT / 'runs' / ('%s_%s' % (cfg['name'], datetime.now().strftime('%Y%m%dT%H%M%S'))))
    return validate(cfg)


def parse_config(argv=None):
    """The configuration of a command line (argv=None: this process's)."""
    return _finish(vars(_parser().parse_args(argv)))


def default_config(**over):
    """The line's defaults with `over` on top: default_config(lane_w=.58, layout='aligned', video=False).
    out_dir may be given too (default: runs/<name>_<time>/ in the project folder)."""
    unknown = sorted(set(over) - set(NAMES) - {'out_dir'})
    if unknown:
        raise ValueError('not a parameter of the line: %s' % ', '.join(unknown))
    return _finish({name: p.default for name, p in NAMES.items()} | over)


def validate(cfg):
    """Refuse a configuration that is incomplete, misspelt or inconsistent; returns it."""
    missing, unknown = sorted(set(NAMES) - set(cfg)), sorted(set(cfg) - set(NAMES) - {'out_dir'})
    if missing or unknown:
        raise ValueError('configuration%s%s; build it with config.default_config(...) or parse_config(...)'
                         % (' lacks ' + ', '.join(missing) if missing else '',
                            ('; ' if missing else ' ') + 'has unknown ' + ', '.join(unknown) if unknown else ''))
    c = cfg
    for name, p in NAMES.items():
        if isinstance(p.kind, tuple) and c[name] not in p.kind:
            raise ValueError('%s must be one of %s, got %r' % (name, ', '.join(p.kind), c[name]))
        if p.kind is float and not math.isfinite(c[name]):
            raise ValueError('%s must be finite' % name)
    positive = ('dt', 'duration', 'solref', 'dampratio', 'v_belt', 'belt_mass', 'motor_slip',
                'size_min', 'size_max', 'size_long_max', 'belt_force_max',
                'side_belt_force_max', 'face_force_max', 'face_swing_s',
                'jam_window', 'feed_len', 'solver_tolerance', 'penetration_limit',
                'face_kp', 'face_kv', 'face_carrier_mass', 'face_position_tol', 'face_velocity_tol')
    nonnegative = ('friction_belt', 'friction_steel', 'friction_block', 'ramp',
                   'jam_speed', 'torsional_friction', 'rolling_friction',
                   'feed_drop_height', 'face_bearing_drag')
    for key in positive + nonnegative:
        x = c[key]
        if not math.isfinite(x) or (x <= 0 if key in positive else x < 0):
            raise ValueError('%s must be finite and %s' % (key, 'positive' if key in positive else 'nonnegative'))
    # the batch
    gangue, middlings = c['gangue_fraction'], c['middlings_fraction']
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (gangue, middlings)) or gangue + middlings > 1:
        raise ValueError('gangue_fraction and middlings_fraction must be in [0, 1] and sum <= 1')
    for name in ('coal', 'gangue'):
        lo, hi = c[name + '_density_min'], c[name + '_density_max']
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
            raise ValueError('%s density bounds must be finite, positive and ordered' % name)
    if c['size_min'] > c['size_max'] or c['size_long_max'] < c['size_max']:
        raise ValueError('need size_min <= size_max <= size_long_max (otherwise the long-axis cap is false)')
    for key in ('count', 'solver_iterations'):
        if int(c[key]) != c[key] or c[key] < 1:
            raise ValueError('%s must be a positive integer' % key)
    if int(c['unjam_max']) != c['unjam_max'] or c['unjam_max'] < 0:
        raise ValueError('unjam_max must be a non-negative integer')
    # contact and numerics
    if c['contact_condim'] not in (3, 4, 6):
        raise ValueError('contact_condim must be 3, 4 or 6')
    if c['torsional_friction'] and c['contact_condim'] < 4:
        raise ValueError('torsional friction requires contact_condim >= 4')
    if c['rolling_friction'] and c['contact_condim'] < 6:
        raise ValueError('rolling friction requires contact_condim = 6')
    if c['solref'] < 2 * c['dt']:
        raise ValueError('solref must be >= 2*dt; changing dt must not silently change contact stiffness')
    if c['dt'] > SAMPLE_S or abs(SAMPLE_S / c['dt'] - round(SAMPLE_S / c['dt'])) > 1e-7:
        raise ValueError('dt must divide the 10 ms controller interval exactly')
    if not 0 < abs(c['face_swing_deg']) < 90:
        raise ValueError('face_swing_deg must have a nonzero magnitude below 90 degrees')
    return cfg
