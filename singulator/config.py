"""The command-line configuration of the line.

One line (user, 2026-09-30: 把当前的完整实验结构当作主线; the gated mainline and the feed-belt-only trial were
deleted that day): a separately driven feed belt a step above the main belt lets the batch go lump by lump,
the plough face and the 0.60 m lane put the lumps in single file, and behind the main belt's head edge a
buffer belt, a measuring belt (weight and volume) and the flip separator sort each lump by its measured
density; the station, the plough face and the alarms read modelled sensor signals (--sensing vision,
singulator/perception.py), the feed belt controller the lumps' true centroids (--feeder-sensing oracle). Every
default below is that line -- the baseline fixed on 2026-10-05; a flag changes one thing about it. Neither
geometric checks nor model clearance establish real-machine reliability.
"""
import argparse
from datetime import datetime
from pathlib import Path

from .feeder import DEFAULTS as FEEDER_DEFAULTS, HEAD_DEFAULTS
from .lumps import FRICTION
from .physics import DEFAULTS as PHYSICS_DEFAULTS, validate
from .station import DEFAULTS as STATION_DEFAULTS

ROOT = Path(__file__).resolve().parents[1]     # project folder
FONT = Path(r'C:\Windows\Fonts\msyh.ttc')

LAYOUTS = ('scatter', 'aligned', 'touching', 'flat', 'oblique')


def parse_config(argv=None):
    """Resolve the same CLI defaults for simulation and assembly-only export."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, value in FEEDER_DEFAULTS.items():
        p.add_argument('--' + key.replace('_', '-'), type=float, default=value)
    p.add_argument('--feeder-head-d', type=float, default=HEAD_DEFAULTS['feeder_head_d'],
                   help='head drum diameter of the feed belt, m; 0 = the idealised sharp head edge (default)')
    p.add_argument('--feeder-tail-d', type=float, default=HEAD_DEFAULTS['feeder_tail_d'],
                   help='tail pulley diameter of the main belt under the feed belt head, m; 0 = a square plate end')
    p.add_argument('--feeder-handover', type=float, default=HEAD_DEFAULTS['feeder_handover'],
                   help="horizontal distance from the head drum's top tangent to where the main belt runs flat, m "
                        "(the hand-over position; at least what the drum and pulley sizes need, see "
                        "feeder.min_handover). With --feeder-step these make the transfer adjustable")
    p.add_argument('--feeder-release', choices=('lane', 'predict'), default='predict',
                   help="when the feed belt lets the next lump go: 'predict' (default) stages the next lump at the "
                        "creep zone and lets it go as soon as the ones before are predicted to be out of the funnel "
                        "when it gets there; 'lane' once every released lump is in the straight lane")
    for key, value in STATION_DEFAULTS.items():
        p.add_argument('--' + key.replace('_', '-'), type=float, default=value)
    p.add_argument('--weigh-stop', choices=('rear', 'centre'), default='rear',
                   help="where the measuring belt stops an item: 'rear' (default) = as soon as it is wholly on the "
                        "belt (its rear 5-12 cm past the joint); 'centre' = in the middle of the weigh zone, so a "
                        "round lump that rocks after the stop does not roll back to the joint")
    p.add_argument('--buffer-approach', choices=('full', 'slow'), default='full',
                   help="how the buffer belt brings its lead item to the joint: 'full' (default) = at its own speed; "
                        "'slow' = at the measuring belt's speed over the last 0.45 m (it stops shorter at S3 and is "
                        "not thrown onto the slower belt)")
    p.add_argument('--sensing', choices=('oracle', 'vision'), default='vision',
                   help="what the controllers read: 'vision' (default) = modelled sensor signals (estimated "
                        "outlines, velocity and confidence from the cameras, debounced beams with diagnoses, the "
                        "scanner's verdict; singulator/perception.py); 'oracle' = the true state, for comparison")
    p.add_argument('--feeder-sensing', choices=('oracle', 'vision'), default='oracle',
                   help="what the feed belt controller reads (with --sensing vision): 'oracle' (default since "
                        "2026-10-05, user: 给料直接读质心) = one object per lump with its true centroid and outline; "
                        "'vision' = the cameras' objects like the rest of the line (lumps lying against each other "
                        "are one object, led by its front edge). The drop beam is the stop signal either way")
    p.add_argument('--weigh-model', choices=('fixed', 'steady'), default='fixed',
                   help="the measuring device on M: 'fixed' (default since 2026-10-05, user: 称重和测体积是固定耗时 "
                        "1 s 的概念装置) = mass and volume are read scan_s after M is at rest, whatever the reading "
                        "does; 'steady' = the indicator has to be steady first, and an item whose reading is not "
                        "steady within 3 s is void ('unsteady')")
    p.add_argument('--fault', action='append', default=[],
                   help="inject a sensor fault, repeatable: scan_fail:N (the N-th scan, from 0, fails), "
                        "beam_dirty:NAME@T (reads blocked from T s), beam_dead:NAME@T (never blocks from T s), "
                        "vision_off:T0-T1 (every camera dark). NAME: beam_feed, beam_in, beam_stop, beam_gangue")
    p.add_argument('--layout', choices=LAYOUTS, default='scatter',
                   help="how the batch lies on the feed belt: 'scatter' (default: random x, y and yaw, one layer, "
                        "lumps may touch), or a bench layout (singulator/trial.py): 'aligned' (并齐), 'touching' "
                        "(相贴), 'flat' (扁平), 'oblique' (斜放)")
    p.add_argument('--bench', action='store_true',
                   help='the feed head bench trial (designs/transfer_trial/): the run ends once every lump lies on '
                        'the main belt below the feed belt head')
    p.add_argument('--scenario', choices=('none', 'touching', 'touching_lane', 'off_scale_neighbour', 'off_scale_bridge'),
                   default='none',
                   help="acceptance scenarios (2026-09-30): 'touching' = two lumps laid touching end to end on the "
                        "buffer belt at t = 0, they reach the measuring belt together; 'touching_lane' = the same "
                        "pair laid in the lane; 'off_scale_neighbour' = while the first item is weighed a lump is "
                        "laid across the buffer / measuring belt joint against it; 'off_scale_bridge' = the lump "
                        "being weighed is pushed back across the joint. Scan failure: --fault scan_fail:0")
    p.add_argument('--face-kind', choices=('plate', 'rollers'), default='rollers',
                   help="'rollers' (default): the plough face is a row of free-spinning vertical rollers (D 90 mm "
                        "at 0.11 m pitch, 0.50 m tall) the lumps ROLL against, which removes the sliding friction "
                        "that queues lumps, spins them and anchors an arch on that abutment; 'plate': a steel "
                        "plate they SLIDE along. The lane's outer wall and the skirts stay plates")
    p.add_argument('--curve-top-deg', type=float, default=55.,
                   help='the face turns from this angle to the belt travel at the belt edge (shearing abreast pairs '
                        'apart on the steep top) to --curve-exit-deg at the lane. A plate face must stay below '
                        'atan(1/friction-steel) here -- 65.8 deg on steel -- or lumps lock on it instead of '
                        'sliding along it')
    p.add_argument('--curve-exit-deg', type=float, default=20.,
                   help='angle of the face at the lane, i.e. the direction lumps enter the lane with')
    p.add_argument('--belt-w', type=float, default=1.2, help='clear belt width the stream arrives on')
    p.add_argument('--lane-y', type=float, default=.05, help='y of the low skirt, the lane datum')
    p.add_argument('--lane-w', type=float, default=.60,
                   help='lane clear width = the outlet (0.60 m, user 2026-10-05; 0.65 m from 2026-09-23 until then). '
                        'The feed head lets about one release in ten go as two lumps, and a pair that fits through '
                        'the lane abreast reaches the scale together: 11.4 %% of pairs fit at 0.65 m, 2.9 %% at '
                        '0.60 m, 1.4 %% at 0.58 m (EXPERIMENTS.md S5); the narrower the lane, the less room a lump '
                        'coming at an angle has (S5: one jam at the lane mouth in 40 batches at 0.58 m)')
    p.add_argument('--lane-len', type=float, default=.9, help='straight lane between the bend and the exit plane')
    p.add_argument('--plough-x', type=float, default=.50, help='x of the high end of the face, at the belt edge')
    p.add_argument('--v-belt', type=float, default=.40, help='main belt speed, m/s')
    p.add_argument('--friction-belt', type=float, default=FRICTION['block_belt'],
                   help='lump on the rubber belt cover')
    p.add_argument('--friction-steel', type=float, default=FRICTION['block_steel'],
                   help='lump on steel: a plate face (--face-kind plate), the lane wall and the skirts; 0.20 '
                        'stands for a UHMW-PE lining')
    p.add_argument('--friction-block', type=float, default=FRICTION['block_block'],
                   help='lump on lump')
    p.add_argument('--dampratio', type=float, default=1.,
                   help='soft-contact damping ratio, uncalibrated; not a measured restitution coefficient')
    for key, value in PHYSICS_DEFAULTS.items():
        p.add_argument('--' + key.replace('_', '-'), type=type(value), default=value)
    p.add_argument('--size-min', type=float, default=.30,
                   help='screen size class of the feed, m: the square aperture the lump passed, which bounds '
                        'its INTERMEDIATE axis. The long axis follows from the shape family (size/b, up to '
                        '2.2x) and is limited by --size-long-max')
    p.add_argument('--size-max', type=float, default=.50)
    p.add_argument('--size-long-max', type=float, default=.50,
                   help='cap on the LONG axis, m (user, 2026-09-20): stands for the upstream picking or '
                        'breaking that keeps the longest slivers out of a 300-500 mm class. It caps the extent '
                        'along the body x axis, NOT the largest caliper (Feret) size: at 0.50 m about 6 %% of '
                        'lumps still span more than 0.55 m in plan (max ~0.63 m), and two lumps can wedge in '
                        'the lane whatever their size. Pass a large value for an uncapped screen class')
    p.add_argument('--count', type=int, default=5, help='lumps per batch (3-5 in service)')
    p.add_argument('--gangue-fraction', type=float, default=.3,
                   help='gangue-rich category probability by COUNT, not batch gangue mass fraction')
    p.add_argument('--middlings-fraction', type=float, default=.25,
                   help='intergrown category probability by COUNT; coal probability = 1 - gangue - middlings')
    for name, limits in (('coal', (1250., 1450.)), ('gangue', (2250., 2700.))):
        for bound, value in zip(('min', 'max'), limits):
            p.add_argument('--%s-density-%s' % (name, bound), type=float, default=value,
                           help='effective dry endmember particle density, kg/m3; composition model only')
    p.add_argument('--feed-len', type=float, default=1.2,
                   help='length of the scatter patch, m; it grows in 0.2 m steps until the batch fits in '
                        'one layer, so a small value just means "as dense as it will pack"')
    p.add_argument('--side-belt', dest='side_belt', action='store_true', default=True,
                   help='a driven side belt in place of the static low-side skirt over the funnel and lane (default)')
    p.add_argument('--no-side-belt', dest='side_belt', action='store_false',
                   help='the static low-side skirt instead of the side belt')
    p.add_argument('--side-belt-ratio', type=float, default=1.5,
                   help='side-belt surface speed / main belt speed; below 1.0 it is a brake')
    p.add_argument('--side-belt-back', type=float, default=.92,
                   help='how far upstream of the bend the side belt starts (0.92 m: over the whole funnel -- an arch '
                        'needs two abutments, the roller face removed the upper one and the side belt the lower one)')
    p.add_argument('--side-belt-force-max', type=float, default=15000.,
                   help='side belt drive force limit, N (1.5 kN stalled in most jammed batches)')
    p.add_argument('--belt-force-max', type=float, default=3000.,
                   help='belt drive force limit at belt speed; the plough drag is mu*(mass on the face)*g')
    p.add_argument('--belt-mass', type=float, default=200.)
    p.add_argument('--motor-slip', type=float, default=.05,
                   help='speed droop at the force limit; note the droop line gain is force-max/slip, so raising '
                        'the limit also stiffens the drive (1.5 -> 15 kN at 5%% slip is a 10x stiffer drive)')
    p.add_argument('--dt', type=float, default=.00025,
                   help='physics step (s); default 0.25 ms after paired refinement exposed 0.5 ms sensitivity')
    p.add_argument('--solref', type=float, default=.002, help='contact time constant; must stay >= 2*dt')
    p.add_argument('--ramp', type=float, default=.4)
    p.add_argument('--duration', type=float, default=150.)
    p.add_argument('--face-swing-deg', type=float, default=-25.,
                   help='swing angle of the face about its hinge at the bend, CCW positive in the top view (+x flow '
                        'right, +y up); negative opens the funnel')
    p.add_argument('--face-swing-s', type=float, default=1., help='time to swing out, and again to swing back')
    p.add_argument('--face-hold-s', type=float, default=.3,
                   help='minimum time the face stays retracted before it may close')
    p.add_argument('--face-hold-max-s', type=float, default=20.,
                   help='the face waits, retracted, until the section is clear and no lump reaches into the '
                        'area it sweeps on its way home, and only then closes (pausing if a lump enters that area '
                        'while it closes). Not allowed to close after this long: the run stops with the face '
                        'retracted (jammed / retract_hold_timeout); it never closes onto material')
    p.add_argument('--face-force-max', type=float, default=10000.,
                   help='face servo torque limit divided by the chord, N; assumed, not cylinder sizing. '
                        'Default physical hinge includes carrier/roller inertia and soft joint stops')
    p.add_argument('--unjam-max', type=int, default=4,
                   help='retract actions per batch before a further stall counts as a jam (0: the first detected '
                        'stall ends the run)')
    p.add_argument('--jam-window', type=float, default=4.,
                   help='a lump in the section (plough start - 0.3 m .. lane exit) whose centroid advances less '
                        'than jam-speed x jam-window over this window, with no tail leaving the lane, is a stall')
    p.add_argument('--jam-speed', type=float, default=.02)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no-video', dest='video', action='store_false')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--video-speed', type=float, default=2.)
    p.add_argument('--name', default='pl')
    a = p.parse_args(argv)
    cfg = vars(a).copy()
    cfg.update(width=1280, height=1000, top_h=340,
               out_dir=ROOT / 'runs' / ('%s_%s' % (a.name, datetime.now().strftime('%Y%m%dT%H%M%S'))))
    return validate(cfg)
