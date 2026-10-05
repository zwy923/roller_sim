"""Measuring and sorting station behind the lane: the second half of the line (third version 2026-09-30).

    lane -> main belt head edge -> buffer belt (one step down) -> measuring belt (weigh + volume) -> flip separator

History. First version (2026-09-24 morning): weigh belt -> volume belt, the whole section upstream stopped
for every lump. Second version (afternoon): the former weigh belt became a buffer, weighing and the volume
scan share one belt, sensors as hardware, the plate moves as soon as the density is known. Third version
(2026-09-30, user): 优先让无效测量真正停止分选 -- a void measurement no longer only gets a flag and is sorted
anyway; and the controllers read what real sensors deliver (singulator/perception.py), not the true state.
Same day (user): the station became the line itself (the gated mainline was deleted), the step onto B went to
10 cm, the separator plate got side walls, and B got its own speed (buffer_speed) so that it can run faster
than the main belt and pull apart lumps that come over the head edge together.

Hardware as modelled:
  * buffer belt B: station_w wide, buffer_len long, its top station_step BELOW the main belt, from the main
    belt's head edge, running at buffer_speed; a lump tips onto it once its centroid passes that edge. Entry
    beam S2 across B S2_X past the edge; staging beam S3 far enough before B's far end for B to stop a lead
    that cuts it (STOP_BACK at least), STOP_Z above it;
  * measuring belt M: same width and height as B, measure_len long, a flat joint with B. Load cells carry M
    and its skirts (the reading is every contact force on them, averaged over the physics steps of each
    10 ms sample); the volume scanner (three depth heads) over it;
  * the flip separator (singulator/separator.py), as wide as M, with side walls (separator_wall high) riding
    on the plate, behind it; beam S4 across the drop gap under the plate inlet; camera C3 over the plate.

What the controllers read (--sensing vision): the camera objects of perception.Vision (estimated outline,
extents, outline centroid, velocity, confidence, estimated lump count, track lineage), the beams S1-S4 as
debounced signals, the load-cell reading, the scanner's verdict, the belt drives' own speed and the plate
angle. The true state is used only to write the verification (keys starting with '_' / 'truth').

Items. An object whose outline centroid passes the head edge joins an item: the item of an object it
descends from (track lineage), else the last item on B if it is closer than GROUP_GAP behind it, else a
new one. Two items whose objects run together become one. An item on B whose objects have drawn apart by
GROUP_GAP (B running faster than the main belt pulls lumps that came over the edge together apart) becomes
two items, each keeping only its own objects.

Void = the measurement cannot stand for one lump. Reasons (all recorded, any one is enough):
  multi        the item had two objects at once, or an object the cameras count as two lumps;
  handover     S2 did not confirm the hand-over as one lump: no interruption over the stretch of B the item
               passed S2 on, more interruptions than objects, or one interruption blocked over more belt travel
               than a single lump can be;
  outside      while weighing, the item's outline is not inside the weigh zone with ISO_MARGIN, or another
               object is within ISO_GAP of it (it may lean on something off the scale);
  unsteady     no steady reading within WEIGH_MAX_S;
  scan         the scanner's verdict is not usable (failed, not one lump, not wholly in the zone, coverage);
  implausible  measured density outside DENSITY_OK (a part of the weight carried elsewhere reads low);
  tare         the empty-scale reading before this item was off by more than TARE_TOL (something left on M);
  track_lost   the cameras lost the item's object on the way and found it again (identity not certain).
A void item is HELD: M does not discharge it, the plate does not move for it, the feed belt releases nothing
more, B stages its lead at S3 and stops, and the section upstream stops as soon as B cannot take the next
lump (the second version's occupancy rule). On the prototype someone then separates, cleans and re-measures
by hand (no automatic return). The simulation does not model that (user 2026-09-30: 不需要模拟人工操作): it
takes the held item's lumps off the line at once, counts the stop, and the line runs on (took_off).

Control (as in the second version, now on the sensors):
  section upstream (main belt, side belt, feed belt): runs; stops only when B cannot take the next lump
     (an object's centroid within HOLD_ZONE of the head edge while B is stopped or has less than ENTRY_ROOM
     free behind its last object, or an object across the head edge while B is stopped);
  B  runs, except that it stops when its lead item is at S3 -- the beam blocked, or the cameras see its front
     STAGE_CAM past the line (then S3 is suspect) -- and M cannot take it;
  M  runs while empty, receiving or discharging; stops once an item is wholly on it (estimated rear
     CLEAR + V_MARGIN past the joint) and the item before has left; at rest it weighs and scans;
  two optional rules (2026-10-04; EXPERIMENTS.md S5 tried them as a patch, S6 as options; off by default):
     --weigh-stop centre: M carries an item that is wholly on it further, until the item is in the middle of the
        weigh zone (or its front is at the stop position: a long item stops as before). A round lump rocks after
        the stop; stopped with its rear 5-12 cm past the joint it can roll back to the joint ('outside');
     --buffer-approach slow: B runs at M's speed while the front of its lead item is within APPROACH of the
        joint. The lead then meets S3 and the joint at 0.4 m/s whatever M is doing: it stops shorter at S3 (at
        0.8 m/s a large round lump cut the 5 cm high beam late and its nose came to rest over the joint, in the
        scanner's zone: the item being weighed was void), and it is not thrown onto the slower M;
  plate  moves to the route's position the moment the route is known, once everything sent before has left
     its path: S4 has seen a gangue lump fall through and is clear again AND the cameras see the plate zone
     empty (coal: the zone empty). Not cleared within PATH_TIMEOUT_S after its discharge: stop the line.
Alarms that stop everything (all belts, the plate stays): vision not healthy (camera outage, a track lost
and not found again) -- the line resumes once it is healthy again, and stops for good after FREEZE_MAX_S;
a beam diagnosed dirty, dead or blocked too long, and a plate-path timeout -- the run stops there (no
recovery modelled).

Nothing here is calibrated: ideal load cells (sum of contact forces), ideal volume value, placeholder
sensor noise; belts are held plates with a prescribed surface velocity. The sorting threshold, the density
window and every tolerance are placeholders.
"""
import math

import mujoco
import numpy as np

from . import separator
from .lumps import FRICTION
from .perception import CONF_OK, LATENCY_S, POS_SIGMA

DEFAULTS = dict(station_w=.70, buffer_len=1.5, measure_len=.90, station_step=.10, station_speed=.40,
                buffer_speed=.80, scan_s=1.,
                sort_density=1800., separator_swing_s=2., separator_friction=FRICTION['lined_steel'],
                separator_wall=.25)       # m: side walls on the plate (0 = none)
RAMP_S = .30          # s: start / stop ramp of B, M and of the held section upstream
FILTER_S = .30        # s: the indicator's moving average of the load-cell samples ...
STEADY_S = .50        # s: ... is steady when over this long it stays within ...
STEADY_BAND = .01     # ... this fraction of its value (at least STEADY_MIN_N); 60 lumps on the second version all
STEADY_MIN_N = 2.     # N: settled within 0.8 s this way, 5 never did under the old 2 % std rule (rocking contacts)
WEIGH_MAX_S = 3.      # s: at rest this long without a steady reading: void (unsteady)
CLEAR = .02           # m: a lump is wholly on a belt once its rear is this far past the belt's start
FRONT_MARGIN = .05    # m: a lump at rest on M keeps its front this far short of M's head edge
STOP_BACK = .15       # m: the staging beam is at least this far before B's far end (more when B runs fast:
                      # its stop distance plus the camera latency plus STAGE_ROOM) ...
STOP_Z = .05          # ... this high above B (a flat lump is at least 6 cm thick)
STAGE_CAM = .03       # m: the cameras see the lead's front this far past S3 while S3 is clear: stop B anyway
STAGE_ROOM = .05      # m: a staged lead stops at least this far short of B's far end
ACCEPT_AFTER = .30    # m: M takes the next item while discharging once the item on it has its rear this far on M
APPROACH = .45        # m: --buffer-approach slow: B runs at M's speed while its lead's front is this close to the joint
                      # (S3 is 0.22 m before the joint at 0.8 m/s, and B needs 0.09 m to come down to 0.4 m/s)
GROUP_GAP = .10       # m: a lump going over the head edge this close behind the last one joins its item
HOLD_ZONE = .10       # m: a lump whose centroid is this close to the head edge is about to go over it
ENTRY_ROOM = .55      # m: B takes a lump only with this much free belt behind its last lump
GANGUE_X = .25        # m: the gangue beam is this far past M's head edge ...
GANGUE_DZ = .45       # ... and this far below the plate inlet (under the plate's closing sweep)
PLATE_MARGIN_S = .30  # s: the plate is in position this long before the lump's front reaches M's head edge
HANG_S = .50          # s: a lump across the head edge without progress this long gets a creep jog of the section
JOG = .075            # jog speed / belt speed: 0.03 m/s at 0.40, the feed belt's creep
GAP = .02             # m: M's head edge -> plate inlet
DROP = .02            # m: M's top -> rib tops at the plate inlet
PLATE_TOL_DEG = .5    # plate in position within this
PLATE_FAULT = 3.      # plate not in position after this many swing times: fault, stop
STALL_S = 20.         # s: a lump in hand and no station state change this long: stop the run
FORCE_MAX = 2000.     # N: B and M drive limits, assumed (as the take-off belt's)
OBS_DT = .01          # s: observation sample
G = 9.81
# ---- void rules (placeholders, see the module docstring) ----
V_MARGIN = 3 * POS_SIGMA   # m: position margin on camera-judged edges (0 with the oracle view)
ISO_MARGIN = .02      # m: while weighing the item's outline keeps this far inside the weigh zone (x) ...
ISO_GAP = .05         # m: ... and no other object comes this close to it
DENSITY_OK = (1000., 3200.)   # kg/m3: a density outside this is not a lump weighed whole
TARE_TOL = 20.        # N: the empty scale must read within this of zero (about 2 kg) ...
ZERO_BAND = 3.        # N: ... it follows the empty reading only within this band (zero tracking), by 1/ZERO_N ...
ZERO_N = 50           # ... per sample, and only with nothing within ZERO_CLEAR of the measuring belt
ZERO_CLEAR = .10      # m
ZERO_OFF_S = 1.       # s: the empty reading outside the band this long: the scale is not clear (tare fault)
HANDOVER_LEN = .70    # m: one S2 interruption over this much belt travel = more than one lump in a row (a
                      # single lump spans at most ~0.63 m in plan at the 0.50 m long-axis cap)
S2_X = .65            # m: S2 lies this far past the head edge, where a lump lies flat on B again ...
S2_Z = .015           # ... this high above B: low enough that two touching lumps mostly leave a gap under their
                      # junction. 0.10 m (half the 5 cm step) until 2026-09-30: a lump's rear crossed it still in the
                      # air (seed 392). 0.35 m then; with the 10 cm step a lump touches the beam, bounces clear of it
                      # for 0.22-0.34 s and comes down again there -- two interruptions for one lump (6 of 160 in the
                      # buffer sweep, 2026-10-01) -- so 0.65 m
S2_ROOM = .30         # m: S2 lies at least this far before S3
HANDOVER_MIN = .05    # m: an S2 interruption over less belt travel than this is a touch or a bounce, not a lump
                      # (the smallest lump blocks the low beam over about 0.1 m; seed 4001: a 2.8 cm blip)
HANDOVER_JOIN = .08   # m: S2 clear for less than this much buffer belt travel between two blocks = one
                      # interruption: a lump still rocking from the drop lifts off the low beam (seed 392: 6.4 cm)
HANDOVER_JOIN_S = .20  # s: ... or for less than this long: a lift-off lasts about 0.16 s whatever B's speed, so at
                      # 0.6-0.8 m/s it spans 10-13 cm of travel (oblique 6001, 2026-09-30)
PATH_TIMEOUT_S = 6.   # s: a sent item must be out of the plate's path this long after its discharge started
FREEZE_MAX_S = 5.     # s: the vision not healthy this long (not found again): stop the run
DEAD_AFTER = 2       # hand-overs (S2) or releases (S1) in a row without the beam: the beam is dead
ACROSS_IN = .10      # m: the cameras see a lump's body across a beam when the line is this far inside its outline

STAGE_ZH = dict(buffer='缓冲', transfer='上计量带', onm='在计量带上', measure='称重+测体积', decided='等排料板',
                discharge='排出', hold='作废停住')
PLATE_ZH = dict(closed='落板（煤）', opening='抬起中', open='抬起（矸）', closing='回落中')
REASON_ZH = dict(multi='多块', handover='交接未确认', outside='搭秤外', unsteady='读数不稳', scan='扫描无效',
                 implausible='密度不合理', tare='秤未清空', track_lost='跟踪丢失')


def geometry(cfg, x1, y_c):
    """B from the main belt head edge x1, M after it, the separator after that, all centred on the lane
    centre line y_c."""
    for k in DEFAULTS:
        if not math.isfinite(cfg[k]) or cfg[k] < 0 or (cfg[k] == 0 and k != 'separator_wall'):
            raise ValueError('--%s must be finite and positive' % k.replace('_', '-'))
    if cfg['station_w'] <= cfg['lane_w']:
        raise ValueError('--station-w %.3f m must exceed the lane width %.3f m: the lane hands over into the buffer'
                         % (cfg['station_w'], cfg['lane_w']))
    span = math.hypot(cfg['size_long_max'], cfg['size_max'])       # plan diagonal of a flat-lying lump, bound
    stop = cfg['station_speed'] * (RAMP_S / 2 + OBS_DT)
    stop_back = max(STOP_BACK, cfg['buffer_speed'] * (RAMP_S / 2 + OBS_DT + LATENCY_S) + STAGE_ROOM)
    need = dict(buffer_len=max(CLEAR + span, S2_X + S2_ROOM) + stop_back, measure_len=CLEAR + stop + span + FRONT_MARGIN)
    for k, v in need.items():
        if cfg[k] < v:
            raise ValueError('--%s %.3f m cannot hold one lump (%.3f m plan extent) wholly: needs %.3f m'
                             % (k.replace('_', '-'), cfg[k], span, v))
    top = -cfg['station_step']
    b0, b1 = x1, x1 + cfg['buffer_len']
    m0, m1 = b1, b1 + cfg['measure_len']
    sep = separator.for_width(cfg['station_w'], side_wall_height=cfg['separator_wall'])
    origin = np.array([m1 + GAP, y_c, top - DROP - sep.rib_height - sep.inlet_height])
    world = lambda P: [round(float(a + b), 4) for a, b in zip(origin, P)]
    pivot = world(separator.layout(sep)[0])
    closed, opened = separator.state(sep, sep.closed_deg), separator.state(sep, sep.open_deg)
    mu = cfg['separator_friction']
    inlet = world(closed['inlet'])
    gz = inlet[2] - GANGUE_DZ
    out_c, in_o = world(closed['outlet']), world(opened['inlet'])
    # the plate's path: where a lump sent over the plate may still be while the plate must not move
    zone = ((m1, y_c - cfg['station_w'] / 2 - .10, min(gz, out_c[2]) - .05),
            (out_c[0] + .10, y_c + cfg['station_w'] / 2 + .10, inlet[2] + .50))
    return dict(sep=sep, width_m=cfg['station_w'], y_c=y_c, top_z=top, step_m=cfg['station_step'],
                speed_m_s=cfg['station_speed'], buffer_speed_m_s=cfg['buffer_speed'], ramp_s=RAMP_S,
                buffer=dict(x0=b0, x1=b1, length_m=cfg['buffer_len'], speed_m_s=cfg['buffer_speed']),
                measure=dict(x0=m0, x1=m1, length_m=cfg['measure_len'], scan_s=cfg['scan_s'],
                             filter_s=FILTER_S, steady_s=STEADY_S, steady_band=STEADY_BAND,
                             max_wait_s=WEIGH_MAX_S, skirts_on_weigh_frame=True,
                             volume='true hull volume (ideal value); usability judged by the scanner model'),
                beam_in=dict(x=b0 + S2_X, z=top + S2_Z), beam_stop=dict(x=b1 - stop_back, z=top + STOP_Z),
                beam_gangue=dict(x=m1 + GANGUE_X, z=gz),
                plate_zone=[[round(float(v), 4) for v in p] for p in zone],
                longest_plan_extent_m=round(span, 4), stop_distance_m=round(stop, 4),
                separator=dict(width_m=sep.width, rib_count=sep.rib_count, inlet_height_m=sep.inlet_height,
                               origin=[round(float(v), 4) for v in origin], floor_z=round(float(origin[2]), 4),
                               inlet=inlet, pivot=pivot, outlet_closed=out_c,
                               inlet_open=in_o, outlet_open=world(opened['outlet']),
                               closed_deg=sep.closed_deg, open_deg=sep.open_deg, swing_s=cfg['separator_swing_s'],
                               deck_friction=mu,
                               coal_slides_when_closed=bool(math.tan(math.radians(sep.closed_deg)) > mu)),
                sort_density_kg_m3=cfg['sort_density'],
                void_rules=dict(density_ok_kg_m3=list(DENSITY_OK), iso_margin_m=ISO_MARGIN, iso_gap_m=ISO_GAP,
                                tare_tol_N=TARE_TOL, handover_len_m=HANDOVER_LEN, weigh_max_s=WEIGH_MAX_S,
                                path_timeout_s=PATH_TIMEOUT_S,
                                on_void='hold: no discharge, no plate move, no release; the simulation then takes '
                                        'the lumps off the line (manual re-measuring is not modelled)'),
                # a lump whose lowest point is below this has left the machine (the closed plate's outlet - 0.2 m)
                drop_z=round(float(origin[2] + closed['outlet'][2] - .2), 4),
                end_x=out_c[0])


def report(st):
    """The JSON-safe part of geometry()."""
    return {k: v for k, v in st.items() if k != 'sep'}


def landing(st, com):
    """Where a lump that fell below drop_z went, by its centroid: through the gap in front of the pivot
    ('sorted', 'gangue'), over the plate's far end ('sorted', 'coal'), else off a side or upstream
    ('dropped', None)."""
    if abs(com[1] - st['y_c']) > st['width_m'] / 2 + .10 or com[0] < st['measure']['x1'] - .05:
        return 'dropped', None
    return 'sorted', ('gangue' if com[0] < st['separator']['pivot'][0] else 'coal')


def weigher_reading(model, data, weigher, lump_of, load_only=False):
    """Load-cell reading of M: the downward contact force of everything touching M and its skirts, N.
    Also, for the verification only, which lumps load M and what else every lump touches (other lumps by
    id, 'equipment' off the weigh frame). load_only: the force alone (every physics step while weighing)."""
    Fz, on, other, w = 0., set(), {}, np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        k1, k2 = lump_of.get(g1), lump_of.get(g2)
        if g1 in weigher or g2 in weigher:
            k = k2 if g1 in weigher else k1
            if k is None:
                continue
            mujoco.mj_contactForce(model, data, i, w)
            f = con.frame.reshape(3, 3).T @ w[:3]            # force of geom1 on geom2, world frame
            Fz += f[2] if g1 in weigher else -f[2]
            on.add(k)
            continue
        if load_only:
            continue
        for k, ok in ((k1, k2), (k2, k1)):
            if k is not None:
                other.setdefault(k, set()).add(ok if ok is not None else 'equipment')
    return Fz if load_only else (Fz, on, other)


def _box(o):
    return o['x0'], o['x1'], o['y0'], o['y1']


def gap2(a, b):
    """Plan gap between two objects' bounding boxes (0 when they overlap)."""
    dx = max(0., max(a['x0'], b['x0']) - min(a['x1'], b['x1']))
    dy = max(0., max(a['y0'], b['y0']) - min(a['y1'], b['y1']))
    return math.hypot(dx, dy)


class Station:
    """B, M and the plate. The drives are Conveyors motors and the plate cylinder a MuJoCo position servo;
    this sets their targets from what the sensors report."""

    def __init__(self, cfg, d, blocks):
        self.cfg, self.d, self.g, self.blocks, self.n = cfg, d, d['station'], blocks, len(blocks)
        self.vision = cfg.get('sensing', 'vision') == 'vision'
        self.vm = V_MARGIN if self.vision else 0.
        c = self.sep = self.g['sep']
        deg = np.linspace(c.closed_deg, c.open_deg, 221)
        self.ext = (deg, np.array([separator.state(c, a)['piston_extension_m'] for a in deg]))
        self.rate = (c.open_deg - c.closed_deg) / cfg['separator_swing_s']      # plate, deg/s
        self.v, self.vb = cfg['station_speed'], cfg['buffer_speed']     # M, B
        self.centre = cfg.get('weigh_stop', 'rear') == 'centre'
        self.slow = cfg.get('buffer_approach', 'full') == 'slow'
        # drive factors (1 = rated speed) and their goals; u is the whole section upstream of B
        self.u, self.b, self.m = 1., 0., 0.
        self.u_goal, self.b_goal, self.m_goal = 1., 1., 1.
        self.plate = 'closed'
        self.plate_ref = self.plate_goal = self.plate_deg = c.closed_deg
        self.plate_t0 = 0.
        self.items, self.line, self.sent = [], [], []
        self.jog, self.fault, self.alarm = None, None, None
        self.hist = {}                              # (t, centroid x) per object, HANG_S long
        self.samples, self.cell, self.tare, self.zero_off = [], 0., 0., 0.
        self.load_sum, self.load_n = 0., 0          # per-step load while weighing, summed over one sample
        self.counts = dict(holds=0, section_holds=0, late=0, jogs=0, buffer_stops=0, plate_moves=0, discharge_early=0,
                           s3_by_camera=0, freezes=0, splits=0)
        self.events, self.held_s, self.buffer_stopped_s, self.frozen_s = [], 0., 0., 0.
        self.key, self.since = None, 0.
        self.frozen, self.frozen_since = False, None
        self.s2, self.b_odo, self.s2_missed = [], 0., 0   # S2 interruptions; buffer belt odometer; misses in a row
        self.objs = {}
        self.taken = set()                          # verification: true lumps seen past the head edge

    # ---- model hookup ---------------------------------------------------------------------------
    def bind(self, model):
        P = separator.PREFIX
        self.q_plate = model.jnt_qposadr[model.joint(P + 'plate_hinge').id]
        self.act = model.actuator(P + 'cylinder_position').id
        mb = model.body('mbelt').id
        self.weigher = {g for g in range(model.ngeom) if model.geom_bodyid[g] == mb
                        or (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('mskirt')}

    def place(self, model, data):
        """Plate down (coal) at t = 0."""
        separator.place(model, data, self.sep, self.sep.closed_deg)

    def angle(self, data):
        return math.degrees(float(data.qpos[self.q_plate]))

    def _stage(self, *stages):
        return [it for it in self.line if it['stage'] in stages]

    @property
    def weighing(self):
        return bool(self._stage('measure', 'hold'))

    def load_step(self, Fz):
        """One physics step's load on M while weighing (averaged into the next sample)."""
        self.load_sum += Fz
        self.load_n += 1

    @property
    def section_held(self):
        """The section upstream of B is stopped (or jogging) for the station."""
        return self.u_goal < 1.

    @property
    def inhibit_release(self):
        """The feed belt must not let another lump go: a void item is held, or an alarm stops the line."""
        return bool(self._stage('hold')) or self.frozen

    def waiting(self, k, state, cx):
        """Planned wait (stall bookkeeping only, verification side): a lump the station has taken, or any
        lump upstream of the head edge while the station holds the section."""
        return state == 'on_belt' and (k in self.taken or (self.section_held or self.u < 1.)
                                       and cx <= self.g['buffer']['x0'])

    # ---- per physics step ------------------------------------------------------------------------
    def step(self, data, dt):
        """Ramp the drive factors toward their goals, move the plate reference; returns (u, b, m)."""
        r = dt / RAMP_S
        ramp = lambda x, goal: min(goal, x + r) if goal > x else max(goal, x - r)
        self.u, self.b, self.m = ramp(self.u, self.u_goal), ramp(self.b, self.b_goal), ramp(self.m, self.m_goal)
        self.plate_ref += float(np.clip(self.plate_goal - self.plate_ref, -self.rate * dt, self.rate * dt))
        data.ctrl[self.act] = float(np.interp(self.plate_ref, *self.ext))
        return self.u, self.b, self.m

    # ---- every camera sample ---------------------------------------------------------------------
    def observe(self, t, view, beams, plate_deg, scanner, audit=None):
        """view: perception frame {tid: object} (health in view.health); beams: {name: perception.Beam};
        plate_deg: the plate encoder; scanner(t) -> the scanner's verdict on the measuring zone now;
        audit: the true state, for the verification only (dict: reading=(Fz, on, other) of the weigh frame,
        com={lump: centroid}, states, bins). The load cell's own reading is audit['reading'][0] (a sensor)."""
        g = self.g
        self.plate_deg, self.audit = plate_deg, audit or {}
        # the load cell: the mean of every physics step since the last sample while weighing, else this instant
        self.cell = self.load_sum / self.load_n if self.load_n else self.audit.get('reading', (0.,))[0]
        self.load_sum, self.load_n = 0., 0
        self.held_s += OBS_DT * self.section_held
        self.buffer_stopped_s += OBS_DT * (self.b_goal == 0.)
        self.frozen_s += OBS_DT * self.frozen
        x_e, x_j, m_end = g['buffer']['x0'], g['buffer']['x1'], g['measure']['x1']
        self.objs = objs = {tid: o for tid, o in view.items() if o['cx'] > x_e and o['x0'] < g['plate_zone'][1][0]}
        for o in objs.values():
            self.taken.update(o['_truth'])                # bookkeeping of planned waits only
        for tid, o in view.items():
            h = self.hist.setdefault(tid, [])
            h.append((t, o['cx']))
            while len(h) > 2 and h[1][0] < t - HANG_S - .02:
                h.pop(0)
        self._s2_log(t, beams['beam_in'])

        # ---- objects -> items --------------------------------------------------------------------
        self._assign(t, objs)
        for it in self.line + [e for e in self.sent if 't_clear_s' not in e]:
            it['objs'] = [objs[tid] for tid in it['now']]
            if it['objs']:
                it['box'] = (min(o['x0'] for o in it['objs']), max(o['x1'] for o in it['objs']),
                             min(o['y0'] for o in it['objs']), max(o['y1'] for o in it['objs']))
                it['n_obj_max'] = max(it['n_obj_max'], len(it['objs']))
                it['n_est_max'] = max(it['n_est_max'], max(o['n_est'] for o in it['objs']))
        self._split(t)
        self._handover(t, beams['beam_in'])

        # ---- stage changes on B and M ------------------------------------------------------------
        on_m = lambda it: it['stage'] in ('transfer', 'onm', 'measure', 'decided', 'discharge', 'hold')
        for i, it in enumerate(self.line):
            if 'box' not in it:
                continue
            x0, x1 = it['box'][:2]
            if not it['objs'] and it['stage'] == 'discharge':
                x0 = m_end + 1.                                   # gone over the head edge and out of view
            if it['stage'] == 'buffer' and x1 > x_j:
                it.update(stage='transfer', t_transfer_s=round(t, 3))
            if it['stage'] == 'transfer' and x0 > x_j + CLEAR + self.vm:
                it['stage'] = 'onm'
            first = not any(on_m(e) for e in self.line[:i])
            at_front = x1 >= m_end - FRONT_MARGIN - self.v * (RAMP_S / 2 + OBS_DT)
            # --weigh-stop centre: an item wholly on M, M free for it, rides on to the middle of the weigh zone
            riding = self.centre and it['stage'] == 'onm' and first
            centred = (x0 + x1) / 2 >= (x_j + m_end) / 2 - self.v * (RAMP_S / 2 + OBS_DT)
            if it['stage'] in ('transfer', 'onm') and self.m_goal == 1. and at_front and not riding:
                # M is carrying it and its front is at the stop position before the item is wholly on (it is longer
                # than the belt, or its lumps have drawn apart; seed 4014): stop now -- nothing leaves M unmeasured --
                # and hold it
                self._void(it, 'outside', detail='front at the measuring belt head before the item was on it')
                it.update(stage='measure', t_stop_s=round(t, 3), tare_N=round(self.tare, 2))
                self.samples = []
            if it['stage'] == 'onm' and first and (not self.centre or centred or at_front):
                it.update(stage='measure', t_stop_s=round(t, 3), tare_N=round(self.tare, 2))
                self.samples = []
            if it['stage'] == 'discharge' and x0 > m_end + CLEAR:
                it.update(stage='sent', t_off_s=round(t, 3))
        for it in [e for e in self.line if e['stage'] == 'sent']:
            self.line.remove(it)
            self.sent.append(it)

        # ---- the plate's path, and what has landed ------------------------------------------------
        path_clear = self._path(t, view, beams['beam_gangue'])
        # ---- M: tare while empty, weigh and scan at rest -----------------------------------------------
        if not self.frozen and not any(on_m(e) for e in self.line) \
                and not any(o['x1'] > x_j - ZERO_CLEAR and o['x0'] < m_end + ZERO_CLEAR for o in objs.values()):
            self._zero()
        for it in self._stage('measure'):
            self._measure(t, it, scanner)
        # ---- plate: move the moment the route is known and the path is clear ---------------------------
        self._plate(t, path_clear)
        # ---- M: discharge as soon as the plate will be in position in time -----------------------------
        for it in self._stage('decided'):
            if self.frozen:
                break
            want = 'open' if it['route'] == 'gangue' else 'closed'
            t_front = max(0., m_end - it['box'][1]) / self.v + RAMP_S / 2
            moving = self.plate == ('opening' if want == 'open' else 'closing')
            remaining = abs(self.plate_goal - self.plate_deg) / self.rate
            if self.plate == want or (moving and remaining + PLATE_MARGIN_S <= t_front):
                it.update(stage='discharge', t_discharge_s=round(t, 3), plate_at_start=self.plate)
                self.counts['discharge_early'] += self.plate != want
            break                                             # one at a time, in order

        # ---- alarms, belt goals ------------------------------------------------------------------------
        self._alarms(t, view, beams)
        self.m_goal = 0. if (self._stage('measure', 'decided', 'hold') or self.frozen) else 1.
        busy = [e for e in self.line if on_m(e)]
        accepting = all(e['stage'] == 'discharge' and e['box'][0] > x_j + ACCEPT_AFTER for e in busy)
        lead = next((e for e in self.line if e['stage'] == 'buffer'), None)
        staged = False
        if lead is not None and 'box' in lead:
            by_cam = lead['box'][1] > g['beam_stop']['x'] + STAGE_CAM
            staged = beams['beam_stop'].blocked or by_cam
            if by_cam and not beams['beam_stop'].blocked and not lead.get('s3_by_camera'):
                lead['s3_by_camera'] = round(t, 3)
                self.counts['s3_by_camera'] += 1
                self._event(t, 's3_by_camera', item=lead['item'])
        if self.frozen:
            b_goal = 0.
        elif self._stage('transfer'):
            b_goal = self.m_goal * self.v / self.vb           # an item crossing moves only with M, at its speed
        else:
            b_goal = 0. if staged and not accepting else 1.
            if b_goal and self.slow and lead is not None and 'box' in lead and lead['box'][1] > x_j - APPROACH:
                b_goal = self.v / self.vb                     # --buffer-approach slow: the last stretch at M's speed
        if b_goal == 0. and self.b_goal > 0.:
            self.counts['buffer_stops'] += 1
            self._event(t, 'buffer_stop', item=None if lead is None else lead['item'])
        self.b_goal = b_goal
        self._section(t, view)
        self._watch(t)

    # ---- items -------------------------------------------------------------------------------------
    def _new_item(self, t):
        it = dict(item=len(self.items), tids=set(), now=[], objs=[], joined=[], t_in_s=round(t, 3), stage='buffer',
                  reasons=[], n_obj_max=0, n_est_max=0, handover=dict(done=False))
        self.items.append(it)
        self.line.append(it)
        return it

    def _void(self, it, why, **info):
        if why not in it['reasons']:
            it['reasons'].append(why)
            if info:
                it.setdefault('reason_info', {})[why] = info

    def _assign(self, t, objs):
        """Every object past the head edge to an item: by track lineage (items still on the line, or sent
        and not yet out of the plate's path), else as a new arrival on B."""
        g = self.g
        live = self.line + [e for e in self.sent if 't_clear_s' not in e]
        for it in live:
            it['now'] = []
        new = []
        for tid in sorted(objs, key=lambda k: -objs[k]['x1']):
            o = objs[tid]
            its = [it for it in live if it['tids'] & o['lineage']]
            line_its = [it for it in its if it in self.line]
            if len(line_its) > 1:                             # two items ran together: one item now
                keep = line_its[0]
                for it in line_its[1:]:
                    keep['tids'] |= it['tids']
                    keep['joined'].append('item %d' % it['item'])
                    self.line.remove(it)
                    live.remove(it)
                    it['stage'] = 'merged_into_%d' % keep['item']
                    self._void(keep, 'multi', detail='items ran together', absorbed=it['item'])
                    self._event(t, 'items_merged', item=keep['item'], absorbed=it['item'])
                its = [keep]
            if its:
                its[0]['tids'].add(tid)
                its[0]['now'].append(tid)
            elif o['cx'] < g['measure']['x1']:
                new.append(tid)                               # past M's head edge unclaimed: nothing to do
        for tid in new:
            o = objs[tid]
            orphan = [it for it in self.line if not it['now'] and 'box' in it
                      and it['box'][0] - .3 < o['cx'] < it['box'][1] + .3]
            last = self.line[-1] if self.line else None
            if orphan:                                        # found again after the cameras lost it
                it = orphan[0]
                self._void(it, 'track_lost')
            elif last is not None and last['stage'] == 'buffer' and 'box' in last \
                    and o['x1'] >= last['box'][0] - GROUP_GAP and o['cx'] < g['buffer']['x1']:
                it = last
                it['joined'].append(tid)
            else:
                it = self._new_item(t)
                if o['cx'] > g['buffer']['x0'] + .5:
                    self._void(it, 'track_lost', detail='appeared on the station, not over the head edge')
                if self.section_held:
                    self.counts['late'] += 1                  # went over while the section was held
                self._event(t, 'went', item=it['item'], late=self.section_held)
            it['tids'].add(tid)
            it['now'].append(tid)

    def _split(self, t):
        """An item on B whose objects have drawn apart by GROUP_GAP (in x) becomes two: the lead objects keep
        the item, the ones behind become a new item right after it. Each keeps only its own objects' track
        ids (a blob's id stays with neither, or its halves would pull the two back together) and its object
        counts restart from what the cameras see now."""
        for it in [e for e in self.line if e['stage'] == 'buffer' and len(e['objs']) > 1]:
            objs = sorted(it['objs'], key=lambda o: o['x0'])
            reach, cut = objs[0]['x1'], None
            for i, o in enumerate(objs[1:], 1):
                if o['x0'] - reach >= GROUP_GAP:
                    cut = i                                       # the last clear gap, from behind
                reach = max(reach, o['x1'])
            if cut is None:
                continue
            behind, ahead = objs[:cut], objs[cut:]
            new = dict(item=len(self.items), tids={o['tid'] for o in behind}, now=[o['tid'] for o in behind],
                       objs=behind, joined=[], t_in_s=it['t_in_s'], stage='buffer', reasons=[],
                       n_obj_max=len(behind), n_est_max=max(o['n_est'] for o in behind), handover=dict(done=False),
                       split_from=it['item'])
            self.items.append(new)
            self.line.insert(self.line.index(it) + 1, new)
            it.update(tids={o['tid'] for o in ahead}, now=[o['tid'] for o in ahead], objs=ahead,
                      n_obj_max=len(ahead), n_est_max=max(o['n_est'] for o in ahead))
            it.setdefault('split', []).append(new['item'])
            if it['handover']['done'] and it['reasons'] == ['handover']:
                it['reasons'], it['handover'] = [], dict(done=False)   # judged together: judge again, alone
                it.pop('reason_info', None)
            for e in (it, new):
                e['box'] = (min(o['x0'] for o in e['objs']), max(o['x1'] for o in e['objs']),
                            min(o['y0'] for o in e['objs']), max(o['y1'] for o in e['objs']))
            self.counts['splits'] += 1
            self._event(t, 'item_split', item=it['item'], new=new['item'])

    def _s2_log(self, t, s2):
        """S2 interruptions on the buffer belt odometer, with the belt travel while blocked; a clear over less
        than HANDOVER_JOIN of travel or HANDOVER_JOIN_S between two blocks is one interruption (a lump lifting
        off the low beam after the drop, a rocking lump, or B stopped)."""
        step = self.b * self.vb * OBS_DT
        self.b_odo += step
        log = self.s2
        if s2.blocked:
            if not log or (log[-1]['t1'] is not None and self.b_odo - log[-1]['odo1'] > HANDOVER_JOIN
                           and t - log[-1]['t1'] > HANDOVER_JOIN_S):
                log.append(dict(t0=round(t, 3), t1=None, odo0=self.b_odo, odo1=None, blocked=0.))
            elif log[-1]['t1'] is not None:
                log[-1]['t1'] = log[-1]['odo1'] = None        # a lift-off or a bounce: the same interruption goes on
            log[-1]['blocked'] += step
        elif log and log[-1]['t1'] is None:
            log[-1]['t1'], log[-1]['odo1'] = round(t, 3), self.b_odo

    def _handover(self, t, s2):
        """S2 confirms each hand-over onto B once the item's rear is past the beam: the interruptions over the
        stretch of B it passed the beam on, as many as its objects, none blocked longer than one lump. A lump
        rides B from the beam on, so it passed the beam where the odometer read b_odo minus its distance past
        the beam (the camera's position is LATENCY_S old: B moved on meanwhile)."""
        bx = self.g['beam_in']['x']
        lag = self.b * self.vb * LATENCY_S
        pad = self.vm + lag + .02
        for it in self.line:
            h = it['handover']
            if h['done'] or 'box' not in it or it['box'][0] <= bx + self.vm + .01:
                continue
            lo = self.b_odo - (it['box'][1] + lag - bx) - pad          # its front at the beam
            hi = self.b_odo - (it['box'][0] + lag - bx) + pad          # its rear at the beam
            if self.s2 and self.s2[-1]['t1'] is None and self.s2[-1]['odo0'] <= hi:
                continue                                               # still blocked over its stretch
            seen = [s for s in self.s2 if s['t1'] is not None and s['odo1'] >= lo and s['odo0'] <= hi]
            ints = [s for s in seen if s['blocked'] >= HANDOVER_MIN]
            travel = max((s['blocked'] for s in ints), default=0.)
            h.update(done=round(t, 3), interruptions=len(ints), blips=len(seen) - len(ints),
                     longest_travel_m=round(travel, 3))
            self.s2_missed = self.s2_missed + 1 if not ints else 0
            if self.s2_missed >= DEAD_AFTER and s2.alarm is None:
                s2.alarm = dict(t_s=round(t, 3), beam=s2.name, why='dead', stop=True,
                                note='%d hand-overs in a row the cameras saw and S2 did not' % self.s2_missed)
            if not ints:
                self._void(it, 'handover', detail='S2 saw nothing')
            elif len(ints) > max(1, it['n_obj_max']):
                self._void(it, 'handover', detail='S2 saw %d interruptions, the cameras %d object(s)'
                           % (len(ints), it['n_obj_max']))
            elif travel > HANDOVER_LEN:
                self._void(it, 'handover', detail='one interruption over %.2f m of belt' % travel)

    # ---- the plate's path ------------------------------------------------------------------------
    def _in_zone(self, view):
        (x0, y0, z0), (x1, y1, _) = self.g['plate_zone']
        return [o for o in view.values() if o['x1'] > x0 and o['x0'] < x1 and o['y1'] > y0 and o['y0'] < y1
                and o['ztop'] > z0]

    def _path(self, t, view, s4):
        """Everything sent has left the plate's path: gangue -- S4 saw it fall and is clear again, and the
        cameras see nothing of it or of anything unknown in the plate zone; coal -- the zone empty the same
        way. Objects of items sent later do not count against an earlier one. Not out within PATH_TIMEOUT_S
        of its discharge: stop the line."""
        zone = self._in_zone(view)
        order = {it['item']: i for i, it in enumerate(self.sent)}
        owner = lambda o: next((it for it in self.sent if it['tids'] & o['lineage']), None)
        for it in self.sent:
            if 't_clear_s' in it:
                continue
            mine = [o for o in zone if owner(o) is None or order[owner(o)['item']] <= order[it['item']]]
            occupied = bool(mine) or not view.health['ok']
            fell = s4.blocks_since(it['t_discharge_s']) > 0 and not s4.blocked
            if not occupied and (fell or it['route'] == 'coal'):
                it['t_clear_s'] = round(t, 3)
                it['path_confirmed_by'] = 'S4 + plate-zone camera' if it['route'] == 'gangue' else 'plate-zone camera'
            elif t - it['t_discharge_s'] > PATH_TIMEOUT_S and self.fault is None:
                self.fault = dict(t_s=round(t, 2), reason='separator_path_timeout', item=it['item'],
                                  route=it['route'], s4_interruptions=s4.blocks_since(it['t_discharge_s']),
                                  s4_blocked=s4.blocked, zone_occupied=occupied,
                                  note='the plate path was not confirmed clear: S4 alone does not prove it')
        audit = self.audit
        for it in self.sent:
            lumps = it.get('truth_lumps', [])
            if 't_landed_s' not in it and lumps and 'states' in audit and all(audit['states'][k] in ('sorted', 'dropped')
                                                        for k in lumps):
                bins = {str(k): audit['bins'][k] for k in lumps}
                it.update(t_landed_s=round(t, 3), bins=bins,
                          bins_match_route=all(b == it['route'] for b in bins.values()))
                self._event(t, 'landed', item=it['item'], bins=bins)
        return all('t_clear_s' in it for it in self.sent)

    # ---- weighing and scanning -------------------------------------------------------------------
    def _zero(self):
        """M seen empty: zero tracking. The zero follows the reading only within ZERO_BAND of it; a reading
        off the zero for ZERO_OFF_S means something lies on the scale (tare fault: the next item is void)."""
        if abs(self.cell - self.tare) <= ZERO_BAND:
            self.tare += (self.cell - self.tare) / ZERO_N
            self.zero_off = 0.
        else:
            self.zero_off += OBS_DT
            if self.zero_off >= ZERO_OFF_S - 1e-9:
                self.tare, self.zero_off = self.cell, 0.       # re-zero there: the item will be void (tare)

    def _measure(self, t, it, scanner):
        """Weigh (steady indicator) and scan (scan_s) at rest, together; the void checks all along."""
        rest = it['t_stop_s'] + RAMP_S                    # M at rest from here
        if t < rest - 1e-9:
            return
        g, box = self.g, it.get('box')
        x_j, m_end = g['buffer']['x1'], g['measure']['x1']
        # isolation, every sample at rest: inside the weigh zone, nothing else close, one object of one lump
        if box is not None and (box[0] < x_j + ISO_MARGIN or box[1] > m_end - ISO_MARGIN):
            self._void(it, 'outside', detail='outline %.3f..%.3f m, weigh zone %.3f..%.3f m'
                       % (box[0], box[1], x_j + ISO_MARGIN, m_end - ISO_MARGIN))
        mine = set(it['now'])
        if it['objs'] and any(min(gap2(o, m) for m in it['objs']) < ISO_GAP
                              for tid, o in self.objs.items() if tid not in mine):
            self._void(it, 'outside', detail='another object within %.2f m' % ISO_GAP)
        if it['n_obj_max'] > 1 or it['n_est_max'] > 1:
            self._void(it, 'multi', detail='objects %d, estimated lumps %d' % (it['n_obj_max'], it['n_est_max']))
        if abs(it['tare_N']) > TARE_TOL:
            self._void(it, 'tare', detail='empty reading %.1f N' % it['tare_N'])
        # the truth, for the verification: what physically loads the weigh frame, and what else it touches
        if 'reading' in self.audit:
            _, on, other = self.audit['reading']
            closure, stack = set(on), list(on)
            while stack:                                  # lumps leaning on the lumps that load the frame
                for x in other.get(stack.pop(), ()):
                    if not isinstance(x, str) and x not in closure:
                        closure.add(x)
                        stack.append(x)
            iso = closure <= set(on) and all('equipment' not in other.get(k, ()) for k in closure)
            it['_on'] = it.get('_on', set()) | closure
            it['truth_isolated'] = it.get('truth_isolated', True) and iso
        if 'mass_kg' not in it:
            self.samples.append(self.cell - it['tare_N'])
            nf, ns = int(round(FILTER_S / OBS_DT)), int(round(STEADY_S / OBS_DT))
            F = np.array(self.samples)
            steady = False
            if len(F) >= nf + ns - 1:
                ma = np.convolve(F, np.ones(nf) / nf, 'valid')[-ns:]
                steady = ma.max() - ma.min() <= max(STEADY_MIN_N, STEADY_BAND * abs(ma.mean()))
            if steady or t - rest >= WEIGH_MAX_S - 1e-9:
                win = F[-ns:]
                it.update(t_weigh_s=round(t, 3), weigh_wait_s=round(t - rest, 3), steady=bool(steady),
                          mass_kg=round(float(win.mean()) / G, 3), reading_std_N=round(float(win.std()), 2))
                if not steady:
                    self._void(it, 'unsteady')
        if 'scan' not in it and t >= rest + self.cfg['scan_s'] - 1e-9:
            res = scanner(t)
            it['scan'] = {k: v for k, v in res.items() if not k.startswith('_')}
            if not res['valid']:
                self._void(it, 'scan', detail=','.join(res['reasons']))
            else:
                it['volume_m3'] = res['volume_m3']
        if 'mass_kg' in it and 'scan' in it:
            self._decide(t, it)

    def _decide(self, t, it):
        thr = self.cfg['sort_density']
        vol = it.get('volume_m3')
        rho = it['mass_kg'] / vol if vol else float('nan')
        lumps = sorted(it.pop('_on', set()))              # truth: the lumps on (or leaning on) the weigh frame
        true = sum(self.blocks[k]['mass_kg'] for k in lumps)
        it.update(t_decided_s=round(t, 3), density_kg_m3=round(rho, 1) if math.isfinite(rho) else None,
                  truth_lumps=lumps, truth_single=len(lumps) == 1, truth_mass_kg=round(true, 3),
                  mass_error_pct=round(100. * (it['mass_kg'] - true) / true, 3) if true else None,
                  truth_density_kg_m3=[round(self.blocks[k]['density'], 1) for k in lumps],
                  material=[self.blocks[k]['material'] for k in lumps],
                  ideal_route=['gangue' if self.blocks[k]['density'] >= thr else 'coal' for k in lumps])
        if math.isfinite(rho) and not DENSITY_OK[0] <= rho <= DENSITY_OK[1]:
            self._void(it, 'implausible', detail='%.0f kg/m3' % rho)
        it['void'] = bool(it['reasons'])
        if it['void']:
            self._hold(t, it)
            return
        it.update(stage='decided', route='gangue' if rho >= thr else 'coal')
        it['route_as_ideal'] = all(r == it['route'] for r in it['ideal_route'])
        self._event(t, 'decided', item=it['item'], mass_kg=it['mass_kg'], volume_m3=it['volume_m3'],
                    density_kg_m3=it['density_kg_m3'], route=it['route'])

    def _hold(self, t, it):
        it.update(stage='hold', t_hold_s=round(t, 3))
        self.counts['holds'] += 1
        self._event(t, 'hold', item=it['item'], reasons=list(it['reasons']))

    # ---- the simulation takes a held item off the line (line.TakeOff) -----------------------------------
    def took_off(self, t, lumps):
        """The held item's lumps (and anything else on M, straddling the joint included) are off the line: the
        stop is over. An item crossing onto M whose every lump went with them goes too: left in place it stood
        on M for ever and M never stopped for the next item, which rode to M's head and was held 'outside'
        (lane 0.58 m trial, aligned 6006, 2026-10-01)."""
        gone = set(lumps)
        for it in list(self.line):
            truth = {k for o in it['objs'] for k in o['_truth']}
            if it['stage'] == 'hold' or (it['stage'] in ('transfer', 'onm') and truth and truth <= gone):
                self.line.remove(it)
                it.update(stage='taken_off', t_taken_off_s=round(t, 3), taken_off=list(lumps))
        self.tare, self.zero_off = 0., 0.                     # M cleaned: the scale is clear again
        self._event(t, 'taken_off', lumps=list(lumps))

    # ---- alarms ------------------------------------------------------------------------------------
    def _alarms(self, t, view, beams):
        """Beam diagnoses stop the run; the vision not healthy freezes the line until it is again (at most
        FREEZE_MAX_S, then the run stops)."""
        d, g = self.d, self.g
        fd = d['feeder']
        belt = dict(beam_feed=self.u, beam_in=self.b, beam_stop=self.b, beam_gangue=1.)
        x_on = dict(beam_feed=fd['x1'], beam_in=g['buffer']['x0'], beam_stop=g['buffer']['x0'],
                    beam_gangue=g['measure']['x1'])
        for name, b in beams.items():
            near = [o for o in view.values() if o['x0'] - .05 < b.x < o['x1'] + .05]
            # a lump's body across the line (not a raised nose or tail: ACROSS_IN inside its outline both ways),
            # on the belt under the beam, seen with confidence
            across = [o for o in near if o['x0'] + ACROSS_IN < b.x < o['x1'] - ACROSS_IN and o['cx'] > x_on[name]
                      and o['conf'] >= CONF_OK]
            seen_clear = view.health['ok'] and not near
            expected = name == 'beam_stop' or belt[name] < .5    # staged, or the belt under it stopped
            # 'dead' by the cameras only for S3: at S1 and S2 a lump still overhanging the edge above the beam
            # looks the same from above as one that went over; those two are judged dead by repeated misses
            # (the feed belt's beam-miss stops, S2 hand-overs it did not see)
            a = b.diagnose(t, seen_clear, bool(across) and name == 'beam_stop', expected)
            if a is not None and self.alarm is None:
                self.alarm = dict(a)
                self._event(t, 'beam_alarm', **{k: v for k, v in a.items() if k != 't_s'})
                if self.fault is None:
                    self.fault = dict(t_s=round(t, 2), reason='beam_%s' % a['why'], beam=name, note=a['note'])
        frozen = self.alarm is not None or not view.health['ok']
        if frozen and not self.frozen:
            self.counts['freezes'] += 1
            self.frozen_since = t
            self._event(t, 'freeze', why=view.health.get('why') if not view.health['ok'] else self.alarm['why'])
        elif not frozen and self.frozen:
            self._event(t, 'unfreeze')
        elif frozen and t - self.frozen_since > FREEZE_MAX_S and self.fault is None:
            self.fault = dict(t_s=round(t, 2), reason='vision_lost', why=view.health.get('why'),
                              at=view.health.get('at'), note='not found again within %.0f s' % FREEZE_MAX_S)
        self.frozen = frozen

    def _no_progress(self, tid, t):
        h = self.hist.get(tid, [])
        if not h or h[0][0] > t - HANG_S + 1e-9:
            return False
        past = next(x for tt, x in reversed(h) if tt <= t - HANG_S + 1e-9)
        return h[-1][1] - past < .02 * HANG_S

    def _section(self, t, view):
        """Hold the section upstream only when B cannot take the next lump; jog a lump hanging on the edge;
        everything stops while the line is frozen."""
        x_e = self.g['buffer']['x0']
        if self.frozen:
            self.u_goal, self.jog = 0., None
            return
        on_b = [o for it in self.line if it['stage'] in ('buffer', 'transfer') for o in it['objs']]
        if self.jog is not None:
            j = self.jog
            if j not in view or view[j]['x0'] > x_e + CLEAR or self.b_goal == 0.:
                self.jog = None
                self._event(t, 'jog_done', tid=j)
            else:
                return
        bridging = [o for o in on_b if o['x0'] < x_e]
        room = not on_b or min(o['x0'] for o in on_b) >= x_e + ENTRY_ROOM
        taken = {tid for it in self.line for tid in it['now']}
        approaching = any(tid not in taken and x_e - HOLD_ZONE < o['cx'] <= x_e for tid, o in view.items())
        hold = (approaching and (self.b_goal == 0. or not room)) or (bridging and self.b_goal == 0.)
        hanging = [o['tid'] for o in bridging if o['x1'] > x_e and self._no_progress(o['tid'], t)] \
            if self.b_goal > 0. else []
        if hanging and not hold:
            self.jog, self.u_goal = hanging[0], JOG
            self.counts['jogs'] += 1
            self._event(t, 'jog', tid=hanging[0])
            return
        if hold and self.u_goal == 1.:
            self.counts['section_holds'] += 1
            self._event(t, 'section_hold', why='buffer stopped' if self.b_goal == 0. else 'no room on the buffer')
        elif not hold and self.u_goal < 1.:
            self._event(t, 'resume')
        self.u_goal = 0. if hold else 1.

    def _plate(self, t, path_clear):
        c = self.sep
        if self.plate in ('opening', 'closing'):
            if abs(self.plate_deg - self.plate_goal) <= PLATE_TOL_DEG:
                self.plate = 'open' if self.plate_goal == c.open_deg else 'closed'
                self._event(t, 'plate_' + self.plate, angle_deg=round(self.plate_deg, 2))
            elif t - self.plate_t0 > PLATE_FAULT * self.cfg['separator_swing_s'] and self.fault is None:
                self.fault = dict(t_s=round(t, 2), reason='separator_fault', plate=self.plate,
                                  angle_deg=round(self.plate_deg, 2), goal_deg=self.plate_goal)
                return
        if self._stage('discharge') or not path_clear or self.frozen:
            return                                      # a lump is on its way over the plate: do not move
        nxt = next((e for e in self.line if e['stage'] == 'decided'), None)
        if nxt is not None:
            want = 'open' if nxt['route'] == 'gangue' else 'closed'
        elif self._stage('transfer', 'onm', 'measure', 'hold'):
            return                                      # the next route is about to be known: stay put
        else:
            want = 'closed'                             # rest position
        goal = c.open_deg if want == 'open' else c.closed_deg
        if goal != self.plate_goal or self.plate not in (want, 'opening', 'closing'):
            # at rest in the other position, or on its way there: go (a swing reverses at once)
            self.plate = 'opening' if want == 'open' else 'closing'
            self.plate_goal, self.plate_t0 = goal, t
            self.counts['plate_moves'] += 1
            self._event(t, 'plate_' + self.plate, item=None if nxt is None else nxt['item'])
            if nxt is not None:
                nxt['t_plate_move_s'] = round(t, 3)

    def _watch(self, t):
        """A lump in hand and no state change for STALL_S: stop the run."""
        key = (tuple(it['stage'] for it in self.line), self.plate, len(self.items),
               sum('t_clear_s' in it for it in self.sent), self.jog, self.b_goal, self.u_goal)
        busy = (bool(self.line) or any('t_clear_s' not in it for it in self.sent)) and not self.frozen
        if key != self.key or not busy:
            self.key, self.since = key, t
        elif t - self.since > STALL_S and self.fault is None:
            self.fault = dict(t_s=round(t, 2), reason='station_stall', plate=self.plate,
                              items=[(it['item'], it['stage']) for it in self.line],
                              not_cleared=[it['item'] for it in self.sent if 't_clear_s' not in it])

    def _event(self, t, event, **info):
        self.events.append(dict(t_s=round(t, 3), event=event, **info))

    # ---- output ----------------------------------------------------------------------------------
    def status_lines(self):
        """Video overlay: station states, and the last measured item."""
        where = lambda st: ' '.join('#%d%s' % (it['item'] + 1, STAGE_ZH.get(it['stage'], it['stage']))
                                    for it in self.line if it['stage'] in st) or '空'
        flags = ('（停，等计量带）' if self.b_goal == 0. and not self.frozen else '') + \
                ('  （上游暂停）' if self.section_held and not self.frozen else '') + \
                ('  【全线暂停：%s】' % (self.alarm['why'] if self.alarm else '视觉') if self.frozen else '')
        lines = [('缓冲带：%s  计量带：%s  排料板：%s %.0f°%s'
                  % (where(('buffer',)), where(('transfer', 'onm', 'measure', 'decided', 'discharge', 'hold')),
                     PLATE_ZH.get(self.plate, self.plate), self.plate_deg, flags), False, (60, 45, 110))]
        done = [it for it in self.items if 't_decided_s' in it]
        if done:
            it = done[-1]
            lines.append(('第 %d 件：称得 %.1f kg  体积 %s  密度 %s → %s'
                          % (it['item'] + 1, it['mass_kg'],
                             '%.1f L' % (1000 * it['volume_m3']) if it.get('volume_m3') else '无效',
                             '%.0f kg/m³' % it['density_kg_m3'] if it.get('density_kg_m3') else '—',
                             '作废停住（%s）' % '、'.join(REASON_ZH[r] for r in it['reasons']) if it['void']
                             else ('矸石' if it['route'] == 'gangue' else '煤')),
                          False, (170, 40, 40) if it['void'] else (120, 70, 30) if it.get('route') == 'gangue'
                          else (40, 40, 45)))
        return lines

    def report(self):
        done = [it for it in self.items if 't_decided_s' in it]
        valid = [it for it in done if not it['void']]
        clean = lambda it: {k: (sorted(v) if isinstance(v, (set, frozenset)) else v) for k, v in it.items()
                            if k not in ('objs', 'now', 'tids')}
        # verification: a valid item must be one true lump, weighed alone, within 2 % (true state); and no lump may
        # land in a bin without such an item (sorted unmeasured)
        measured = {k for it in valid if it.get('truth_single') and 't_discharge_s' in it for k in it['truth_lumps']}
        states, bins = self.audit.get('states') or [], self.audit.get('bins') or []
        unmeasured = [k for k, st in enumerate(states) if st == 'sorted' and k not in measured]
        false_valid = [it['item'] for it in valid if not it['truth_single'] or not it.get('truth_isolated', True)
                       or it['mass_error_pct'] is None or abs(it['mass_error_pct']) > 2.]
        false_void = [it['item'] for it in done if it['void'] and it['truth_single'] and it.get('truth_isolated', True)
                      and it.get('mass_error_pct') is not None and abs(it['mass_error_pct']) <= 2.
                      and 'scan' not in it['reasons']]
        reasons = {}
        for it in done:
            for r in it['reasons']:
                reasons[r] = reasons.get(r, 0) + 1
        return dict(geometry=report(self.g), items=[clean(it) for it in self.items], events=self.events,
                    counts=dict(self.counts, items=len(self.items), measured=len(done), valid=len(valid),
                                void=len(done) - len(valid), void_reasons=reasons,
                                gangue=sum(it.get('route') == 'gangue' for it in valid),
                                coal=sum(it.get('route') == 'coal' for it in valid),
                                route_not_as_ideal=sum(not it['route_as_ideal'] for it in valid),
                                bins_not_as_route=sum(it.get('bins_match_route') is False for it in self.items),
                                landed=sum('t_landed_s' in it for it in self.items),
                                void_discharged=sum(it['void'] and 't_discharge_s' in it for it in done)),
                    verification=dict(false_valid=false_valid, false_void=false_void, landed_unmeasured=unmeasured,
                                      rule='valid = one true lump, touching nothing off the weigh frame, weighed '
                                           'within 2 %; false_void excludes scan failures (a scan can fail on a '
                                           'good lump); landed_unmeasured = lumps in a bin without a valid '
                                           'single-lump item discharged'),
                    mass_error_pct_abs_max=max((abs(it['mass_error_pct']) for it in valid
                                                if it['mass_error_pct'] is not None), default=None),
                    upstream_held_s=round(self.held_s, 2), buffer_stopped_s=round(self.buffer_stopped_s, 2),
                    frozen_s=round(self.frozen_s, 2), fault=self.fault,
                    s2_interruptions=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for e in self.s2],
                    final=dict(plate=self.plate, plate_deg=round(self.plate_deg, 2),
                               items=[(it['item'], it['stage']) for it in self.line]),
                    control='void items are held on the measuring belt (no discharge, no plate move, no release), '
                            'then taken off the line by the simulation (manual re-measuring not modelled); section '
                            'upstream stops only when the buffer cannot take the next lump; the buffer stages its '
                            'lead item at S3 (or where the cameras see it past S3) while the measuring belt is busy; '
                            'the measuring belt weighs (indicator: %.1f s average steady within %.0f %% for %.1f s) and '
                            'scans (%.1f s) at the same time; route = gangue if mass/volume >= %.0f kg/m3; the plate '
                            'moves as soon as the route is known and S4 plus the plate-zone camera say its path is '
                            'clear; the discharge starts once the plate will be in position %.1f s before the lump '
                            'reaches it' % (FILTER_S, 100 * STEADY_BAND, STEADY_S, self.cfg['scan_s'],
                                            self.cfg['sort_density'], PLATE_MARGIN_S))
