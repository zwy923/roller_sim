"""The line put together for one batch, and advanced in time.

Line(cfg) builds everything once: the batch, the model, the simulated plant (physics/), the sensors (sensing/), the
controllers and their supervisor (control/), the witnesses (verify/) and the scene's set-up (layouts, scenarios).
advance() then runs it to the next 10 ms sample:

    every physics step (_step)   the controllers' targets go to the drives and actuators, MuJoCo steps, the motors
                                 answer the loads of the step;
    every 10 ms (_sample)        the true state is recorded, the sensors are formed from it, the controllers read
                                 the sensors and decide. The order of that sample is the order of the method: read
                                 it there.

Information flows one way: truth -> sensors -> controllers -> drive targets. The witnesses and the record read the
truth next to it and feed nothing back; the two exceptions are the simulation's own hand (TakeOff, Scenario), which
moves lumps and tells the station what it took.
"""
import math

import mujoco
import numpy as np

from ..config import SAMPLE_S
from ..control.face import FaceRetract
from ..control.feeder import Feeder
from ..control.side_gate import SideGate
from ..control.station import Station
from ..control.supervisor import Supervisor
from ..lumps import make_blocks, set_mass_properties
from ..machine import derive
from ..machine.assembly import build_xml
from ..physics.actuators import FaceServo, PlateDrive
from ..physics.drives import Conveyors
from ..physics.lumps import END_STATES, Lump
from ..physics.numerics import Diagnostics
from ..physics.side_gate import GateDrive
from ..sensing.suite import Sensors
from ..verify.entry import EntryWatch
from ..verify.feeder import FeederWitness
from ..verify.station import StationWitness, frame_contacts
from ..verify.transfer import TransferWatch
from .layouts import arrange, plan_scatter
from .record import Trace
from .scenarios import Scenario, TakeOff

INITIAL_OVERLAP = .001      # m: a batch laid deeper than this into equipment or another lump is not a valid start
DONE_AFTER_S = 2.           # s after the start-up ramp: from here the run ends once every lump has ended


class Line:
    def __init__(self, cfg):
        self.cfg = cfg
        # ---- the batch, the machine, the model ---------------------------------------------------------
        rng = np.random.default_rng(cfg['seed'])
        blocks = self.blocks = make_blocks(cfg, rng)
        d = self.d = derive(cfg)
        self.xml = build_xml(cfg, d, blocks)
        model = self.model = mujoco.MjModel.from_xml_string(self.xml)
        data = self.data = mujoco.MjData(model)
        lumps = self.lumps = [Lump(model, k) for k in range(len(blocks))]
        for L, b in zip(lumps, blocks):
            set_mass_properties(b, model.body_mass[L.body])
        self.geom_lump = {L.geom: L.k for L in lumps}
        # ---- the scene at t = 0 ------------------------------------------------------------------------
        layer, self.feed_rear, self.feed_front = plan_scatter(cfg, blocks, rng, d)
        for k, x, y, z, yaw in layer:
            lumps[k].place(data, x, y, z, yaw)
        self.belts = Conveyors(cfg, d, model)
        self.belts.command(data)                    # place every slat before the first contact evaluation
        self.plate = PlateDrive(model, d['station']['sep'])
        self.plate.place(data)                      # separator plate down (coal), linkage closed
        mujoco.mj_forward(model, data)
        self.scenario = Scenario(cfg, d, model, data, lumps, blocks)
        self.scenario.setup()                       # touching: two lumps laid touching in the lane
        # ---- sensors, controllers, witnesses -----------------------------------------------------------
        sensors = self.sensors = Sensors(cfg, d, model, data, lumps, blocks)
        self.station = Station(cfg, d, sensors.weigher, sensors.scanner, sensors.margin)
        self.witness = StationWitness(cfg, blocks).attach(self.station)     # the station's record against the truth
        self.takeoff = TakeOff(d, model, data, lumps)
        self.layout = arrange(cfg, d, model, data, lumps, rng)      # a bench layout; scatter / flat: as placed
        self.transfer = TransferWatch(cfg, d, model, lumps)
        self.feeder = Feeder(cfg, d)
        self.feeder.start(0.)                       # the whole batch lies on the feed belt; the first release starts at once
        self.feed_witness = FeederWitness(self.feeder, len(lumps))  # which lumps went in which release
        self.face = FaceRetract(cfg, FaceServo(cfg, d, model, data))
        self.supervisor = Supervisor(cfg, d, self.feeder, self.face, self.station, sensors.margin)
        self.diagnostics = Diagnostics(model, cfg)
        self.gone = frozenset()                     # geoms of the lumps that have left the line (see _sample)
        self.trace = Trace(cfg, d, model, lumps)
        # the experimental side gate (--side-pusher), and the lane-entrance audit that goes with it
        self.gate = SideGate(cfg, d) if cfg['side_pusher'] != 'off' else None
        self.gate_drive = GateDrive(cfg, self.gate.g, model) if self.gate else None
        self.entry_watch = EntryWatch(d, lumps) if self.gate else None
        if self.gate:
            self.feeder.pause(0.)                   # the gate's start-up look comes first
            self.trace.traj['gate_m'], self.trace.traj['gate_buffer_m'] = [], []
        # Layout overlap is an invalid initial condition, not an impact generated by the machine.
        deepest = min(data.contact[:data.ncon], key=lambda c: c.dist, default=None)
        self.initial_pen = max(0., -float(deepest.dist)) if deepest is not None else 0.
        if self.initial_pen > INITIAL_OVERLAP:
            name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or 'geom %d' % g
            raise ValueError('initial overlap %.6f m (%s into %s) exceeds 1 mm'
                             % (self.initial_pen, name(deepest.geom[1]), name(deepest.geom[0])))
        # ---- time ----------------------------------------------------------------------------------------
        self.dt = model.opt.timestep
        self.stride = max(1, round(SAMPLE_S / self.dt))     # physics steps per sample
        self.n_steps = int(math.ceil(cfg['duration'] / self.dt))
        self.steps = 0
        self.t = 0.                                 # time of the last sample
        self.ended = None                           # why the run ended: 'numerics', 'bench', 'stop', 'done', 'duration'

    # ---- the run -----------------------------------------------------------------------------------------
    def advance(self):
        """Run to the next 10 ms sample and take it. False once the run has ended (self.ended says why)."""
        while self.steps < self.n_steps:
            if not self._step():
                self.ended = 'numerics'
                return False
            self.steps += 1
            if self.steps % self.stride == 0 or self.steps == self.n_steps:
                self.ended = self._sample()
                return self.ended is None
        self.ended = 'duration'
        return False

    def _step(self):
        """One physics step. False when the numerical screen failed on it."""
        data, t = self.data, self.data.time
        section, feed, buffer, measure = self.supervisor.drive_targets(t, self.dt)
        self.plate.command(data, self.station.plate_ref)
        self.belts.targets(section, feed, buffer, measure)
        self.belts.command(data)
        self.face.command(t)
        if self.gate:
            self.gate_drive.command(data, self.gate.targets, self.dt)
        mujoco.mj_step(self.model, data)
        if self.gate:
            self.gate_drive.record(data, self.dt)
        self.diagnostics.observe(data, t)       # forces/contacts that produced this integration step
        if self.diagnostics.failure:
            return False
        self.face.record(t)
        self.belts.after_step(data, t, self.dt)
        self.sensors.step(self.station.weighing)
        return True

    def _sample(self):
        """The 10 ms sample. Returns why the run ends here, or None."""
        cfg, data, lumps = self.cfg, self.data, self.lumps
        sensors, station, feeder, face, sup = self.sensors, self.station, self.feeder, self.face, self.supervisor
        # mj_step leaves derived poses and contacts at the START of the step, but qpos/time have advanced.
        # Restore the held conveyor surfaces, then synchronise all observation fields. This is independent
        # of video and cannot be triggered only by rendering.
        self.belts.command(data)
        mujoco.mj_forward(self.model, data)
        t = self.t = float(data.time)
        # ---- the truth, recorded -----------------------------------------------------------------------
        self.trace.observe(t, data, self._waiting)
        if self.entry_watch:
            self.entry_watch.observe(t)
        # ---- sensors -----------------------------------------------------------------------------------
        view = sensors.frame(t)
        # ---- the plough face: closing permission; what it touches --------------------------------------
        sup.see_section(t, view)
        face_N = self.trace.face_force(data)
        face.contact(face_N, self.trace.dt_sample)
        # ---- the station -------------------------------------------------------------------------------
        self.witness.truth([L.state for L in lumps], [L.bin for L in lumps],
                           *frame_contacts(data, sensors.load_cell.frame, self.geom_lump))
        station.observe(t, view, sensors.beams, self.plate.angle(data))
        self.takeoff.act(t, station, sensors)       # the simulation's hand: a held item leaves the line
        self.scenario.act(t, station)
        gone = frozenset(L.geom for L in lumps if L.state in END_STATES)
        if gone != self.gone:                       # lumps off the line: their landing is not screened
            self.gone = gone
            self.diagnostics.set_landed(gone)
        # ---- the feed belt -----------------------------------------------------------------------------
        gate = self.gate
        if gate:
            # the gate decides only while the feed belt runs forward: a release, or staging
            still = feeder.jog is not None or (feeder.phase != 'feeding' and not feeder.staging)
            gate.observe(t, sensors.feed_view, self.gate_drive.feedback(data),
                         self.belts.feed.f * cfg['feeder_speed'], station.section_held or station.inhibit_release
                         or (gate.phase in ('startup', 'idle', 'brake') and still),
                         beam=sensors.beams['beam_feed'].blocked)
            feeder.hold_back(gate.current['held'] if gate.phase == 'hold' else ())
        if gate and gate.blocking:
            feeder.pause(t)
            if sensors.beams['beam_feed'].blocked:
                feeder.saw_beam()           # the lead tips into the beam while the plate retracts
        else:
            sup.feed(t, sensors.feed_view, sensors.beams['beam_feed'])
        if not feeder.paused or gate and gate.blocking:
            self.feed_witness.observe(t, {L.k: L.last_pose[0] for L in lumps if L.state == 'on_belt'})
        self.transfer.observe(t, data, feeder, self.belts.feed.f)
        if cfg['bench'] and all(r['gap_before_m'] is not None for r in self.transfer.rec.values()) and all(
                s.get('done') for s in self.transfer.stops):
            return 'bench'                          # every lump lies on the main belt, its questions are answered
        self.trace.frame(t, data, face.theta, face_N, self.belts.factors(), sensors.weigher.reading,
                         station.plate_deg)
        if gate:
            fb = self.gate_drive.feedback(data)
            self.trace.traj['gate_m'].append(fb['position_m'])
            self.trace.traj['gate_buffer_m'].append(fb['buffer_m'])
            if gate.fault:
                sup.stop = gate.fault
                return 'stop'
        # ---- stall -> retract, or stop for the operator ------------------------------------------------
        if sup.judge(t, view):
            return 'stop'
        if t > cfg['ramp'] + DONE_AFTER_S and all(L.state in END_STATES for L in lumps) and not station.path_pending:
            return 'done'
        return None

    def _waiting(self, L):
        """Planned waiting, for the stall bookkeeping of the lumps' record only: queued on the feed belt, or held
        (or taken) by the station."""
        if self.feed_witness.waiting(L.k, L.state, L.last_pose[0]):
            return 'feeder'
        if self.witness.waiting(L.k, L.state, L.last_pose[0]):
            return 'station'
        return False

    # ---- for the video -----------------------------------------------------------------------------------
    def followed(self):
        """Front-edge x of the lumps the camera follows: those released and still on the machine (all of them
        while none is)."""
        went = self.feed_witness.went
        live = [L.k for L in self.lumps if L.state not in END_STATES and L.k in went]
        return [self.lumps[k].s_hist[-1][1] for k in (live or range(len(self.lumps)))]
