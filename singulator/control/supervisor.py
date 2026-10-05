"""The rules of the line that no single controller owns: what ties the feed belt, the plough face and the station
together.

    drive targets   every physics step: the start-up ramp times the station's hold of the section upstream for the
                    main and side belts, the feed belt control's target (zero while it is paused), the station's
                    for its two belts;
    section         every sample: the singulation section as the cameras see it (SectionWatch) -- the plough
                    face's closing permission and the stall rule;
    interlock       the feed belt lets nothing go while the station holds the section upstream, holds a void item,
                    or an alarm stops the line. Its drop beam S1 is taken for dead when DEAD_AFTER releases in a
                    row went past it unseen (the station then stops the line: the alarm is the beam's);
    stall           a stall in the section with nothing leaving the lane retracts the plough face, at most
                    unjam_max times a batch; then the line stops;
    stops           for the operator: the face retracted and not allowed to close (it never closes onto material),
                    a fault of the station, a fault of the face.

Like the controllers it reads sensor signals only.
"""
import numpy as np

from . import station as station_rules

SECTION_BACK = .3      # m: the singulation section starts this far before the plough start (and ends at the lane exit)
STALL_AFTER_S = 3.     # s: no stall is acted on until this long after the start-up ramp


class SectionWatch:
    """Vision only: the singulation section (plough start - 0.3 m .. lane exit) as the cameras see it -- which
    objects are in it, which have not advanced jam_speed * jam_window over jam_window (the stall rule on the
    tracked outline centroid), and whether a tail left the lane recently."""

    def __init__(self, cfg, d, lo):
        self.cfg, self.lo, self.exit_x = cfg, lo, d['exit_x']
        self.hist, self.since, self.tail = {}, {}, {}

    def update(self, t, view, waiting):
        W = self.cfg['jam_window']
        section = {tid: o for tid, o in view.items() if o['x0'] < self.exit_x and o['x1'] > self.lo}
        for tid, o in view.items():
            h = self.hist.setdefault(tid, [])
            h.append((t, o['cx']))
            while len(h) > 2 and h[1][0] <= t - W + 1e-9:
                h.pop(0)
            if o['x0'] > self.exit_x and tid not in self.tail and o['x0'] < self.exit_x + .5:
                self.tail[tid] = t
        stuck = []
        for tid, o in section.items():
            if waiting(o):
                self.since.pop(tid, None)
                continue
            s = self.since.setdefault(tid, t)
            h = self.hist[tid]
            if t - s >= W - 1e-9 and h[0][0] <= t - W + .02 and h[-1][1] - h[0][1] < self.cfg['jam_speed'] * W:
                stuck.append(tid)
        for tid in [k for k in self.since if k not in section]:
            del self.since[tid]
        passing = any(tt > t - W for tt in self.tail.values())
        return section, stuck, passing

    @staticmethod
    def plans(section, margin):
        """Outlines grown by the cameras' position margin, for the face's closing-sweep check."""
        out = {}
        for tid, o in section.items():
            P = o['pts']
            c = P.mean(0)
            r = np.maximum(np.linalg.norm(P - c, axis=1, keepdims=True), 1e-9)
            out[tid] = P + (P - c) / r * margin
        return out


class Supervisor:
    def __init__(self, cfg, d, feeder, face, station, margin=0.):
        """margin: the position margin on camera-judged edges (sensing.suite.Sensors.margin)."""
        self.cfg, self.d = cfg, d
        self.feeder, self.face, self.station, self.margin = feeder, face, station, margin
        self.watch = SectionWatch(cfg, d, d['P0'][0] - SECTION_BACK)
        self.section, self.stuck, self.passing = {}, [], False
        self.stop = None                    # why the line was stopped for the operator (outcome.jam_stop)

    def drive_targets(self, t, dt):
        """One physics step. Returns the speed factors asked of (main and side belts, feed belt, buffer belt,
        measuring belt); the station moves its plate reference in the same step."""
        startup = min(1., t / self.cfg['ramp']) if self.cfg['ramp'] > 0 else 1.
        u, b, m = self.station.step(dt)         # the section upstream, the buffer belt, the measuring belt
        return startup * u, (0. if self.feeder.paused else self.feeder.drive_target(dt)), b, m

    def see_section(self, t, view):
        """The section this sample, and the face's closing permission: the section is clear, and no outline (grown
        by the cameras' position margin) reaches into what the face will sweep on its way home -- checked again
        while it closes."""
        st = self.station
        x_edge = st.g['buffer']['x0']
        # held by the station upstream of the head edge: waiting, not stalled
        self.section, self.stuck, self.passing = self.watch.update(
            t, view, lambda o: st.section_held and o['cx'] <= x_edge)
        self.face.zone_busy = bool(self.section)
        self.face.check_sweep(SectionWatch.plans(self.section, self.margin))

    def feed(self, t, view, beam):
        """The feed belt's sample under the interlock. view: the feed belt control's view; beam: its drop beam S1
        (sensing.beams.Beam)."""
        f, st = self.feeder, self.station
        if st.section_held or st.inhibit_release:
            f.pause(t)                          # the station holds the section, a void item, an alarm: no release
            return
        f.resume(t)
        f.observe(t, view, beam.blocked, self.face.at_home)
        if f.miss_streak >= station_rules.DEAD_AFTER and beam.alarm is None:
            beam.alarm = dict(t_s=round(t, 3), beam=beam.name, why='dead', stop=True,
                              note='two releases in a row went without the beam')

    def judge(self, t, view):
        """After the controllers' samples: retract the face on a stall, or stop the line. Returns the stop (also
        kept in self.stop), None while the line runs."""
        cfg, face, st = self.cfg, self.face, self.station
        left, tried = len(self.section), len(face.events)
        if face.hold_timed_out(t):
            # retracted and still not clear: the machine has nothing left to try. Stop with the face open --
            # closing now would ram the lumps with the actuator
            stop = dict(t_s=round(t, 2), reason='retract_hold_timeout', blocks_left=left, stalled_blocks=self.stuck,
                        in_closing_sweep=face.in_sweep, unjam_pulses_tried=tried,
                        action='face stays retracted; stop conveying; operator clears the section')
        elif st.fault:
            # the separator did not get into position, a sensor alarm, or a lump in hand without progress
            stop = dict(st.fault, blocks_left=left, unjam_pulses_tried=tried,
                        action='stop the line; operator clears the station')
        elif face.fault:
            # the face did not get home: no further action on this batch
            stop = dict(face.fault, reason='face_motion_fault', cause=face.fault['reason'], blocks_left=left,
                        unjam_pulses_tried=tried,
                        action='terminate simulation; physical stopping transient not simulated')
        elif (self.stuck and not self.passing and face.phase == 'idle' and t > cfg['ramp'] + STALL_AFTER_S
              and t > face.next_ok):
            if tried < cfg['unjam_max']:
                face.start(t, blocks_left=left, stalled_blocks=self.stuck,
                           centroid_x=[round(view[k]['cx'], 2) for k in self.stuck])
                return None
            stop = dict(t_s=round(t, 2), reason='unjam_exhausted', blocks_left=left, stalled_blocks=self.stuck,
                        unjam_pulses_tried=tried)
        else:
            return None
        self.stop = stop
        return stop
