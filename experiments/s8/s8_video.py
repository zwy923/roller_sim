"""Close-up video of one feed belt release (S8; user, 2026-10-06: 渲染改法二一次放下多块的视频).

    python experiments/s8/s8_video.py --layout aligned --seed 7231 --count 4 --out runs/s8_video/aligned_7231
    MUJOCO_GL=glfw xvfb-run -a python experiments/s8/s8_video.py ...      (a headless Linux: no EGL / OSMesa here)

Any plough.py arguments pass through; without any the line runs as it is (--feeder-stop centroid since S8). The
batch runs twice. The first run renders nothing: it finds the release (--release N, or the first one that let more
than one lump go) and the items those lumps ended up in. The second run is the same batch -- the render only reads
the state and recolours geoms -- and draws two clips:
  1. the release, slowed down --slow times (every 10 ms sample is a frame): a side view and a three-quarter view of
     the head edge, a plan view beside them. The lumps of the release in colour, the others faded; the near skirt and
     the camera gantries see-through. Marked: the head edge (yellow), the drop beam S1 (red, it is part of the model),
     each coloured lump's true centroid (a dot, white until it is past the edge, then yellow). The header: the feed
     belt's actual speed and what its controller is doing, the beam, how far each coloured centroid is past the
     edge, and the events as they happen (a centroid over the edge, the stop command, a jog);
  2. those lumps arriving on the measuring belt, at 1x, with what the station made of them.
The second run's releases and crossing times are checked against the first's before anything is written. Outputs in
--out: video.mp4, result.json (the first run's result, as plough.py writes it) and clip.json (what the clip
shows: the release, its crossing and stop times, the events, the items, the time windows).
Nothing here is a measurement: it shows the model.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from singulator import tuning  # noqa: E402
from singulator.config import parse_config, validate  # noqa: E402
from singulator.sim import results  # noqa: E402
from singulator.sim.line import Line  # noqa: E402
from singulator.sim.video import FONT, REASON_ZH, station_lines  # noqa: E402

W, H, HEAD = 1280, 720, 104                     # frame, and the header strip on top of it
FPS = 25
COLOURS = [(1., .50, .05), (.15, .45, 1.), (.15, .75, .30), (.85, .20, .75)]   # the release's lumps, in crossing order
NAMES_ZH = ['橙', '蓝', '绿', '紫']
FADED = (.55, .55, .58, .30)
LAYOUT_ZH = dict(scatter='随机', aligned='并齐', touching='相贴', oblique='斜放', flat='扁平')


def font(size):
    return ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()


def run(cfg, sample=None, stop_after=None):
    """The batch through the line; sample(line) every 10 ms. Stops early once line.t > stop_after."""
    with tuning.applied(cfg['set']):
        line = Line(cfg)
        if sample:
            sample(line, first=True)
        while line.advance():
            if sample:
                sample(line)
            if stop_after is not None and line.t > stop_after:
                break
        return line, results.assemble(line, 0.)


def pick(r, which):
    """The release to show and the station items its lumps ended up in, from a finished run's result."""
    rels = [x for x in r['feeder']['releases'] if x['members']]
    rel = (next(x for x in r['feeder']['releases'] if x['release'] == which) if which is not None
           else next((x for x in rels if len(x['members']) > 1), None))
    if rel is None:
        raise SystemExit('no release let more than one lump go in this batch; pick one with --release')
    tip = {x['lump']: x['t_tip'] for x in r['transfer']['lumps']}
    members = sorted(rel['members'], key=lambda k: tip[k])
    items = [it for it in r['station']['items'] if set(it.get('truth_lumps') or []) & set(members)]
    return rel, members, tip, items


class Clip:
    """Renders the frames (see the module docstring)."""

    def __init__(self, line, args, rel, members, tip, items, label):
        self.line, self.args, self.rel, self.members, self.label = line, args, rel, members, label
        m, d, cfg = line.model, line.d, line.cfg
        self.x1, self.step, self.lane_y, self.belt_w = d['feeder']['x1'], d['feeder']['step_m'], cfg['lane_y'], \
            cfg['belt_w']
        self.beam = d['feeder']['beam']
        t_tips = [tip[k] for k in members]
        self.a = (t_tips[0] - .6, max(max(t_tips), rel['t_stop_s'] or 0.) + 1.5)       # clip 1, s
        ins = [it['t_in_s'] for it in items]
        ends = [it.get('t_hold_s') or it.get('t_discharge_s') or it.get('t_decided_s') or it['t_in_s'] + 6.
                for it in items]
        self.b = (min(ins) - .8, max(ends) + 1.5) if items else None                   # clip 2, s
        self.items = items
        # colours: the release's lumps by crossing order, every other lump faded; see-through near skirt, gantries
        self.col = {k: COLOURS[i % len(COLOURS)] for i, k in enumerate(members)}
        for L in line.lumps:
            m.geom_rgba[L.geom] = (*self.col[L.k], 1.) if L.k in self.col else FADED
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            if name in ('skirt_in', 'bskirt_in', 'mskirt_in'):
                m.geom_rgba[g][3] = .12
            elif name.startswith(('dev_cam', 'dev_scanner')):
                m.geom_rgba[g][3] = 0.
        self.views = dict(side=mujoco.Renderer(m, 308, 640), oblique=mujoco.Renderer(m, 308, 640),
                          top=mujoco.Renderer(m, H - HEAD, 640), station=mujoco.Renderer(m, H - HEAD, W))
        cams = dict(side=(90., -5., 1.05, (self.x1 + .10, .62, .14)),
                    oblique=(48., -28., 1.55, (self.x1 + .02, .62, .10)),
                    top=(90., -90., 1.62, (self.x1 + .02, .63, .10)))
        st = d['station']
        mid = ((st['buffer']['x0'] + st['measure']['x1']) / 2, st['y_c'], st['top_z'])
        cams['station'] = (68., -30., 2.7, mid)
        self.cam = {}
        for k, (az, el, dist, look) in cams.items():
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.azimuth, c.elevation, c.distance = az, el, dist
            c.lookat[:] = look
            self.cam[k] = c
        self.big, self.mid, self.small = font(24), font(19), font(16)
        self.events, self.cut = [], None
        self.frames, self.state = [], {}

    # ---- the scene ------------------------------------------------------------------------------
    def _geom(self, scn, kind, size, pos, rgba, mat=None):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, float), np.asarray(pos, float),
                            np.eye(3).ravel() if mat is None else mat, np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def _marks(self, scn, view):
        """The head edge and the coloured lumps' centroids, drawn into a view's scene."""
        d, x1, h = self.line.data, self.x1, self.step
        box, sph = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE
        if view == 'side':     # the edge as a vertical stroke in front of everything
            self._geom(scn, box, (.0025, .0025, .16), (x1, self.lane_y - .05, h + .06), (1., .85, 0., 1.))
        else:                  # the edge as a line across the belt, on the feed belt's top
            self._geom(scn, box, (.004, (self.belt_w - self.lane_y) / 2, .002),
                       (x1, (self.belt_w + self.lane_y) / 2, h + .003), (1., .85, 0., 1.))
        for L in self.line.lumps:
            if L.k not in self.col:
                continue
            c = d.xipos[L.body]
            V = L.world(d)
            past = c[0] > x1
            rgba = (1., .85, 0., 1.) if past else (1., 1., 1., 1.)
            if view == 'side':
                pos = (c[0], float(V[:, 1].min()) - .03, c[2])          # in front of the lump, on the camera's side
            else:
                pos = (c[0], c[1], float(V[:, 2].max()) + .015)         # on top of it
            self._geom(scn, sph, (.013, 0., 0.), pos, rgba)

    def _render(self, view, marks=True):
        r = self.views[view]
        r.update_scene(self.line.data, self.cam[view])
        if marks:
            self._marks(r.scene, view)
        return Image.fromarray(r.render())

    # ---- the header -----------------------------------------------------------------------------
    def _feed_state(self):
        line = self.line
        f, fe = line.belts.feed.f, line.feeder
        v = max(0., f * line.cfg['feeder_speed'])           # the brake leaves it at -0.0
        if fe.jog is not None:
            what = '点动'
        elif fe.phase == 'feeding' and fe.goal >= 1.:
            what = '前送'
        elif fe.phase == 'feeding':
            what = '慢走'
        elif v > 1e-4:
            what = '刹车中'
        else:
            what = '停'
        return v, what

    def _watch(self, t):
        """Events of the release, as they happen."""
        line, rel = self.line, self.line.feeder.releases[self.rel['release']] \
            if len(self.line.feeder.releases) > self.rel['release'] else None
        for k in self.members:
            tt = line.transfer.rec[k]['t_tip']
            if tt is not None and ('tip', k) not in self.state:
                self.state[('tip', k)] = tt
                v, what = self._feed_state()
                self.events.append((tt, '料 %d（%s）质心过边' % (k, NAMES_ZH[self.members.index(k)])
                                    + ('——给料带已停' if what == '停' else '——给料带在刹车' if what == '刹车中'
                                       else '')))
        if rel is not None and rel['t_stop_s'] is not None and 'stop' not in self.state:
            self.state['stop'] = rel['t_stop_s']
            why = {'went': '停带命令（质心过边 %.0f mm）' % (1000 * (line.feeder.stop_past or 0.)),
                   'beam': '停带命令（S1 被挡）', 'beam_missed': '停带（兜底）'}.get(rel['stop'], '停带：' + rel['stop'])
            self.events.append((rel['t_stop_s'], why))
        if line.feeder.jog is not None and 'jog' not in self.state:
            self.state['jog'] = t
            self.events.append((round(t, 3), '点动（挂边）'))
        beam = line.sensors.beams['beam_feed'].blocked
        if beam and 'beam' not in self.state and t >= self.rel['t_start_s']:
            self.state['beam'] = t
            self.events.append((round(t, 3), 'S1 被挡'))

    def _header(self, img, t, title, lines):
        dr = ImageDraw.Draw(img)
        dr.rectangle((0, 0, W, HEAD), fill=(250, 250, 248))
        dr.text((14, 6), title, font=self.big, fill=(20, 25, 35))
        for i, (text, rgb) in enumerate(lines):
            dr.text((14, 38 + 22 * i), text, font=self.mid if i == 0 else self.small, fill=rgb)

    def _caption(self, img, xy, text):
        dr = ImageDraw.Draw(img)
        x, y = xy
        w = dr.textlength(text, font=self.small)
        dr.rectangle((x - 4, y - 2, x + w + 4, y + 20), fill=(255, 255, 255))
        dr.text((x, y), text, font=self.small, fill=(30, 30, 40))

    # ---- frames ---------------------------------------------------------------------------------
    def release_frame(self, t):
        self._watch(t)
        line, d = self.line, self.line.data
        img = Image.new('RGB', (W, H), (255, 255, 255))
        img.paste(self._render('side'), (0, HEAD))
        img.paste(self._render('oblique'), (0, HEAD + 308))
        img.paste(self._render('top'), (640, HEAD))
        self._caption(img, (8, HEAD + 6), '侧视（近侧挡边已透明）')
        self._caption(img, (8, HEAD + 314), '斜视')
        self._caption(img, (648, HEAD + 6), '俯视：黄线 = 机头边缘，红线 = 落料光束 S1，圆点 = 真实质心（过边后变黄）')
        v, what = self._feed_state()
        beam = line.sensors.beams['beam_feed'].blocked
        cxs = '   '.join('料 %d（%s）质心 %+.0f mm' % (k, NAMES_ZH[i], 1000 * (d.xipos[line.lumps[k].body][0] - self.x1))
                         for i, k in enumerate(self.members))
        ev = ' → '.join('%.2f s %s' % e for e in sorted(self.events)[-4:]) or '—'
        self._header(img, t, self.label, [
            ('t = %.2f s   慢放 %d×   给料带：%s %.3f m/s   S1：%s' % (t, self.args.slow, what, v, '挡' if beam else '通'),
             (20, 60, 40) if what in ('前送', '慢走', '点动') else (130, 40, 30)),
            (cxs, (40, 40, 60)),
            (ev, (120, 30, 30))])
        return img

    def station_frame(self, t):
        line = self.line
        img = Image.new('RGB', (W, H), (255, 255, 255))
        img.paste(self._render('station', marks=False), (0, HEAD))
        lines = [(text, tuple(int(c * .8) for c in rgb)) for text, big, rgb in station_lines(line.station)]
        who = '、'.join('料 %d（%s）' % (k, NAMES_ZH[i]) for i, k in enumerate(self.members))
        verdict = []
        for it in self.items:
            if t >= (it.get('t_decided_s') or it.get('t_hold_s') or 1e9):
                verdict.append('第 %d 件 = 料 %s：%s' % (
                    it['item'] + 1, '+'.join(str(k) for k in it['truth_lumps']),
                    '作废停住（%s）' % '、'.join(REASON_ZH.get(r, r) for r in it['reasons']) if it['void']
                    else '有效，%s' % ('矸石' if it.get('route') == 'gangue' else '煤')))
        self._header(img, t, '%s到计量带（1×）' % who, [
            ('t = %.2f s   ' % t + (lines[0][0] if lines else ''), (40, 40, 80)),
            ('；'.join(verdict) or '—', (170, 30, 30)),
            ('只看这几块：其余料淡显', (90, 90, 100))])
        return img

    def freeze(self, img, seconds):
        """The last picture before a held item is taken off the line (the simulation's hand does that at once), with
        the station's verdict over it."""
        img = img.copy()
        dr = ImageDraw.Draw(img)
        lines = []
        for it in self.items:
            who = '+'.join('料 %d' % k for k in it['truth_lumps'])
            if it.get('void'):
                lines.append('第 %d 件 = %s：作废停住（%s）' % (it['item'] + 1, who,
                                                        '、'.join(REASON_ZH.get(r, r) for r in it['reasons'])))
                lines.append('两块一起上了秤，这一件不算数；线停下，等人把它们分开重测（模型里直接移走）')
            else:
                lines.append('第 %d 件 = %s：有效' % (it['item'] + 1, who))
        y0 = H - 40 - 34 * len(lines)
        dr.rectangle((0, y0 - 12, W, H - 20), fill=(255, 245, 240))
        for i, text in enumerate(lines):
            dr.text((24, y0 + 34 * i), text, font=self.big if i == 0 else self.mid, fill=(170, 30, 30))
        return [img] * int(seconds * FPS)

    def card(self, text, sub, seconds):
        img = Image.new('RGB', (W, H), (245, 245, 242))
        dr = ImageDraw.Draw(img)
        y = H // 2 - 60
        for txt, fnt, rgb in ((text, self.big, (20, 25, 35)),) + tuple((s, self.mid, (60, 60, 70)) for s in sub):
            w = dr.textlength(txt, font=fnt)
            dr.text(((W - w) / 2, y), txt, font=fnt, fill=rgb)
            y += 40
        return [img] * int(seconds * FPS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--release', type=int, help='the release to show (default: the first with more than one lump)')
    p.add_argument('--slow', type=int, default=4, help='the release clip runs this many times slower than real time')
    p.add_argument('--no-station', action='store_true', help='leave out the clip at the measuring belt')
    a, rest = p.parse_known_args()
    cfg = validate(parse_config(rest + ['--no-video']))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1: find the release and where its lumps went
    _, r1 = run(cfg)
    rel, members, tip, items = pick(r1, a.release)
    print('release %d: lumps %s over the edge at %s s, stopped at %s s (%s); items %s'
          % (rel['release'], members, [tip[k] for k in members], rel['t_stop_s'], rel['stop'],
             [(it['item'], it.get('truth_lumps'), it.get('void'), it.get('reasons')) for it in items]), flush=True)

    # 2: the same batch again, rendering around them
    import imageio.v2 as imageio
    label = '改法二（头一块质心过机头 %.0f mm 就停带）· %s %s · 第 %d 次放料一次放下 %d 块' % (
        1000 * (r1['feeder']['geometry'].get('stop_past_m') or 0.), LAYOUT_ZH.get(cfg['layout'], cfg['layout']),
        cfg['seed'], rel['release'] + 1, len(members)) if r1['feeder']['geometry'].get('stop_rule') == 'centroid' \
        else '%s %s · 第 %d 次放料一次放下 %d 块（光束停带）' % (LAYOUT_ZH.get(cfg['layout'], cfg['layout']), cfg['seed'],
                                                     rel['release'] + 1, len(members))
    clip_end = None
    holds = [it['t_hold_s'] for it in items if it.get('t_hold_s') is not None]
    hold = max(holds) if holds else None                 # a held item is taken off the line the same sample
    if not a.no_station:
        ends = [it.get('t_hold_s') or it.get('t_discharge_s') or it.get('t_decided_s') or it['t_in_s'] + 6.
                for it in items]
        clip_end = (hold + .05 if hold is not None else max(ends) + 1.6) if ends else None
    state = {}
    every = max(1, round(100 / (FPS * a.slow)))
    writer = imageio.get_writer(str(out / 'video.mp4'), fps=FPS, codec='libx264', quality=8, macro_block_size=16)

    def sample(line, first=False):
        if first:
            state['clip'] = clip = Clip(line, a, rel, members, tip, items, label)
            gap = ', '.join('%.2f s' % (tip[k] - tip[members[0]]) for k in members[1:])
            for f in clip.card(label, ['慢放 %d×：机头侧视、斜视、俯视' % a.slow,
                                       '第二块比头一块晚过边 %s；停带命令在头一块过边后 %.2f s' % (
                                           gap, (rel['t_stop_s'] or 0.) - tip[members[0]])], 2.5):
                writer.append_data(np.asarray(f))
            return
        clip, t = state['clip'], line.t
        if clip.a[0] <= t <= clip.a[1]:
            if round(t * 100) % every == 0:             # a frame every `every` 10 ms samples: --slow times slower
                writer.append_data(np.asarray(clip.release_frame(t)))
                state['n1'] = state.get('n1', 0) + 1
            else:
                clip._watch(t)
        elif clip.b and not a.no_station and clip.b[0] <= t <= clip.b[1]:
            if not state.get('card2'):
                state['card2'] = True
                for f in clip.card('这几块到计量带', ['实时（1×）'], 1.5):
                    writer.append_data(np.asarray(f))
            if hold is not None and t >= hold:          # taken off at this sample: freeze the last picture
                if not state.get('frozen') and state.get('last') is not None:
                    state['frozen'] = True
                    for f in clip.freeze(state['last'], 3.):
                        writer.append_data(np.asarray(f))
            elif round(t * 100) % (100 // FPS) == 0:
                state['last'] = clip.station_frame(t)
                writer.append_data(np.asarray(state['last']))
                state['n2'] = state.get('n2', 0) + 1

    line, r2 = run(cfg, sample, stop_after=clip_end if clip_end else max(tip[k] for k in members) + 2.)
    writer.close()

    # the second run is the first one (until it was stopped early)
    rel2 = next(x for x in r2['feeder']['releases'] if x['release'] == rel['release'])
    tip2 = {x['lump']: x['t_tip'] for x in r2['transfer']['lumps']}
    same = rel2['members'] == rel['members'] and all(tip2[k] == tip[k] for k in members) \
        and rel2['t_stop_s'] == rel['t_stop_s']
    if not same:
        raise SystemExit('the rendered run differs from the first one: %s vs %s' % (rel2, rel))
    clip = state['clip']
    info = dict(argv=rest, release=rel['release'], members=members, crossing_s={k: tip[k] for k in members},
                stop_s=rel['t_stop_s'], stop=rel['stop'], after_stop=rel['after_stop'], events=sorted(clip.events),
                items=[{k: it.get(k) for k in ('item', 'truth_lumps', 'void', 'reasons', 't_in_s', 't_hold_s',
                                                'route')} for it in items],
                clip_release_s=[round(x, 3) for x in clip.a],
                clip_station_s=None if clip.b is None else [round(x, 3) for x in clip.b],
                frames_release=state.get('n1', 0),
                frames_station=state.get('n2', 0), slow=a.slow, fps=FPS, same_as_first_run=same)
    (out / 'clip.json').write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
    cfg_out = dict(r1['config'], out_dir=str(out))
    (out / 'result.json').write_text(json.dumps(dict(r1, config=cfg_out), indent=1, ensure_ascii=False),
                                     encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False))


if __name__ == '__main__':
    main()
