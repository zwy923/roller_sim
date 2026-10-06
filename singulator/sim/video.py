"""Video: a camera following the batch (top) over an orthographic plan view (bottom), with a text overlay."""
from pathlib import Path

import mujoco
import numpy as np

# the overlay's Chinese font, the first that exists (Windows, then Linux); without one the default font is used
FONTS = (Path(r'C:\Windows\Fonts\msyh.ttc'), Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
         Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
FONT = next((p for p in FONTS if p.exists()), FONTS[0])

FACE_PHASE_ZH = dict(idle='常位', out='撤离中', hold='撤离保持', back='复位中', fault='犁面运动故障')
STAGE_ZH = dict(buffer='缓冲', transfer='上计量带', onm='在计量带上', measure='称重+测体积', decided='等排料板',
                discharge='排出', hold='作废停住')
PLATE_ZH = dict(closed='落板（煤）', opening='抬起中', open='抬起（矸）', closing='回落中')
REASON_ZH = dict(multi='多块', handover='交接未确认', outside='搭秤外', unsteady='读数不稳', scan='扫描无效',
                 implausible='密度不合理', tare='秤未清空', track_lost='跟踪丢失')


class Recorder:
    def __init__(self, cfg, d, model):
        from PIL import ImageFont
        import imageio.v2 as imageio
        cfg['out_dir'].mkdir(parents=True, exist_ok=True)
        self.cfg, self.d = cfg, d
        self.iso = mujoco.Renderer(model, cfg['height'] - cfg['top_h'], cfg['width'])
        self.top = mujoco.Renderer(model, cfg['top_h'], cfg['width'])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 4.2, 58., -40.
        self.cam.lookat[:] = (d['P0'][0], cfg['belt_w'] / 2, 0.)
        self.follow_x1 = d['station']['separator']['pivot'][0]
        self.follow_drop = max(self.follow_x1 - d['exit_x'], 1e-9)
        self.top_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'top')
        self.top_pos, self.top_extent = model.cam_pos[self.top_cam].copy(), float(model.cam_fovy[self.top_cam])
        self.iso_h = cfg['height'] - cfg['top_h']
        self.font = ImageFont.truetype(str(FONT), 24) if FONT.exists() else ImageFont.load_default()
        self.small = ImageFont.truetype(str(FONT), 18) if FONT.exists() else self.font
        self.writer = imageio.get_writer(str(cfg['out_dir'] / 'video.mp4'), fps=cfg['fps'] * cfg['video_speed'],
                                         codec='libx264', quality=7, macro_block_size=2)
        self.next_frame, self.every = 0., 1. / cfg['fps']

    def due(self, t):
        return t >= self.next_frame

    def frame(self, data, fronts, lines, labels=()):
        """fronts: front-edge x of every lump (the camera follows their median); lines: (text, big, rgb); labels:
        (text, x, y) written at world (x, y) in the plan view."""
        from PIL import Image, ImageDraw
        self.next_frame += self.every
        d, cfg = self.d, self.cfg
        x = min(max(float(np.median(fronts)), d['P0'][0] - .8), self.follow_x1)
        # past the lane the camera also looks down toward the separator
        self.cam.lookat[:] = (x, cfg['belt_w'] / 2, -.4 * min(1., max(0., (x - d['exit_x']) / self.follow_drop)))
        self.iso.update_scene(data, self.cam)
        self.top.update_scene(data, self.top_cam)
        img = Image.fromarray(np.vstack([self.iso.render(), self.top.render()]))
        dr = ImageDraw.Draw(img)
        for i, (text, big, rgb) in enumerate(lines):
            dr.text((16, 12 + 32 * i), text, font=self.font if big else self.small, fill=rgb)
        for text, x, y in labels:                   # the plan view is orthographic: world (x, y) -> pixels
            u, w = self.plan_px(x, y)
            box = dr.textbbox((u, w), text, font=self.small, anchor='mm')
            dr.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(255, 255, 255))
            dr.text((u, w), text, font=self.small, fill=(10, 10, 10), anchor='mm')
        self.writer.append_data(np.asarray(img))

    def plan_px(self, x, y):
        """Pixel of world (x, y) in the plan view (its camera looks straight down, x to the right, y up)."""
        h = self.cfg['top_h']
        scale = h / self.top_extent
        return (self.cfg['width'] / 2 + (x - self.top_pos[0]) * scale,
                self.iso_h + h / 2 - (y - self.top_pos[1]) * scale)

    def close(self):
        self.writer.close()
        self.iso.close()
        self.top.close()


def overlay(line):
    """The text lines over the video, from the running line (sim.line.Line): (text, big, rgb). Live readings only
    (user, 2026-10-06: 只要实时的必要数据): time, the feed belt's actual speed and what its controller is doing, the drop
    beam S1 as its controller reads it, every lump's centroid against the head edge, then the station."""
    cfg, t, face, feeder, d = line.cfg, line.t, line.face, line.feeder, line.data
    v = max(0., line.belts.feed.f * cfg['feeder_speed'])         # the brake leaves it at -0.0
    state = ('点动' if feeder.jog is not None else '预送' if feeder.staging else '暂停' if feeder.paused
             else '前送' if feeder.phase == 'feeding' and feeder.goal >= 1. else '慢走' if feeder.phase == 'feeding'
             else '停')
    beam = line.sensors.beams['beam_feed'].blocked
    head = 't = %6.2f s    给料带 %.3f m/s %s    S1 %s' % (t, v, state, '挡' if beam else '通')
    if face.phase != 'idle':
        head += '    犁面 %s' % FACE_PHASE_ZH[face.phase] + (' %.0f s' % (t - face.t0) if face.phase == 'hold' else '')
    x1 = line.d['feeder']['x1']
    gone = dict(sorted='落仓', taken_off='移出', dropped='掉落')

    def where(L):
        if L.state in gone:
            return gone[L.state]
        dx = 1000. * (float(d.xipos[L.body][0]) - x1)
        return '%+.0f mm' % dx if abs(dx) < 1000. else '%+.2f m' % (dx / 1000.)
    lines = [(head, True, (15, 20, 30)),
             ('质心−机头边缘  ' + '   '.join('%d: %s' % (L.k, where(L)) for L in line.lumps), False, (30, 40, 55))]
    return lines + station_lines(line.station)


def plan_labels(line):
    """Each lump's number at its centroid, for the plan view: (text, x, y), lumps still on the machine."""
    d = line.data
    return [(str(L.k), float(d.xipos[L.body][0]), float(d.xipos[L.body][1])) for L in line.lumps
            if L.state not in ('sorted', 'taken_off', 'dropped')]


def station_lines(s):
    """The station's states, and the last measured item."""
    where = lambda st: ' '.join('#%d%s' % (it['item'] + 1, STAGE_ZH.get(it['stage'], it['stage']))
                                for it in s.line if it['stage'] in st) or '空'
    flags = ('（停）' if s.b_goal == 0. and not s.frozen else '') + \
            ('  上游暂停' if s.section_held and not s.frozen else '') + \
            ('  全线暂停：%s' % (s.alarm['why'] if s.alarm else '视觉') if s.frozen else '')
    lines = [('缓冲带：%s  计量带：%s  排料板：%s %.0f°%s'
              % (where(('buffer',)), where(('transfer', 'onm', 'measure', 'decided', 'discharge', 'hold')),
                 PLATE_ZH.get(s.plate, s.plate), s.plate_deg, flags), False, (60, 45, 110))]
    done = [it for it in s.items if 't_decided_s' in it]
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
