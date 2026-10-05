"""Video: a camera following the batch (top) over an orthographic plan view (bottom), with a text overlay."""
from pathlib import Path

import mujoco
import numpy as np

FONT = Path(r'C:\Windows\Fonts\msyh.ttc')      # the overlay's Chinese font; without it the default font is used

FACE_PHASE_ZH = dict(idle='常位', out='撤离中', hold='撤离保持', back='复位中', fault='犁面运动故障')
FEEDER_PHASE_ZH = dict(feeding='放料', stopped='停（等放下的料进直道）', empty='已放空')
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
        self.font = ImageFont.truetype(str(FONT), 24) if FONT.exists() else ImageFont.load_default()
        self.small = ImageFont.truetype(str(FONT), 18) if FONT.exists() else self.font
        self.writer = imageio.get_writer(str(cfg['out_dir'] / 'video.mp4'), fps=cfg['fps'] * cfg['video_speed'],
                                         codec='libx264', quality=7, macro_block_size=2)
        self.next_frame, self.every = 0., 1. / cfg['fps']

    def due(self, t):
        return t >= self.next_frame

    def frame(self, data, fronts, lines):
        """fronts: front-edge x of every lump (the camera follows their median); lines: (text, big, rgb)."""
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
        self.writer.append_data(np.asarray(img))

    def close(self):
        self.writer.close()
        self.iso.close()
        self.top.close()


def overlay(cfg, t, n_tail, n_total, together_s, belt_f, face, feeder, n_went, station):
    lines = [('主带 %.2f m/s  缓冲带 %.2f m/s  计量带 %.2f m/s  t=%5.1f s'
              % (cfg['v_belt'], cfg['buffer_speed'], cfg['station_speed'], t), True, (15, 20, 30)),
             ('尾缘过线%d/%d  同时过线%.2f s  主带驱动%.0f%%  犁面：%s'
              % (n_tail, n_total, together_s, 100 * belt_f,
                 FACE_PHASE_ZH[face.phase] + (' %.0f s' % (t - face.t0) if face.phase == 'hold' else '')),
              False, (30, 40, 55))]
    state = ('点动（把挂在边上的料送下去）' if feeder.jog is not None
             else '预送（下一块送到慢走区）' if feeder.staging
             else '停（等前面的料快出漏斗）' if feeder.predict and feeder.phase == 'stopped'
             else FEEDER_PHASE_ZH[feeder.phase]) + (
        '（慢走）' if feeder.phase == 'feeding' and 0. < feeder.goal < 1. and feeder.jog is None else '') + (
        '（缓冲带满，暂停）' if feeder.paused else '')
    lines.append(('给料带：%s  第 %d 次放料  已放 %d/%d 块' % (state, len(feeder.releases), n_went, n_total),
                  False, (30, 90, 70) if feeder.phase == 'feeding' else (90, 90, 100)))
    return lines + station_lines(station)


def station_lines(s):
    """The station's states, and the last measured item."""
    where = lambda st: ' '.join('#%d%s' % (it['item'] + 1, STAGE_ZH.get(it['stage'], it['stage']))
                                for it in s.line if it['stage'] in st) or '空'
    flags = ('（停，等计量带）' if s.b_goal == 0. and not s.frozen else '') + \
            ('  （上游暂停）' if s.section_held and not s.frozen else '') + \
            ('  【全线暂停：%s】' % (s.alarm['why'] if s.alarm else '视觉') if s.frozen else '')
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
