"""Video: a camera following the batch (top) over an orthographic plan view (bottom), with a text overlay."""
from pathlib import Path

import mujoco
import numpy as np

FONT = Path(r'C:\Windows\Fonts\msyh.ttc')      # the overlay's Chinese font; without it the default font is used

FACE_PHASE_ZH = dict(idle='常位', out='撤离中', hold='撤离保持', back='复位中', fault='犁面运动故障')
FEEDER_PHASE_ZH = dict(feeding='放料', stopped='停（等放下的料进直道）', empty='已放空')


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


def overlay(cfg, t, n_tail, n_total, together_s, belt_f, face, feeder, station):
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
    went = feeder.lumps_went if feeder.vision else feeder.released
    lines.append(('给料带：%s  第 %d 次放料  已放 %d/%d 块' % (state, len(feeder.releases), len(went), n_total),
                  False, (30, 90, 70) if feeder.phase == 'feeding' else (90, 90, 100)))
    return lines + station.status_lines()
