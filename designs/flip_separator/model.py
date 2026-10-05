"""Standalone hinged coal/gangue diverter: renders, interactive viewer, checks.json. SI units.

The geometry (Config, build, state, verify, material_clearance) lives in singulator/separator.py, shared
with the line (singulator/station.py), which embeds the same parts behind the measuring belt.
The preview prescribes consistent joint positions (kinematics, not a load test).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from singulator.machine.separator import Config, build, material_clearance, pose, state, verify  # noqa: E402,F401


def render(model, data, c, out):
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    renderer = mujoco.Renderer(model, height=800, width=1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.72, 0, 1.10]
    cam.distance, cam.azimuth, cam.elevation = 4.9, 130, -23
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 26)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)

    def frame(deg, title):
        s = pose(model, data, c, deg)
        renderer.update_scene(data, camera=cam)
        im = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 1280, 116), fill=(246, 248, 251))
        d.text((24, 12), title, font=font, fill=(25, 35, 48))
        d.text((24, 52), f'板宽 {c.width:.2f} m｜暂定板长 {c.length:.2f} m｜倾角 {deg:.1f}°｜缸伸出 {s["piston_extension_m"]*1000:.0f} mm', font=small, fill=(45, 55, 70))
        d.text((24, 82), '空载运动学示意；来料从蓝色参考面进入；尺寸、筋条、转轴及油缸参数待确认', font=small, fill=(110, 60, 35))
        return im

    try:
        frame(c.closed_deg, '落板位：煤沿板面通过').save(out/'closed.png')
        frame(c.open_deg, '抬板位：入口端抬起，矸石从缺口落下').save(out/'open.png')
        cam.azimuth, cam.elevation = 90, -3
        frame(c.open_deg, '侧视：抬板后的落料窗口').save(out/'side_open.png')
        cam.azimuth, cam.elevation = 130, -23
        with imageio.get_writer(out/'motion.mp4', fps=24, codec='libx264', quality=8) as writer:
            for t in np.arange(0, 8, 1/24):
                if t < 1.5:
                    u, title = 0., '落板位：煤沿板面通过'
                elif t < 3.5:
                    u, title = (t-1.5)/2, '收到外部矸石信号：抬板'
                elif t < 5:
                    u, title = 1., '抬板保持：为矸石打开落料缺口'
                elif t < 7:
                    u, title = 1-(t-5)/2, '示意复位：实际应等待落料区域清空'
                else:
                    u, title = 0., '回到落板位'
                smooth = u*u*(3-2*u)
                writer.append_data(np.asarray(frame(c.closed_deg+smooth*(c.open_deg-c.closed_deg), title)))
    finally:
        renderer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--length', type=float, default=1.60)
    ap.add_argument('--width', type=float, default=1.20)
    ap.add_argument('--closed-deg', type=float, default=20.)
    ap.add_argument('--open-deg', type=float, default=75.)
    ap.add_argument('--pivot-fraction', type=float, default=.72)
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--view', action='store_true', help='Interactive kinematic viewer; SPACE toggles plate')
    ap.add_argument('--out', type=Path, default=ROOT/'output'/'flip_separator')
    args = ap.parse_args()
    c = Config(length=args.length, width=args.width, closed_deg=args.closed_deg,
               open_deg=args.open_deg, pivot_fraction=args.pivot_fraction)
    xml = build(c)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    checks = verify(model, data, c)
    checks['material_path_clearance'] = material_clearance(xml, c)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'model.xml').write_text(xml, encoding='utf-8')
    (args.out/'config.json').write_text(json.dumps(dict(parameters=asdict(c),
        confirmed=['standalone width 1.20 m (the line uses 0.70 m)', 'closed: coal passes; open: gangue drops',
                   'side walls riding on the plate (2026-09-30)'],
        assumed=['all other dimensions and angles', 'fixed longitudinal ribs', 'pivot near downstream end'],
        coordinate_system='x downstream, y across width, z up; positive y-axis rotation lifts inlet',
        closed=state(c, c.closed_deg), opened=state(c, c.open_deg)), ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out/'checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    if not args.no_video:
        render(model, data, c, args.out)
    print(json.dumps(dict(output=str(args.out), **checks), indent=2))
    if args.view:
        import time
        from mujoco import viewer as mj_viewer
        toggle = [False]
        def keypress(key):
            if key == 32:
                toggle[0] = not toggle[0]
        angle = c.closed_deg
        with mj_viewer.launch_passive(model, data, key_callback=keypress) as viewer:
            viewer.cam.lookat[:] = [.72, 0, 1.10]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.9, 130, -23
            while viewer.is_running():
                goal = c.open_deg if toggle[0] else c.closed_deg
                angle += float(np.clip(goal-angle, -27.5/60, 27.5/60))
                with viewer.lock():
                    pose(model, data, c, angle)
                viewer.sync()
                time.sleep(1/60)


if __name__ == '__main__':
    main()
