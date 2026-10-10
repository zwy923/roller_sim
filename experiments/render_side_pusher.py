"""Render an actual saved side-gate trajectory, including both plates. No simulation rerun.

python experiments/render_side_pusher.py RUN_DIR --start 2 --end 12 --out example.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from singulator.machine import plough, separator
from singulator.physics.drives import Conveyors
from singulator.sim.video import FONT
from singulator.machine import derive


def render(folder, out, start=0., end=20., fps=15):
    r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    cfg=r['config']
    d=derive(cfg)
    model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
    data=mujoco.MjData(model)
    Conveyors(cfg,d,model).command(data)
    tr=np.load(folder/'trajectory.npz')
    feed_index=list(tr['drive_names']).index('feeder')
    renderer=mujoco.Renderer(model,720,1280)
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[d['feeder']['x1']-.55,cfg['belt_w']/2,.15]
    camera.distance=3.1
    camera.azimuth=90
    camera.elevation=-75
    font=ImageFont.truetype(str(FONT),25)
    small=ImageFont.truetype(str(FONT),19)
    q=[model.jnt_qposadr[model.joint(f'bj{k}').id] for k in range(cfg['count'])]
    fq=model.jnt_qposadr[model.joint(plough.FACE_JOINT).id]
    pq=[model.jnt_qposadr[model.joint(n+'_j').id] for n in ('push_low','push_high')]
    bq=[model.jnt_qposadr[model.joint(n+'_buffer').id] for n in ('push_low','push_high')]
    out.parent.mkdir(parents=True,exist_ok=True)
    end=min(end,float(tr['t'][-1]))
    with imageio.get_writer(str(out),fps=fps,codec='libx264',quality=7,macro_block_size=2) as writer:
        for t in np.arange(start,end,1/fps):
            i=min(np.searchsorted(tr['t'],t),len(tr['t'])-1)
            for k,adr in enumerate(q):
                data.qpos[adr:adr+3]=tr['pos'][i,k]
                data.qpos[adr+3:adr+7]=tr['quat'][i,k]
            data.qpos[fq]=np.radians(tr['face_deg'][i])
            data.qpos[pq]=tr['gate_m'][i]
            data.qpos[bq]=tr['gate_buffer_m'][i]
            separator.place(model,data,d['station']['sep'],float(tr['sep_deg'][i]))
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera)
            img=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(img)
            draw.rectangle((0,0,1280,108),fill=(246,247,249))
            mode='挡板'+('启用' if cfg['side_pusher']=='active' else '停用对照')
            draw.text((22,12),f'{mode} | {cfg["layout"]} | {cfg["count"]}块 | seed {cfg["seed"]} | t={t:.2f}s',font=font,fill=(20,30,40))
            a=tr['gate_m'][i]
            actuator=f'左右伸入 {a[0]*1000:.0f} / {a[1]*1000:.0f} mm'
            draw.text((22,51),f'{actuator}  |  给料带速度 {tr["drive_f"][i,feed_index]*cfg["feeder_speed"]:.3f} m/s',font=small,fill=(50,65,80))
            draw.text((22,79),'真实保存轨迹回放；未标定仿真，不代表实物效果。',font=small,fill=(75,80,90))
            writer.append_data(np.asarray(img))
            if abs(t-(start+end)/2)<1/fps:
                img.save(out.with_suffix('.png'))
    renderer.close()
    print(out.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--start',type=float,default=0.)
    p.add_argument('--end',type=float,default=20.)
    a=p.parse_args()
    render(a.folder,a.out,a.start,a.end)
