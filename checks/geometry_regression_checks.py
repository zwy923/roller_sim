"""Regression checks for the belt support and equipment clearance (2026-09-23).

Checks compiled support surfaces, real contact-driven transport and load, and a deliberate side-belt collision.
These are model assembly checks, not evidence of real-machine performance.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from singulator.machine import assembly, derive, plough  # noqa: E402
from singulator.physics import drives  # noqa: E402
from singulator.config import parse_config  # noqa: E402


def model_for(cfg, blocks=()):
    d = derive(cfg)
    m = mujoco.MjModel.from_xml_string(assembly.build_xml(cfg, d, blocks))
    return d, m, mujoco.MjData(m)


def support_height(m, data, x, y):
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, data, np.array([x, y, .2]), np.array([0., 0., -1.]),
                        None, 1, -1, gid)
    return .2 - dist, int(gid[0])


def main():
    cfg = parse_config(['--no-video'])
    d, m, data = model_for(cfg)
    conveyor = drives.Conveyors(cfg, d, m)
    conveyor.main.motor['f'] = 1.
    conveyor.command(data)
    mujoco.mj_forward(m, data)
    floor = m.body_geomadr[m.body(plough.MAIN_BELT).id]
    for x in (d['P0'][0] - .3, d['P1'][0], d['P1'][0] + .5 * cfg['lane_len'], d['belt_x1'] - .02):
        height, gid = support_height(m, data, x, cfg['lane_y'] + cfg['lane_w'] / 2)
        assert gid == floor and abs(height) < 1e-8, (x, height, gid)
    assert assembly.categorise(m)[floor] == assembly.CATS.index('belt')
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, m.body(plough.MAIN_BELT).id, velocity, 0)
    assert np.allclose(velocity[3:], [cfg['v_belt'], 0., 0.])
    print('PASS the main belt supports the funnel and the whole lane at z 0 and runs at belt speed; belt contact tagging')

    # A physical cube in the lane must stay supported, move forward, and put resisting contact load on the main
    # motor. This catches a plate added with no drive, or on an unrelated joint.
    vertices = np.array([[x, y, z] for x in (-.05, .05) for y in (-.05, .05) for z in (-.05, .05)])
    cube = dict(vertices=vertices, material='coal', density=1300.)
    d, m, data = model_for(cfg, [cube])
    conveyor = drives.Conveyors(cfg, d, m)
    conveyor.main.motor['f'] = 1.
    q = m.jnt_qposadr[m.joint('bj0').id]
    x0 = d['P1'][0] + .4
    data.qpos[q:q + 7] = [x0, cfg['lane_y'] + cfg['lane_w'] / 2, .06, 1., 0., 0., 0.]
    min_load = 0.
    for _ in range(round(.3 / cfg['dt'])):
        conveyor.command(data)
        mujoco.mj_step(m, data)
        min_load = min(min_load, conveyor.main.load(data))
    assert data.qpos[q + 2] > .045, data.qpos[q:q + 3]
    assert data.qpos[q] > x0 + .06, data.qpos[q:q + 3]
    assert min_load < -1., min_load
    print('PASS cube is supported and transported in the lane; contact resistance reaches the main motor')

    result = assembly.motion_clearance(cfg, derive(cfg))
    assert all(v['min_distance_m'] >= -.0002 and not v['unresolved'] for v in result.values()), result
    print('PASS the line clears every moving part over its stroke')

    dd = derive(cfg)
    # Deliberately move the side belt into the face without moving the face. Before the fix every
    # vslat was left at x=0, so the clearance sweep missed this downstream equipment interference.
    shifted = dict(cfg, lane_y=dd['lane_top'] + .03)
    clash = assembly.motion_clearance(shifted, dd, steps=3)['face']
    assert clash['min_distance_m'] < -.01 and any('vslat' in s for s in clash['pair']), clash
    print('PASS clearance sweep detects the deliberate face/side-belt overlap away from x=0')


if __name__ == '__main__':
    main()
