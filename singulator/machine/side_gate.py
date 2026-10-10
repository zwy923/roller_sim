"""Transverse plates retract sideways; deployed plates retain one side of the leading row.

The plate is upstream of the feed head. Real slots replace the skirts in its swept volume.
This is an uncalibrated mechanism proposal, not a purchased actuator specification.
"""
from .parts import skirt

THICKNESS = .024
BOTTOM_GAP = .005
SIDE_CLEARANCE = .025
SLOT_CLEARANCE = .012
MASS = 12.
KP = 30000.
KV = 900.
BUFFER_STROKE = .030
BUFFER_K = 40000.
BUFFER_DAMPING = 400.


def geometry(cfg, d):
    stroke = cfg['gate_stroke']         # its bounds are checked with the other parameters (config.validate)
    return dict(kind='gate', x_m=d['feeder']['x1']+.025, length_m=stroke+.05,
                height_m=cfg['gate_height'], thickness_m=THICKNESS, stroke_m=stroke,
                bottom_z_m=d['feeder']['step_m']+BOTTOM_GAP, speed_m_s=cfg['gate_speed'],
                force_limit_N=cfg['gate_force_max'], load_trip_N=cfg['gate_load_trip'],
                buffer_stroke_m=BUFFER_STROKE, buffer_stiffness_N_m=BUFFER_K, buffer_damping_N_s_m=BUFFER_DAMPING,
                sides=[dict(name='push_low', sign=1., tip_y=cfg['lane_y']-SIDE_CLEARANCE),
                       dict(name='push_high', sign=-1., tip_y=cfg['belt_w']+SIDE_CLEARANCE)])


def hardware(cfg, d):
    g = geometry(cfg,d)
    bodies, actuators = [], []
    L,H,T = g['length_m'],g['height_m'],g['thickness_m']
    for s in g['sides']:
        n,sign = s['name'],s['sign']
        bodies.append(
            f'<body name="{n}" pos="{g["x_m"]} {s["tip_y"]} {g["bottom_z_m"]}">'
            f'<joint name="{n}_j" type="slide" axis="0 {sign} 0" limited="true" range="0 {g["stroke_m"]}" damping="5"/>'
            f'<joint name="{n}_buffer" type="slide" axis="1 0 0" limited="true" range="0 {BUFFER_STROKE}" '
            f'stiffness="{BUFFER_K}" damping="{BUFFER_DAMPING}"/>'
            f'<geom name="{n}_pad" class="wall" type="box" mass="{MASS}" pos="0 {-sign*L/2} {H/2}" '
            f'size="{T/2} {L/2} {H/2}" rgba=".95 .48 .08 1"/></body>')
        actuators.append(f'<position name="{n}_motor" joint="{n}_j" kp="{KP}" kv="{KV}" '
                         f'forcelimited="true" forcerange="{-g["force_limit_N"]} {g["force_limit_N"]}" '
                         f'ctrllimited="true" ctrlrange="0 {g["stroke_m"]}"/>')
    return bodies,''.join(actuators)


def skirts(cfg,d):
    g = geometry(cfg,d)
    a,b = g['x_m']-THICKNESS/2-SLOT_CLEARANCE,g['x_m']+THICKNESS/2+SLOT_CLEARANCE+BUFFER_STROKE
    out=[]
    for y,end,stem in ((cfg['lane_y']-.01,d['sb_x0'] if cfg['side_belt'] else d['belt_x1'],'skirt_in'),
                       (cfg['belt_w']+.01,d['skirt_out_end'],'skirt_out')):
        out += [skirt(stem+'_rear',y,d['belt_x0'],a,.004),skirt(stem+'_front',y,b,end,.004)]
    return out
