"""Finite-force lateral gate drives and assumed mount load-cell signals."""
import math
import mujoco
import numpy as np


class GateDrive:
    def __init__(self,cfg,g,model):
        self.g,self.model=g,model
        self.q=[model.jnt_qposadr[model.joint(s['name']+'_j').id] for s in g['sides']]
        self.v=[model.jnt_dofadr[model.joint(s['name']+'_j').id] for s in g['sides']]
        self.a=[model.actuator(s['name']+'_motor').id for s in g['sides']]
        self.bq=[model.jnt_qposadr[model.joint(s['name']+'_buffer').id] for s in g['sides']]
        self.bv=[model.jnt_dofadr[model.joint(s['name']+'_buffer').id] for s in g['sides']]
        self.geom={model.geom(s['name']+'_pad').id:i for i,s in enumerate(g['sides'])}
        self.geom_ids=tuple(self.geom)
        self.ref=np.zeros(2)
        self.loads=np.zeros(2)
        self.peak=self.peak_load=self.peak_contact=self.peak_buffer=self.saturation_s=0.
        self.buf=np.zeros(6)

    def command(self,data,targets,dt):
        step=self.g['speed_m_s']*dt
        for i in range(2):
            self.ref[i] += max(-step,min(step,targets[i]-self.ref[i]))
            data.ctrl[self.a[i]]=self.ref[i]

    def feedback(self,data):
        return dict(position_m=data.qpos[self.q].tolist(),speed_m_s=data.qvel[self.v].tolist(),
                    force_N=data.actuator_force[self.a].tolist(),load_N=self.loads.tolist(),
                    buffer_m=data.qpos[self.bq].tolist())

    def record(self,data,dt):
        k,damping=self.g['buffer_stiffness_N_m'],self.g['buffer_damping_N_s_m']
        for i in range(2):
            self.loads[i]=abs(k*data.qpos[self.bq[i]]+damping*data.qvel[self.bv[i]])
        contact=[0.,0.]
        low,high=self.geom_ids
        pairs=data.contact.geom[:data.ncon]
        hits=((pairs[:,0]==low) | (pairs[:,0]==high) | (pairs[:,1]==low) | (pairs[:,1]==high)).nonzero()[0]
        for j in hits:
            mujoco.mj_contactForce(self.model,data,int(j),self.buf)
            for geom in pairs[j]:
                if geom in self.geom:
                    contact[self.geom[geom]] += math.sqrt(self.buf[0]**2+self.buf[1]**2+self.buf[2]**2)
        peak=max(abs(float(data.actuator_force[self.a[0]])),abs(float(data.actuator_force[self.a[1]])))
        self.peak=max(self.peak,peak)
        self.peak_load=max(self.peak_load,float(self.loads[0]),float(self.loads[1]))
        self.peak_contact=max(self.peak_contact,float(contact[0]),float(contact[1]))
        self.peak_buffer=max(self.peak_buffer,float(data.qpos[self.bq[0]]),float(data.qpos[self.bq[1]]))
        self.saturation_s += dt*(peak>=.99*self.g['force_limit_N'])

    def report(self):
        return dict(peak_force_N=self.peak,peak_mount_load_N=self.peak_load,peak_contact_N=self.peak_contact,
                    peak_buffer_m=self.peak_buffer,saturation_s=self.saturation_s,
                    note='Uncalibrated lateral servo; longitudinal spring-damper buffer with travel limits; '
                         'mount load inferred from spring/damper reaction; contact peaks recorded separately.')
