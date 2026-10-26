"""Fixed foot trajectory with learned joint residuals, inspired by PMTG.

This is a controller prior, never evidence of learning by itself. Geometry comes
only from the current asset. No new joints, forces, masses or collisions.
"""
import argparse,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

PAIRS=[['L_shoulder_pitch','L_elbow_pitch'],['R_shoulder_pitch','R_elbow_pitch'],['L_hip_pitch','L_knee_pitch'],['R_hip_pitch','R_knee_pitch']]
FEET=['L_wrist_fixed','R_wrist_fixed','L_ankle_fixed','R_ankle_fixed']
def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
def rotation(axis,q):
    k=skew(axis/np.linalg.norm(axis));return np.eye(3)+math.sin(q)*k+(1-math.cos(q))*k@k
def quaternion(q):
    x,y,z,w=q;k=skew([x,y,z]);return np.eye(3)+2*(w*k+k@k)

class CapsuleKinematics:
    def __init__(self,asset,manifest):
        self.model=manifest;self.root=ET.parse(asset).find('worldbody/body');self.q0=dict(manifest['default_angles'])
        assert manifest['foot_bodies']==FEET
        self.limits={j.get('name'):np.fromstring(j.get('range'),sep=' ') for j in self.root.iter('joint') if j.get('range')}
        self.capsules=[]
        for name in FEET:
            body=next(b for b in self.root.iter('body') if b.get('name')==name)
            geom=next(g for g in body.findall('geom') if g.get('type')=='capsule')
            self.capsules.append((name,np.fromstring(geom.get('fromto'),sep=' ').reshape(2,3),float(geom.get('size'))))
        self.base=self.fk(self.q0)
    def fk(self,q):
        poses={}
        def visit(body,position,r):
            position=position+r@np.fromstring(body.get('pos','0 0 0'),sep=' ')
            if body.get('quat'):
                w,x,y,z=np.fromstring(body.get('quat'),sep=' ');r=r@quaternion([x,y,z,w])
            for j in body.findall('joint'):
                if j.get('name') not in q:continue
                pivot=np.fromstring(j.get('pos','0 0 0'),sep=' ');rr=rotation(np.fromstring(j.get('axis'),sep=' '),q[j.get('name')])
                position=position+r@(pivot-rr@pivot);r=r@rr
            poses[body.get('name')]=(position,r)
            for child in body.findall('body'):visit(child,position.copy(),r.copy())
        visit(self.root,np.array(self.model['init_position']),quaternion(self.model['init_quat_xyzw']))
        out=[]
        for name,points,radius in self.capsules:
            position,r=poses[name];world=position+points@r.T;point=world[np.argmin(world[:,2])].copy();point[2]-=radius;out.append(point)
        return np.asarray(out)
    def solve(self,foot,x,z,previous=None):
        q=dict(self.q0);pair=PAIRS[foot]
        if previous is not None:
            for name,value in zip(pair,previous):q[name]=self.q0[name]+value
        target=self.base[foot,[0,2]]+np.array([x,z])
        for _ in range(16):
            error=target-self.fk(q)[foot,[0,2]]
            if np.linalg.norm(error)<1e-8:break
            jac=[]
            for name in pair:
                plus=dict(q);minus=dict(q);plus[name]+=1e-5;minus[name]-=1e-5
                jac.append((self.fk(plus)[foot,[0,2]]-self.fk(minus)[foot,[0,2]])/2e-5)
            delta=np.linalg.solve(np.asarray(jac).T,error)
            for name,d in zip(pair,delta):q[name]=float(np.clip(q[name]+d,*self.limits[name]))
        point=self.fk(q)[foot];error=float(np.linalg.norm(point[[0,2]]-target))
        if error>2e-6:raise ValueError('Unreachable foot target: '+str((foot,x,z,error)))
        margin=min(min(q[n]-self.limits[n][0],self.limits[n][1]-q[n]) for n in pair)
        return np.array([q[n]-self.q0[n] for n in pair]),error,float(margin),float(point[1]-self.base[foot,1])

def build(asset,manifest,output):
    from run_io import atomic_json,sha256
    if output.exists():raise FileExistsError('Refusing to overwrite controller table: '+str(output))
    model=json.loads(manifest.read_text());kin=CapsuleKinematics(asset,model)
    speeds=[0.,.03,.04,.05,.06,.075,.09];period=.8;stance=.75;height=.018;bins=128
    heights=[.018,.018,.023,.023]
    table=np.zeros((len(speeds),bins+1,4,2));max_error=0.;min_margin=math.pi;max_lateral=0.
    for si,speed in enumerate(speeds[1:],1):
        previous=[None]*4;stride=speed*period*stance
        for pi in range(bins+1):
            phase=pi/bins
            if phase<stance:x=stride*(.5-phase/stance);z=0.
            else:
                u=(phase-stance)/(1-stance);x=stride*(-.5+u-math.sin(2*math.pi*u)/(2*math.pi));z=height*math.sin(math.pi*u)**2
            for foot in range(4):
                delta,error,margin,lateral=kin.solve(foot,x*(.75 if foot>=2 else 1.),z*heights[foot]/height,previous[foot]);table[si,pi,foot]=delta;previous[foot]=delta
                max_error=max(max_error,error);min_margin=min(min_margin,margin);max_lateral=max(max_lateral,abs(lateral))
        print('REFERENCE_TABLE_SPEED',speed,flush=True)
    assert np.max(np.abs(table[:,0]-table[:,-1]))<1e-5 and min_margin>0
    atomic_json(output,{'schema':'sable-fixed-foot-reference/1','source_asset_sha256':sha256(asset),
        'source_manifest_sha256':sha256(manifest),'builder_sha256':sha256(__file__),
        'method_reference':'https://proceedings.mlr.press/v87/iscen18a.html',
        'scope':'Fixed kinematic controller prior plus future learned joint residuals; not a learned gait or hardware validation.',
        'speeds_m_s':speeds,'phase_bins':bins,'period_s':period,'stance_fraction':stance,'height_m':height,'height_profile':'sine_squared',
        'height_m_per_foot':heights,'horizontal_table_scale_per_foot':[1.,1.,.75,.75],
        'feet':FEET,'joint_pairs':PAIRS,'deltas_rad':table.tolist(),'max_ik_error_m':max_error,
        'minimum_limit_margin_deg':math.degrees(min_margin),'max_lateral_deviation_m':max_lateral})
    print('REFERENCE_TABLE_COMPLETE',str(output),max_error,math.degrees(min_margin),max_lateral,flush=True)

class GuidedReference:
    def __init__(self,env):
        import torch
        from run_io import read_json,sha256
        self.torch=torch;self.env=env;cfg=env.cfg.reference;self.data=read_json(cfg.file)
        assert self.data['source_asset_sha256']==sha256(env.cfg.asset.file)
        assert self.data['source_manifest_sha256']==sha256(Path(env.cfg.asset.file).with_name('physical_manifest.json'))
        assert self.data['period_s']==env.cfg.gait.period_s and self.data['height_profile']==env.cfg.gait.height_profile
        assert self.data['height_m']==env.cfg.gait.swing_height_m and self.data['stance_fraction']==env.cfg.gait.stance_fraction
        assert self.data['height_m_per_foot']==env.cfg.gait.swing_heights_m
        self.table=torch.tensor(self.data['deltas_rad'],device=env.device);self.grid=torch.tensor(self.data['speeds_m_s'],device=env.device)
        self.speed=torch.zeros(env.num_envs,device=env.device);self.bins=self.data['phase_bins']
        self.dofs=torch.tensor([[env.dof_names.index(n) for n in pair] for pair in PAIRS],device=env.device)
        self.feet=torch.arange(4,device=env.device);self.tick=0
        self.amplitude_multipliers=torch.tensor(cfg.amplitude_multipliers,device=env.device)
        self.start=torch.zeros(env.num_envs,env.num_dof,device=env.device);self.end=torch.zeros_like(self.start)
    def ramp(self,age):
        x=(age/self.env.cfg.reference.ramp_s).clamp(0,1);return x*x*(3-2*x)*self.env.cfg.reference.strength
    def lookup(self,speed,phase,age):
        t=self.torch;speed=speed.clamp(float(self.grid[0]),float(self.grid[-1]))
        hi=t.searchsorted(self.grid,speed.contiguous()).clamp(1,len(self.grid)-1);lo=hi-1
        alpha=((speed-self.grid[lo])/(self.grid[hi]-self.grid[lo]))[:,None,None]
        local=t.remainder(phase[:,None]+self.env.gait_signals.offsets,1.)*self.bins
        p0=local.long().clamp(0,self.bins-1);beta=(local-p0)[:,:,None]
        low=self.table[lo[:,None],p0,self.feet]+beta*(self.table[lo[:,None],p0+1,self.feet]-self.table[lo[:,None],p0,self.feet])
        high=self.table[hi[:,None],p0,self.feet]+beta*(self.table[hi[:,None],p0+1,self.feet]-self.table[hi[:,None],p0,self.feet])
        out=t.zeros(self.env.num_envs,self.env.num_dof,device=self.env.device)
        out[:,self.dofs.flatten()]=((low+alpha*(high-low))*self.ramp(age)[:,None,None]*self.amplitude_multipliers[None,:,None]).reshape(self.env.num_envs,8)
        return out
    def prepare(self):
        env=self.env;age=env.episode_length_buf*env.dt;phase=env.gait_signals.phase(env)
        wanted=env.commands[:,0].clamp(0,float(self.grid[-1]));new_speed=self.speed+(1-math.exp(-env.dt/env.cfg.reference.speed_filter_tau_s))*(wanted-self.speed)
        self.start=self.lookup(self.speed,phase,age);self.end=self.lookup(new_speed,phase+env.dt/env.cfg.gait.period_s,age+env.dt)
        self.speed.copy_(new_speed);self.tick=0
    def target(self):
        self.tick+=1;alpha=min(1.,self.tick/self.env.cfg.control.decimation)
        return self.start+alpha*(self.end-self.start)
    def reset(self,ids):
        self.speed[ids]=0.;self.start[ids]=0.;self.end[ids]=0.

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();build(args.asset,args.manifest,args.output)
