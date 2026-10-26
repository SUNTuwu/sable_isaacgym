"""Remote static IK plus short manual-motion simulation; never a learned-policy result."""
import argparse,json,math,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
runtime=a.run.resolve()/'runtime';sys.path.insert(0,str(runtime))
from isaacgym import gymapi
import torch
from preflight import create_env
from sable_config import SableCfg,MODEL
from run_io import atomic_json,sha256

tree=ET.parse(runtime/'assets/sable.xml');root=tree.find('worldbody/body')
feet=MODEL['foot_bodies'];pairs=[['L_shoulder_pitch','L_elbow_pitch'],['R_shoulder_pitch','R_elbow_pitch'],['L_hip_pitch','L_knee_pitch'],['R_hip_pitch','R_knee_pitch']]
assert feet==['L_wrist_fixed','R_wrist_fixed','L_ankle_fixed','R_ankle_fixed']
limits={j.get('name'):np.fromstring(j.get('range'),sep=' ') for j in tree.iter('joint') if j.get('range')}
def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
def rotation(axis,q):
    axis=axis/np.linalg.norm(axis);k=skew(axis);return np.eye(3)+math.sin(q)*k+(1-math.cos(q))*k@k
def quaternion(q):
    x,y,z,w=q;k=skew([x,y,z]);return np.eye(3)+2*(w*k+k@k)
def fk(q):
    poses={}
    def visit(body,position,r):
        position=position+r@np.fromstring(body.get('pos','0 0 0'),sep=' ')
        if body.get('quat'):
            w,x,y,z=np.fromstring(body.get('quat'),sep=' ');r=r@quaternion([x,y,z,w])
        for j in body.findall('joint'):
            if j.get('name') not in q:continue
            axis=np.fromstring(j.get('axis'),sep=' ');pivot=np.fromstring(j.get('pos','0 0 0'),sep=' ')
            rr=rotation(axis,q[j.get('name')]);position=position+r@(pivot-rr@pivot);r=r@rr
        poses[body.get('name')]=(position,r)
        for child in body.findall('body'):visit(child,position.copy(),r.copy())
    visit(root,np.array(MODEL['init_position']),quaternion(MODEL['init_quat_xyzw']))
    out=[]
    for name in feet:
        body=next(b for b in root.iter('body') if b.get('name')==name)
        geom=next(g for g in body.findall('geom') if g.get('type')=='capsule')
        pts=np.fromstring(geom.get('fromto'),sep=' ').reshape(2,3);position,r=poses[name]
        world=position+pts@r.T;point=world[np.argmin(world[:,2])].copy();point[2]-=float(geom.get('size'));out.append(point)
    return np.asarray(out)
q0=dict(MODEL['default_angles']);base=fk(q0);solutions=[]
for foot,pair in enumerate(pairs):
    q=dict(q0);target=base[foot,[0,2]]+np.array([0.,.006])
    for _ in range(12):
        point=fk(q)[foot,[0,2]];err=target-point
        if np.linalg.norm(err)<1e-8:break
        jac=[]
        for name in pair:
            plus=dict(q);minus=dict(q);plus[name]+=1e-5;minus[name]-=1e-5
            jac.append((fk(plus)[foot,[0,2]]-fk(minus)[foot,[0,2]])/2e-5)
        delta=np.linalg.solve(np.asarray(jac).T,err)
        for name,d in zip(pair,delta):q[name]=float(np.clip(q[name]+d,*limits[name]))
    final=fk(q)[foot];error=float(np.linalg.norm(final[[0,2]]-target))
    assert error<1e-6 and all(abs(q[n]-q0[n])<.5 for n in pair),(foot,error)
    solutions.append({'foot':feet[foot],'joints':pair,'delta_rad':[q[n]-q0[n] for n in pair],
        'delta_deg':[math.degrees(q[n]-q0[n]) for n in pair],'achieved_xyz_delta_m':(final-base[foot]).tolist(),
        'xz_error_m':error,'remaining_limit_margin_deg':[math.degrees(min(q[n]-limits[n][0],limits[n][1]-q[n])) for n in pair]})
print('STATIC_IK',json.dumps(solutions),flush=True)
simulations=[]
for fixed in [True,False]:
    cfg=SableCfg();cfg.env.episode_length_s=8.;env=create_env(4,fixed=fixed,noise=False,cfg=cfg)
    env.reset_noise=0.;env.fixed_command=0.;env.reset();env.capture_eval=True
    indexes=[[MODEL['policy_joint_names'].index(n) for n in pair] for pair in pairs]
    dofs=[[env.dof_names.index(n) for n in pair] for pair in pairs]
    active=np.ones(4,bool);falls=np.zeros(4,bool);peak=np.zeros(4);torque=np.zeros((4,2));speed=np.zeros((4,2));trace=[]
    di=torch.tensor(dofs,device=env.device);bi=torch.arange(4,device=env.device)[:,None]
    active_gpu=torch.ones(4,device=env.device,dtype=torch.bool)
    substep_torque=torch.zeros(4,2,device=env.device);substep_speed=torch.zeros_like(substep_torque)
    utilization=torch.zeros_like(substep_torque);limited_samples=torch.zeros_like(substep_torque);samples=torch.zeros_like(substep_torque)
    original_torque=env._compute_torques
    def measure_torque(actions):
        result=original_torque(actions);tt=result[bi,di];vv=env.dof_vel[bi,di];mask=active_gpu[:,None]
        substep_torque.copy_(torch.where(mask,torch.maximum(substep_torque,tt.abs()),substep_torque))
        substep_speed.copy_(torch.where(mask,torch.maximum(substep_speed,vv.abs()),substep_speed))
        rpm=vv.abs()*60/(2*math.pi);stall=env.torque_limits[di]
        large=torch.where(rpm<=245.1,1.95-(1.95-.78)*rpm/245.1,.78*(421.3-rpm)/(421.3-245.1)).clamp(min=0)
        drive=torch.where(env.large[di],large,stall*(1-vv.abs()/env.max_speed[di]).clamp(min=0))
        cap=torch.where(tt*vv<0,stall,drive);ratio=tt.abs()/cap.clamp(min=1e-6)
        utilization.copy_(torch.where(mask,torch.maximum(utilization,ratio),utilization))
        limited_samples.add_((ratio>.999)&mask);samples.add_(mask)
        return result
    env._compute_torques=measure_torque
    initial_clear=None;start=time.time()
    for k in range(240):
        t=k*env.dt;phase=((t-.4)%.8)/.8 if t>=.4 else 0.
        swing=max(0.,(phase-.75)/.25);weight=math.sin(math.pi*swing) if phase>=.75 and t>=.4 else 0.
        action=torch.zeros(4,12,device=env.device)
        for i,solution in enumerate(solutions):
            action[i,indexes[i]]=torch.tensor(solution['delta_rad'],device=env.device,dtype=action.dtype)*weight/cfg.control.action_scale
        env.step(action);s=env.eval_snapshot;heights=s['clearance'].cpu().numpy()[np.arange(4),np.arange(4)]
        if k==19:initial_clear=heights.copy()
        if k>=20:peak[active]=np.maximum(peak[active],heights[active])
        if k%2==0:trace.append({'t_s':(k+1)*env.dt,'target_weight':weight,'clearance_m':heights.tolist(),'root_z_m':s['xyz'][:,2].cpu().tolist()})
        falls|=active&s['fall'].cpu().numpy();active&=~s['fall'].cpu().numpy()
        active_gpu[:]=torch.tensor(active,device=env.device)
    simulations.append({'fixed_base':fixed,'seconds':240*env.dt,'wall_seconds':time.time()-start,'fell':falls.tolist(),
        'initial_clearance_m':initial_clear.tolist(),'peak_clearance_m':peak.tolist(),
        'joint_peak_torque_nm':substep_torque.cpu().tolist(),'joint_peak_speed_rad_s':substep_speed.cpu().tolist(),
        'torque_speed_limit_peak_fraction':utilization.cpu().tolist(),
        'torque_speed_limited_substep_fraction':(limited_samples/samples.clamp(min=1)).cpu().tolist(),
        'torque_sampling':'every physics substep, before integration; 0.5ms','trace':trace})
    print('PHYSICS_LIFT',json.dumps({k:v for k,v in simulations[-1].items() if k!='trace'}),flush=True);env.gym.destroy_sim(env.sim)
atomic_json(a.output,{'schema':'sable-lift-reachability-probe/1','created_unix':time.time(),'source_sha256':sha256(__file__),
    'asset_sha256':sha256(runtime/'assets/sable.xml'),'model_manifest_sha256':sha256(runtime/'assets/physical_manifest.json'),
    'scope':'Manual joint references derived by static IK; this is feasibility evidence, not learned locomotion or hardware validation.',
    'solutions':solutions,'simulations':simulations})
