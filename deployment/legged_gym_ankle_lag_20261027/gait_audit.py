"""Remote inference audit: actual capsule clearance, stepping, slip and heading."""
import argparse,sys,json,time,math,xml.etree.ElementTree as ET
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=16.);p.add_argument('--num-envs',type=int,default=12);p.add_argument('--seed',type=int,default=947);p.add_argument('--zero',action='store_true');a=p.parse_args()
run=a.checkpoint.resolve().parent.parent;runtime=run/'runtime';sys.path.insert(0,str(runtime))
from isaacgym import gymapi
import torch,numpy as np
from rsl_rl.modules import ActorCritic
from preflight import create_env
from sable_config import SableCfg,MODEL
from run_io import atomic_json,read_json,sha256

torch.manual_seed(a.seed);np.random.seed(a.seed)
cfg=SableCfg();cfg.env.episode_length_s=a.seconds;cfg.commands.resampling_time=1000.
run_config=read_json(run/'config.json');cfg.commands.heading_command=run_config['env']['commands']['heading_command']
if run_config['env'].get('gait',{}).get('enabled'):
    for key,value in run_config['env']['gait'].items():setattr(cfg.gait,key,value)
    cfg.env.num_observations=run_config['env']['env']['num_observations']
env=create_env(a.num_envs,noise=False,cfg=cfg);env.reset_noise=.015
actor=ActorCritic(env.num_obs,env.num_obs,12,**run_config['train']['policy']).to('cuda:0')
cp=torch.load(str(a.checkpoint),map_location='cuda:0');actor.load_state_dict(cp['model_state_dict']);actor.eval()
# The native XML capsule endpoints are expressed in their rigid-body frames.
tree=ET.parse(runtime/'assets/sable.xml');endpoints=[];radii=[]
for name in MODEL['foot_bodies']:
    body=next(b for b in tree.iter('body') if b.get('name')==name)
    geoms=[g for g in body.findall('geom') if g.get('type')=='capsule']
    assert len(geoms)==1,(name,len(geoms))
    endpoints.append(np.fromstring(geoms[0].get('fromto'),sep=' ').reshape(2,3));radii.append(float(geoms[0].get('size')))
endpoints=np.asarray(endpoints)[None,:,:,:];radii=np.asarray(radii)[None,:]
original_check=env.check_termination
def capture():
    original_check()
    if env.capture_eval:
        env.eval_snapshot['foot_quat']=env.rigid_states[:,env.feet_indices,3:7].clone()
        env.eval_snapshot['foot_velocity']=env.rigid_states[:,env.feet_indices,7:10].clone()
        env.eval_snapshot['base_quat']=env.base_quat.clone()
        env.eval_snapshot['local_velocity']=env.base_lin_vel.clone()
env.check_termination=capture;env.capture_eval=False
env.reset();env.episode_length_buf.zero_();env.capture_eval=True
commands=np.array([.05,.10,.15])[np.arange(a.num_envs)%3]
env.commands[:,0]=torch.tensor(commands,device=env.device);env.commands[:,1:]=0.;env.compute_observations();obs=env.get_observations()
start=env.root_states[:,:3].cpu().numpy().copy();end=start.copy();last=end.copy()
active=np.ones(a.num_envs,bool);falls=np.zeros(a.num_envs,bool);duration=np.zeros(a.num_envs)
off=np.zeros((a.num_envs,4));clear_time=np.zeros_like(off);peak_clear_time=np.zeros_like(off);lift=np.zeros((a.num_envs,4,3));steps=np.zeros_like(off,dtype=int)
prior=np.ones_like(off,dtype=bool);max_clear=np.full_like(off,-1.);off_samples=np.zeros_like(off);valid_samples=np.zeros(a.num_envs)
path=np.zeros(a.num_envs);yaw_sum=np.zeros(a.num_envs);last_yaw=np.zeros(a.num_envs);forward=np.zeros(a.num_envs);err=np.zeros(a.num_envs)
contact_clearances=[];slip_sum=np.zeros_like(off);contact_samples=np.zeros_like(off);trace=[];t0=time.time()
with torch.inference_mode():
    for k in range(math.ceil(a.seconds/env.dt)):
        action=torch.zeros(a.num_envs,12,device=env.device) if a.zero else actor.act_inference(obs)
        obs,_,_,done,_=env.step(action);s=env.eval_snapshot
        xyz=s['xyz'].cpu().numpy();feet=s['feet'].cpu().numpy();quat=s['foot_quat'].cpu().numpy();force=s['force'].cpu().numpy()
        q=quat[:,:,None,:3];w=quat[:,:,None,3:4]
        world=feet[:,:,None,:]+endpoints+2*np.cross(q,np.cross(q,endpoints)+w*endpoints)
        clearance=world[:,:,:,2].min(2)-radii;contact=force>1.
        taking=prior&~contact;lift[taking]=feet[taking];clear_time[taking]=0.;peak_clear_time[taking]=0.
        off+=~contact*env.dt;clear_time=np.where(~contact&(clearance>=.002),clear_time+env.dt,0.)
        peak_clear_time=np.maximum(peak_clear_time,clear_time)
        landing=~prior&contact
        if k*env.dt>.5:steps+=(landing&(off>=.06)&(peak_clear_time>=.06)&(np.linalg.norm(feet-lift,axis=2)>=.005)&active[:,None])
        off[contact]=0.;clear_time[contact]=0.;peak_clear_time[contact]=0.;prior=contact
        max_clear[active]=np.maximum(max_clear[active],clearance[active]);off_samples[active]+=(clearance[active]>=.002)
        if k*env.dt>.5:
            contact_clearances.extend(clearance[(force>2.)&active[:,None]].tolist())
            slip_sum[active]+=np.linalg.norm(s['foot_velocity'].cpu().numpy()[active,:,:2],axis=2)*contact[active]
            contact_samples[active]+=contact[active]
        path[active]+=np.linalg.norm(xyz[active,:2]-last[active,:2],axis=1)
        base=s['base_quat'].cpu().numpy();x,y,z,wq=base.T;yaw=np.arctan2(2*(wq*z+x*y),1-2*(y*y+z*z))
        if k:yaw_sum[active]+=np.arctan2(np.sin(yaw[active]-last_yaw[active]),np.cos(yaw[active]-last_yaw[active]))
        last_yaw=yaw;velocity=s['local_velocity'].cpu().numpy();forward[active]+=velocity[active,0]*env.dt
        err[active]+=(velocity[active,0]-commands[active])**2;valid_samples[active]+=1
        end[active]=xyz[active];last=xyz.copy();duration[active]=(k+1)*env.dt;falls[active]|=s['fall'].cpu().numpy()[active]
        if active[0] and k%5==0:trace.append({'t':(k+1)*env.dt,'body_xyz':s['body_xyz'][0].cpu().tolist(),'clearance_m':clearance[0].tolist(),
            'force':force[0].tolist(),'base_quat':base[0].tolist(),'local_velocity':velocity[0].tolist(),'q':s['q'][0].cpu().tolist()})
        active&=~done.cpu().numpy().astype(bool)
        if not active.any():break
dx=end[:,0]-start[:,0];dy=end[:,1]-start[:,1];slip=slip_sum/np.maximum(contact_samples,1)
surface_error=float(np.percentile(np.abs(contact_clearances),95)) if contact_clearances else None
summary={'survival_fraction':float((~falls).mean()),'median_forward_displacement_m':float(np.median(dx)),
    'mean_min_clear_steps':float(steps.min(1).mean()),'mean_clear_steps_per_foot':steps.mean(0).tolist(),
    'moving_all_four_fraction':float(((steps.min(1)>=2)&(dx>.15)&~falls).mean()),
    'median_abs_heading_change_deg':float(np.median(np.abs(yaw_sum))*180/math.pi),
    'median_path_m':float(np.median(path)),'mean_peak_clearance_m':max_clear.mean(0).tolist(),
    'mean_stance_body_origin_speed_m_s':slip.mean(0).tolist(),'contact_surface_abs_error_p95_m':surface_error,
    'capsule_geometry_consistent':surface_error is not None and surface_error<.003}
result={'schema':'sable-capsule-gait-audit/1','checkpoint':str(a.checkpoint),'iteration':int(cp['iter']),'checkpoint_sha256':sha256(a.checkpoint),
    'runtime_sha256':cp['infos']['runtime_sha256'],'auditor_sha256':sha256(__file__),'created_unix':time.time(),'wall_seconds':time.time()-t0,
    'num_envs':a.num_envs,'seconds':a.seconds,'seed':a.seed,'zero_actions':a.zero,'foot_order':MODEL['foot_bodies'],'summary':summary,
    'step_definition':'force <1N, capsule surface clearance >=2mm for >=60ms, then landing with foot-origin travel >=5mm; first 0.5s excluded',
    'scope':'Nominal flat-ground simulation only; stance body-origin velocity includes legitimate capsule rolling and is not contact-point slip.',
    'samples':[{'command_m_s':float(commands[i]),'dx_m':float(dx[i]),'dy_m':float(dy[i]),'path_m':float(path[i]),'body_forward_integral_m':float(forward[i]),
        'heading_change_deg':float(yaw_sum[i]*180/math.pi),'duration_s':float(duration[i]),'fell':bool(falls[i]),'steps':steps[i].tolist(),
        'velocity_rmse_m_s':float(np.sqrt(err[i]/max(valid_samples[i],1)))} for i in range(a.num_envs)],'body_names':env.body_names,'trace':trace}
atomic_json(a.output,result);print('GAIT_AUDIT',json.dumps(summary),flush=True);env.gym.destroy_sim(env.sim)
