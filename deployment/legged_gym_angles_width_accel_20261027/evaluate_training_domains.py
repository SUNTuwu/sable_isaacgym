"""Remote inference only: isolate stops, command changes, friction and action noise."""
import argparse,json,math,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=20.);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
run=a.checkpoint.resolve().parent.parent;sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi
import torch,numpy as np
from rsl_rl.modules import ActorCritic
from preflight import create_env
from sable_config import SableCfg,MODEL
from run_io import atomic_json,read_json,sha256

torch.manual_seed(1367);np.random.seed(1367)
config=read_json(run/'config.json');cfg=SableCfg();cfg.env.episode_length_s=a.seconds
cfg.commands.heading_command=config['env']['commands']['heading_command'];cfg.commands.resampling_time=1000.
if config['env'].get('gait',{}).get('enabled'):
    for key,value in config['env']['gait'].items():setattr(cfg.gait,key,value)
    cfg.env.num_observations=config['env']['env']['num_observations']
schedules={'stop':[0.,0.,0.,0.],'slow':[.04]*4,'forward':[.10]*4,'transition':[.10,0.,.04,.12]}
cases=[{'friction':mu,'action_noise':noisy,'schedule':name,'repeat':r}
       for mu in [.8,1.,1.2] for noisy in [False,True] for name in schedules for r in range(a.repeats)]
env=create_env(len(cases),noise=False,cfg=cfg);env.reset_noise=.03
# Only the four terminal capsule friction values change; all other asset properties stay frozen.
for i,case in enumerate(cases):
    shapes=env.gym.get_actor_rigid_shape_properties(env.envs[i],env.actor_handles[i])
    ranges=env.gym.get_actor_rigid_body_shape_indices(env.envs[i],env.actor_handles[i])
    for body,name in enumerate(env.body_names):
        for k in range(ranges[body].start,ranges[body].start+ranges[body].count):
            spec=MODEL['shape_metadata'][MODEL['collision_shapes_by_body'][name][k-ranges[body].start]]
            if spec['foot']:shapes[k].friction=case['friction']
    env.gym.set_actor_rigid_shape_properties(env.envs[i],env.actor_handles[i],shapes)
actor=ActorCritic(env.num_obs,env.num_obs,12,**config['train']['policy']).to(env.device)
cp=torch.load(str(a.checkpoint),map_location=env.device);actor.load_state_dict(cp['model_state_dict']);actor.eval()
original_check=env.check_termination
def capture():
    original_check()
    if getattr(env,'capture_eval',False):
        env.eval_snapshot['failure_values']={
            'tilt_dot':(env.projected_gravity*env.reference_gravity).sum(1).clone(),
            'height_m':env.root_states[:,2].clone(),
            'torso_force_n':env.contact_forces[:,env.termination_contact_indices].norm(dim=-1).max(1).values.clone(),
            'waist_error_rad':(env.dof_pos[:,env.waist]+math.radians(40)).abs().clone()}
env.check_termination=capture;env.capture_eval=False
env.reset();env.episode_length_buf.zero_();env.capture_eval=True
n=len(cases);active=np.ones(n,bool);falls=np.zeros(n,bool);duration=np.zeros(n)
start=env.root_states[:,:3].cpu().numpy().copy();end=start.copy();peak_clearance=np.zeros((n,4));failure=[None]*n
noise_mask=torch.tensor([c['action_noise'] for c in cases],device=env.device)[:,None]
step_counts=np.zeros((n,4),int);air=np.zeros((n,4));clear=np.zeros_like(air);peak_clear=np.zeros_like(air)
prior=np.ones_like(air,bool);lift=np.zeros((n,4,3));t0=time.time()
with torch.inference_mode():
    for k in range(math.ceil(a.seconds/env.dt)):
        phase=min(3,int(k*env.dt/5.))
        command=np.array([schedules[c['schedule']][phase] for c in cases])
        env.commands[:,0]=torch.tensor(command,device=env.device);env.commands[:,1]=0.;env.commands[:,3]=0.
        env.compute_observations();obs=env.get_observations()
        actions=actor.act_inference(obs)+torch.randn(n,12,device=env.device)*actor.std*noise_mask
        _,_,_,done,_=env.step(actions);s=env.eval_snapshot
        xyz=s['xyz'].cpu().numpy();height=s['clearance'].cpu().numpy();contact=s['force'].cpu().numpy()>1.
        feet=s['feet'].cpu().numpy();taking=prior&~contact;lift[taking]=feet[taking];clear[taking]=0.;peak_clear[taking]=0.
        air+=~contact*env.dt;clear=np.where(~contact&(height>=.002),clear+env.dt,0.);peak_clear=np.maximum(peak_clear,clear)
        landing=~prior&contact
        if k*env.dt>.5:step_counts+=landing&(air>=.06)&(peak_clear>=.06)&(np.linalg.norm(feet-lift,axis=2)>=.005)&active[:,None]
        air[contact]=0.;clear[contact]=0.;peak_clear[contact]=0.;prior=contact
        if k*env.dt>=1.:peak_clearance[active]=np.maximum(peak_clearance[active],height[active])
        end[active]=xyz[active];duration[active]=(k+1)*env.dt
        failed=active&s['fall'].cpu().numpy()
        for i in np.flatnonzero(failed):
            failure[i]={name:float(value[i]) for name,value in s['failure_values'].items()}
            failure[i].update(command_m_s=float(command[i]),coupling_error_rad=float(s['coupling'][i]))
        falls|=failed;active&=~done.cpu().numpy().astype(bool)
        if not active.any():break
groups=[]
for mu in [.8,1.,1.2]:
    for noisy in [False,True]:
        for name in schedules:
            ids=np.array([i for i,c in enumerate(cases) if c['friction']==mu and c['action_noise']==noisy and c['schedule']==name])
            groups.append({'friction':mu,'action_noise':noisy,'schedule':name,'environments':len(ids),
                'survival_fraction':float((~falls[ids]).mean()),'median_duration_s':float(np.median(duration[ids])),
                'median_dx_m':float(np.median(end[ids,0]-start[ids,0])),
                'mean_steps_per_foot':step_counts[ids].mean(0).tolist(),
                'mean_peak_clearance_after_1s_m':peak_clearance[ids].mean(0).tolist()})
result={'schema':'sable-training-domain-audit/1','iteration':int(cp['iter']),'checkpoint':str(a.checkpoint),
    'checkpoint_sha256':sha256(a.checkpoint),'runtime_sha256':cp['infos']['runtime_sha256'],'auditor_sha256':sha256(__file__),
    'created_unix':time.time(),'wall_seconds':time.time()-t0,'horizon_s':a.seconds,'seed':1367,'num_envs':n,
    'reset_joint_noise_rad':.03,'observation_noise':False,'action_std':actor.std.detach().cpu().tolist(),
    'schedules_m_s_per_5s':schedules,'foot_order':MODEL['foot_bodies'],'groups':groups,
    'scope':'Fresh simulation inference only. Three repeats per cell by default; limited diagnostic evidence, not hardware validation.',
    'samples':[dict(cases[i],fell=bool(falls[i]),duration_s=float(duration[i]),dx_m=float(end[i,0]-start[i,0]),steps=step_counts[i].tolist(),failure=failure[i]) for i in range(n)]}
atomic_json(a.output,result);print('DOMAIN_AUDIT',json.dumps(groups),flush=True);env.gym.destroy_sim(env.sim)
