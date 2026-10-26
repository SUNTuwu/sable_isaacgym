"""Remote inference A/B diagnostic for repeated episode resets; no optimization."""
import argparse,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=45.);a=p.parse_args()
run=a.checkpoint.resolve().parent.parent;sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse
import torch,numpy as np
from rsl_rl.modules import ActorCritic
from preflight import create_env
from sable_config import SableCfg
from run_io import atomic_json,read_json,sha256

torch.manual_seed(951);np.random.seed(951);config=read_json(run/'config.json');cfg=SableCfg()
cfg.env.episode_length_s=20.;cfg.commands.resampling_time=1000.;cfg.commands.heading_command=config['env']['commands']['heading_command']
if config['env'].get('gait',{}).get('enabled'):
    for key,value in config['env']['gait'].items():setattr(cfg.gait,key,value)
    cfg.env.num_observations=config['env']['env']['num_observations']
groups=['original','refresh_kinematic_observations','refresh_and_clear_action_history'];n=24
labels=np.repeat(np.arange(3),8);commands=np.tile(np.repeat([.04,.10],4),3)
env=create_env(n,noise=False,cfg=cfg);env.reset_noise=.03
actor=ActorCritic(env.num_obs,env.num_obs,12,**config['train']['policy']).to(env.device)
cp=torch.load(str(a.checkpoint),map_location=env.device);actor.load_state_dict(cp['model_state_dict']);actor.eval()
episode_number=np.zeros(n,int);records=[];reset_checks=[];clock=[0.];recording=[False]
original_reset=env.reset_idx
def reset(ids):
    ii=ids.cpu().numpy()
    if recording[0] and len(ii):
        for i in ii:
            records.append({'env':int(i),'group':groups[labels[i]],'episode':int(episode_number[i]),'t_s':clock[0],
                'duration_s':float(env.episode_length_buf[i]*env.dt),'fell':bool(env.true_fall[i]),
                'command_m_s':float(env.commands[i,0]),'height_m':float(env.root_states[i,2]),
                'torso_force_n':float(env.contact_forces[i,env.termination_contact_indices].norm(dim=-1).max())})
        episode_number[ii]+=1
    original_reset(ids)
    if not len(ii):return
    fix=ids[torch.tensor(labels[ii]>=1,device=env.device)]
    clear=ids[torch.tensor(labels[ii]>=2,device=env.device)]
    if len(fix):
        env.base_lin_vel[fix]=quat_rotate_inverse(env.base_quat[fix],env.root_states[fix,7:10])
        env.base_ang_vel[fix]=quat_rotate_inverse(env.base_quat[fix],env.root_states[fix,10:13])
        env.projected_gravity[fix]=quat_rotate_inverse(env.base_quat[fix],env.gravity_vec[fix])
    if len(clear):env.actions[clear]=0.
    env.commands[ids,0]=torch.tensor(commands[ii],device=env.device,dtype=env.commands.dtype);env.commands[ids,1:]=0.
env.reset_idx=reset
env.reset();env.episode_length_buf.zero_();recording[0]=True;t0=time.time()
noisy=torch.tensor(np.tile([False,True],12),device=env.device)[:,None]
with torch.inference_mode():
    for k in range(int(round(a.seconds/env.dt))):
        clock[0]=(k+1)*env.dt;env.compute_observations()
        action=actor.act_inference(env.get_observations())+torch.randn(n,12,device=env.device)*actor.std*noisy
        _,_,_,done,_=env.step(action)
        ids=torch.nonzero(done,as_tuple=False).flatten()
        if len(ids) and len(reset_checks)<120:
            actual_v=quat_rotate_inverse(env.base_quat[ids],env.root_states[ids,7:10])
            actual_g=quat_rotate_inverse(env.base_quat[ids],env.gravity_vec[ids])
            for j,i in enumerate(ids.cpu().tolist()):
                reset_checks.append({'env':i,'group':groups[labels[i]],'t_s':clock[0],
                    'velocity_cache_error_m_s':float((env.base_lin_vel[i]-actual_v[j]).norm()),
                    'gravity_cache_error':float((env.projected_gravity[i]-actual_g[j]).norm()),
                    'previous_action_norm':float(env.actions[i].norm())})
        if (k+1)%250==0:print('RESET_AUDIT_PROGRESS',round(clock[0],2),len(records),flush=True)
summary=[]
for name in groups:
    rr=[r for r in records if r['group']==name]
    later=[r for r in rr if r['episode']>=1]
    summary.append({'group':name,'completed_episodes':len(rr),'falls':sum(r['fell'] for r in rr),
        'post_first_episode_completed':len(later),'post_first_episode_falls':sum(r['fell'] for r in later),
        'median_duration_s':float(np.median([r['duration_s'] for r in rr])) if rr else None,
        'unique_failed_envs':len({r['env'] for r in rr if r['fell']})})
result={'schema':'sable-reset-audit/1','iteration':int(cp['iter']),'checkpoint':str(a.checkpoint),
    'checkpoint_sha256':sha256(a.checkpoint),'runtime_sha256':cp['infos']['runtime_sha256'],'auditor_sha256':sha256(__file__),
    'created_unix':time.time(),'wall_seconds':time.time()-t0,'total_sim_seconds':a.seconds,'episode_limit_s':20.,
    'seed':951,'num_envs':n,'scope':'Inference diagnostic only; alternate reset behavior is confined to this process.',
    'cases':[{'env':i,'group':groups[labels[i]],'command_m_s':float(commands[i]),'action_noise':bool(noisy[i])} for i in range(n)],
    'summary':summary,'episodes':records,'reset_observation_checks':reset_checks}
atomic_json(a.output,result);print('RESET_AUDIT',json.dumps(summary),flush=True);env.gym.destroy_sim(env.sim)
