"""Bounded remote policy inference: isolate observation/action-noise dependence."""
import argparse,sys,json,time,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
run=a.checkpoint.parent.parent;sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi
import torch,numpy as np
from rsl_rl.modules import ActorCritic
from preflight import create_env
from sable_config import SableCfg
from run_io import atomic_json,sha256,read_json

torch.manual_seed(947);np.random.seed(947)
cfg=SableCfg();cfg.env.episode_length_s=16.;cfg.commands.resampling_time=1000.
env=create_env(72,noise=False,cfg=cfg);env.reset_noise=.015
config=read_json(run/'config.json');cp=torch.load(str(a.checkpoint),map_location='cuda:0')
actor=ActorCritic(48,48,12,**config['train']['policy']).to('cuda:0');actor.load_state_dict(cp['model_state_dict']);actor.eval()
env.reset();env.capture_eval=True;env.commands[:,0]=.08;env.commands[:,1:]=0.;env.compute_observations();obs=env.get_observations()
start=env.root_states[:,0].clone();end=start.clone();active=torch.ones(72,device='cuda:0',dtype=torch.bool)
falls=torch.zeros_like(active);duration=torch.zeros(72,device='cuda:0');air=torch.zeros(72,4,device='cuda:0')
prior=torch.ones(72,4,device='cuda:0',dtype=torch.bool);lift=torch.zeros(72,4,3,device='cuda:0');steps=torch.zeros(72,4,device='cuda:0')
t0=time.time()
with torch.inference_mode():
    for k in range(math.ceil(16./env.dt)):
        noisy=obs.clone();noisy[24:48]+=(2*torch.rand_like(noisy[24:48])-1)*env.noise_scale_vec
        actions=actor.act_inference(noisy);actions[48:]+=torch.randn_like(actions[48:])*actor.std
        obs,_,_,done,_=env.step(actions);s=env.eval_snapshot
        contact=s['force']>1.;taking=prior&~contact;lift[taking]=s['feet'][taking]
        air+=~contact*env.dt;landing=~prior&contact
        steps+=(landing&(air>=.06)&((s['feet']-lift).norm(dim=2)>=.005)&active[:,None]).float()
        air[contact]=0.;prior=contact
        end[active]=s['xyz'][active,0];duration[active]=(k+1)*env.dt;falls[active]|=s['fall'][active]
        active&=~done.bool()
dx=end-start;groups=[]
for i,name in enumerate(['clean_deterministic','observation_noise_only','action_noise_only']):
    sl=slice(i*24,(i+1)*24)
    groups.append({'name':name,'num_envs':24,'survival':float((~falls[sl]).float().mean()),'duration_s':float(duration[sl].mean()),
        'median_displacement_m':float(dx[sl].median()),'mean_displacement_m':float(dx[sl].mean()),
        'mean_min_foot_steps':float(steps[sl].min(1).values.mean()),'mean_steps_per_foot':steps[sl].mean(0).tolist()})
report={'iteration':int(cp['iter']),'checkpoint_sha256':sha256(a.checkpoint),'script_sha256':sha256(__file__),
    'runtime_sha256':cp['infos']['runtime_sha256'],'wall_seconds':time.time()-t0,'created_unix':time.time(),
    'command_m_s':.08,'nominal_friction':1.,'policy_std':actor.std.detach().cpu().tolist(),'groups':groups}
out=run/'diagnostics'/('noise_domains_%06d.json'%int(cp['iter']));atomic_json(out,report)
print('NOISE_DOMAINS',json.dumps(report),flush=True);env.gym.destroy_sim(env.sim)
