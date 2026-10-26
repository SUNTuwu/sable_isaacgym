"""Remote bounded inference at multiple fixed reference amplitudes; no PPO."""
import argparse,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=16.);p.add_argument('--speed',type=float,default=.04)
p.add_argument('--strengths',type=float,nargs='+',default=[.25,.375,.5]);a=p.parse_args()
run=a.checkpoint.resolve().parent.parent;sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi
import torch,numpy as np,json,time
from pathlib import Path
from rsl_rl.modules import ActorCritic
from preflight import create_env
from sable_config import SableCfg
from run_io import read_json,atomic_json,sha256
here=Path(__file__).resolve().parent;root=here.parent.parent
torch.manual_seed(947);np.random.seed(947)
cfg=SableCfg();cfg.gait.enabled=True;cfg.gait.period_s=.8;cfg.gait.height_profile='sine_squared'
cfg.reference.enabled=True;cfg.reference.strength=1.;cfg.env.num_observations=52;cfg.env.episode_length_s=a.seconds;cfg.commands.heading_command=True;cfg.commands.resampling_time=1000.
groups=[{'strength':s,'parent_policy':p} for p in [False,True] for s in a.strengths]
n=len(groups)*6;env=create_env(n,noise=False,cfg=cfg);env.reset_noise=.015;env.reset();env.episode_length_buf.zero_()
amp=torch.tensor([g['strength'] for g in groups for _ in range(6)],device=env.device)
mask=torch.tensor([g['parent_policy'] for g in groups for _ in range(6)],device=env.device)[:,None]
original_ramp=env.motion_reference.ramp;env.motion_reference.ramp=lambda age:original_ramp(age)*amp
env.motion_reference.reset(torch.arange(n,device=env.device));env.commands[:,0]=a.speed;env.commands[:,1:]=0.
config=read_json(run/'config.json');assert config['env']['env']['num_observations']==52
actor_class=ActorCritic
if config.get('mirror_policy'):
    from equivariant_policy import SymmetricActorCritic
    actor_class=SymmetricActorCritic
actor=actor_class(52,52,12,**config['train']['policy']).to(env.device)
actor.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict']);actor.eval()
env.compute_observations();obs=env.get_observations();env.capture_eval=True
start=env.root_states[:,:3].clone();end=start.clone();active=torch.ones(n,dtype=torch.bool,device=env.device);duration=torch.zeros(n,device=env.device)
peaks=torch.zeros(n,4,device=env.device);steps=torch.zeros(n,4,device=env.device);prior=torch.ones(n,4,dtype=torch.bool,device=env.device)
air=torch.zeros_like(steps);peakair=air.clone();lift=torch.zeros(n,4,3,device=env.device)
with torch.inference_mode():
    for k in range(round(a.seconds/env.dt)):
        obs,_,_,done,_=env.step(actor.act_inference(obs)*mask);s=env.eval_snapshot
        contact=s['force']>1.;takeoff=prior&~contact;lift[takeoff]=s['feet'][takeoff];peakair[takeoff]=0.
        air=torch.where(~contact&(s['clearance']>=.002),air+env.dt,0.);peakair=torch.maximum(peakair,air)
        if k*env.dt>.5:steps+=((~prior&contact)&(peakair>=.06)&((s['feet']-lift).norm(dim=2)>=.005)&active[:,None]).float()
        peakair[contact]=0.;prior=contact
        peaks[active]=torch.maximum(peaks[active],s['clearance'][active]);end[active]=s['xyz'][active];duration[active]=(k+1)*env.dt
        active &=~s['fall']
dx=(end[:,0]-start[:,0]).cpu().numpy();rows=[]
samples=[{'group':i//6,'repetition':i%6,'survived':bool(active[i]),'duration_s':float(duration[i]),'displacement_m':float(dx[i]),
    'steps_per_foot':steps[i].cpu().tolist(),'peak_clearance_m':peaks[i].cpu().tolist()} for i in range(n)]
for i,g in enumerate(groups):
    ids=slice(i*6,(i+1)*6);rows.append(dict(g,survival_horizon=float(active[ids].float().mean()),mean_duration_s=float(duration[ids].mean()),
        median_dx_m=float(np.median(dx[ids])),mean_steps_per_foot=steps[ids].mean(0).cpu().tolist(),peak_clearance_m=peaks[ids].mean(0).cpu().tolist(),
        moving_all_four_fraction=sum(r['survived'] and r['displacement_m']>.15 and min(r['steps_per_foot'])>=2 for r in samples if r['group']==i)/6))
report={'created_unix':time.time(),'source_sha256':sha256(__file__),'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),'horizon_s':a.seconds,'envs_per_group':6,'command_m_s':a.speed,'groups':rows,
    'samples':samples,'seed':947,'initial_joint_noise_rad':.015,'paired_initial_states':False,
    'scope':'Fixed-amplitude diagnostic; six independent initial states per group, not paired. Manually supplied reference motion is not learned gait.'}
atomic_json(a.output,report);print('REFERENCE_STRENGTH',json.dumps(report));env.gym.destroy_sim(env.sim)
