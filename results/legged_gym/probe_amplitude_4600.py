"""Remote bounded inference at multiple fixed reference amplitudes; no PPO."""
import argparse,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=16.);p.add_argument('--speed',type=float,default=.04)
p.add_argument('--strengths',type=float,nargs='+',default=[.5,.65,.75]);a=p.parse_args()
run=a.checkpoint.resolve().parent.parent;sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi,gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch,numpy as np,json,time
from pathlib import Path
from rsl_rl.modules import ActorCritic
from preflight import create_env
from sable_config import SableCfg
from run_io import read_json,atomic_json,sha256
from priority_metrics import PriorityMetrics
here=Path(__file__).resolve().parent;root=here.parent.parent
torch.manual_seed(947);np.random.seed(947)
config=read_json(run/'config.json')
assert not a.output.exists(),'Refusing to overwrite evidence'
cfg=SableCfg()
for part in ['gait','reference','posture','termination']:
    for key,value in config['env'][part].items():
        if key!='file':setattr(getattr(cfg,part),key,value)
cfg.gait.enabled=True;cfg.gait.period_s=.8;cfg.gait.height_profile='sine_squared'
cfg.reference.enabled=True;cfg.reference.strength=1.;cfg.env.num_observations=52;cfg.env.episode_length_s=a.seconds;cfg.commands.heading_command=True;cfg.commands.resampling_time=1000.
groups=[{'strength':s,'parent_policy':p} for p in [False,True] for s in a.strengths]
n=len(groups)*6;env=create_env(n,noise=False,cfg=cfg);env.reset_noise=.015;env.reset();env.episode_length_buf.zero_()
# Pair the six sampled initial states across every amplitude/control group.
for group in range(1,len(groups)):
    ids=slice(group*6,(group+1)*6)
    env.dof_pos[ids]=env.dof_pos[:6].clone();env.dof_vel[ids]=env.dof_vel[:6].clone()
    env.root_states[ids]=env.root_states[:6].clone()
    env.root_states[ids,:3]+=env.env_origins[ids]-env.env_origins[:6]
    env.gait_phase_offset[ids]=env.gait_phase_offset[:6].clone()
    env.last_actions[ids]=env.last_actions[:6].clone()
    env.actions[ids]=env.actions[:6].clone()
env.base_lin_vel=quat_rotate_inverse(env.base_quat,env.root_states[:,7:10])
env.base_ang_vel=quat_rotate_inverse(env.base_quat,env.root_states[:,10:13])
env.projected_gravity=quat_rotate_inverse(env.base_quat,env.gravity_vec)
env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state))
env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_all))
assert all(torch.equal(env.dof_pos[i*6:(i+1)*6],env.dof_pos[:6]) for i in range(len(groups)))
amp=torch.tensor([g['strength'] for g in groups for _ in range(6)],device=env.device)
mask=torch.tensor([g['parent_policy'] for g in groups for _ in range(6)],device=env.device)[:,None]
original_ramp=env.motion_reference.ramp;env.motion_reference.ramp=lambda age:original_ramp(age)*amp
assert torch.allclose(env.motion_reference.ramp(torch.full((n,),2.,device=env.device)),amp)
env.motion_reference.reset(torch.arange(n,device=env.device));env.commands[:,0]=a.speed;env.commands[:,1:]=0.
config=read_json(run/'config.json');assert config['env']['env']['num_observations']==52
actor_class=ActorCritic
if config.get('mirror_policy'):
    from equivariant_policy import SymmetricActorCritic
    actor_class=SymmetricActorCritic
actor=actor_class(52,52,12,**config['train']['policy']).to(env.device)
actor.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict']);actor.eval()
env.compute_observations();obs=env.get_observations();env.capture_eval=True
initial_observation_difference=max(float((obs[i*6:(i+1)*6]-obs[:6]).abs().max()) for i in range(len(groups)))
assert initial_observation_difference<1e-6
initial_joint_positions=env.dof_pos[:6].cpu().tolist()
start=env.root_states[:,:3].clone();end=start.clone();active=torch.ones(n,dtype=torch.bool,device=env.device);duration=torch.zeros(n,device=env.device)
peaks=torch.zeros(n,4,device=env.device);steps=torch.zeros(n,4,device=env.device);prior=torch.ones(n,4,dtype=torch.bool,device=env.device)
import math
priorities=PriorityMetrics(env.dof_names,env.body_names,n,env.dt,[math.degrees(cfg.init_state.default_joint_angles[name]) for name in ['L_hip_pitch','R_hip_pitch']],cfg.posture.hip_stand_tolerance_deg)
takeoff_time=torch.zeros(n,4,device=env.device)
air=torch.zeros_like(steps);peakair=air.clone();lift=torch.zeros(n,4,3,device=env.device)
with torch.inference_mode():
    for k in range(round(a.seconds/env.dt)):
        obs,_,_,done,_=env.step(actor.act_inference(obs)*mask);s=env.eval_snapshot
        contact=s['force']>1.;takeoff=prior&~contact;takeoff_time[takeoff]=(k+1)*env.dt;lift[takeoff]=s['feet'][takeoff];peakair[takeoff]=0.
        air=torch.where(~contact&(s['clearance']>=.002),air+env.dt,0.);peakair=torch.maximum(peakair,air)
        travel=(s['feet']-lift).norm(dim=2)
        qualified=(~prior&contact)&(peakair>=.06)&(travel>=.005)&active[:,None]&(k*env.dt>.5)
        steps+=qualified.float()
        priorities.observe((k+1)*env.dt,active.cpu().numpy(),s['q'].cpu().numpy(),s['body_xyz'].cpu().numpy(),s['body_force_norm'].cpu().numpy(),s['clearance'].cpu().numpy(),s['force'].cpu().numpy(),(qualified&(takeoff_time>1.)).cpu().numpy(),travel.cpu().numpy(),s['knee_clearance'].cpu().numpy())
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
    'samples':samples,'seed':947,'initial_joint_noise_rad':.015,'paired_initial_states':True,'initial_observation_max_difference':initial_observation_difference,'paired_initial_joint_positions_rad':initial_joint_positions,'mirror_policy':config.get('mirror_policy',False),'priority_diagnostics':priorities.result(),'frozen_runtime_manifest_sha256':sha256(run/'runtime_manifest.json'),
    'scope':'Fixed-amplitude inference only, six initial states paired across groups; no PPO. Stored priority samples map by environment index to group/repetition. Contact reward amplitude is not used to choose actions; no reward comparison claimed. Manually supplied reference motion is not learned gait; small simulation pilot, not hardware validation.'}
atomic_json(a.output,report);print('REFERENCE_STRENGTH',json.dumps(report));env.gym.destroy_sim(env.sim)
