"""Remote deterministic inference audit; no optimization or controller changes."""
import argparse,sys,json,time,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=8.);a=p.parse_args();run=a.checkpoint.resolve().parent.parent
sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi
import torch,numpy as np
from rsl_rl.modules import ActorCritic
from preflight import create_env
from sable_config import SableCfg
from guided_reference import PAIRS
from run_io import read_json,atomic_json,sha256
config=read_json(run/'config.json');torch.manual_seed(947);np.random.seed(947)
def apply(obj,data):
    for k,v in data.items():
        if k=='file':continue
        if isinstance(v,dict):apply(getattr(obj,k),v)
        else:setattr(obj,k,v)
cfg=SableCfg()
for part in ['gait','reference','commands','rewards']:apply(getattr(cfg,part),config['env'][part])
cfg.env.num_observations=config['env']['env']['num_observations'];cfg.env.episode_length_s=a.seconds;cfg.commands.resampling_time=1000.
n=12;env=create_env(n,noise=False,cfg=cfg);env.reset_noise=.015;env.reset();env.episode_length_buf.zero_()
assert env.motion_reference is not None
env.motion_reference.reset(torch.arange(n,device=env.device));commands=torch.tensor([.03,.045,.06]*4,device=env.device)
env.commands[:,0]=commands;env.commands[:,1:]=0.;env.compute_observations();obs=env.get_observations();env.capture_eval=True
actor_class=ActorCritic
if config.get('mirror_policy'):
    from equivariant_policy import SymmetricActorCritic
    actor_class=SymmetricActorCritic
actor=actor_class(env.num_obs,env.num_obs,12,**config['train']['policy']).to(env.device)
cp=torch.load(str(a.checkpoint),map_location=env.device);actor.load_state_dict(cp['model_state_dict']);actor.eval()
captured={}
for i,name in enumerate(env.reward_names):
    f=env.reward_functions[i]
    def wrap(function=f,key=name):
        value=function();captured[key]=value;return value
    env.reward_functions[i]=wrap
pairs=torch.tensor([[env.dof_names.index(x) for x in pair] for pair in PAIRS],device=env.device)
alive=torch.ones(n,dtype=torch.bool,device=env.device);duration=torch.zeros(n,device=env.device)
start=env.root_states[:,:3].clone();end=start.clone();rows=[];terms={};observations=0
stats={label:{k:torch.zeros(4,2 if k in ['reference_rad','residual_rad','actual_delta_rad','tracking_sq_rad','clipped_fraction'] else 1,device=env.device)
    for k in ['count','reference_rad','residual_rad','actual_delta_rad','tracking_sq_rad','clipped_fraction','clearance_m','target_clearance_m','contact_fraction','force_n']}
    for label in ['swing','stance']}
rootz=torch.zeros(1,device=env.device);raw_negative=torch.zeros(1,device=env.device)
with torch.inference_mode():
    for k in range(round(a.seconds/env.dt)):
        obs,_,_,done,_=env.step(actor.act_inference(obs));s=env.eval_snapshot
        end[alive]=s['xyz'][alive];duration[alive]=(k+1)*env.dt
        alive &=~s['fall'];valid=alive&~done
        if k*env.dt<1. or not bool(valid.any()):continue
        reference=env.motion_reference.end
        residual=torch.zeros_like(env.dof_pos);residual[:,env.active]=env.actions*env.cfg.control.action_scale
        target=env.default_dof_pos+reference+residual
        clipped=torch.maximum(torch.minimum(target,env.hard_limits[:,1]),env.hard_limits[:,0])
        data={'reference_rad':reference[:,pairs],'residual_rad':residual[:,pairs],
            'actual_delta_rad':(s['q']-env.default_dof_pos)[:,pairs],
            'tracking_sq_rad':(clipped-s['q']).square()[:,pairs],
            'clipped_fraction':((clipped-target).abs()>1e-6).float()[:,pairs],
            'clearance_m':s['clearance'][:,:,None],'target_clearance_m':env.gait_target_clearance[:,:,None],
            'contact_fraction':(s['force']>1.).float()[:,:,None],'force_n':s['force'][:,:,None]}
        phase=torch.remainder(env.gait_signals.phase(env)[:,None]+env.gait_signals.offsets,1.)
        progress=(phase-env.cfg.gait.stance_fraction)/(1-env.cfg.gait.stance_fraction)
        swing=(progress>=.15)&(progress<=.85);stance=phase<env.cfg.gait.stance_fraction
        for label,phase in [('swing',swing),('stance',stance)]:
            mask=(valid[:,None]&phase)[:,:,None];stats[label]['count']+=mask.sum(0)
            for key,value in data.items():stats[label][key]+=(value*mask).sum(0)
        count=int(valid.sum());observations+=count;rootz+=s['xyz'][valid,2].sum()
        raw=torch.zeros(n,device=env.device)
        for name,value in captured.items():
            weighted=value*env.reward_scales[name]/env.dt
            terms[name]=terms.get(name,torch.zeros(1,device=env.device))+weighted[valid].sum();raw+=weighted
        raw_negative+=(raw[valid]<0).float().sum()
for label,data in stats.items():
    count=data['count'].clamp(min=1);result={}
    for name,value in data.items():
        if name=='count':result[name]=value[:,0].cpu().tolist();continue
        value=value/count
        if name=='tracking_sq_rad':value=torch.sqrt(value);name='tracking_rms_deg';value=value*180/math.pi
        elif name.endswith('_rad'):name=name[:-4]+'_deg';value=value*180/math.pi
        result[name]=value.cpu().tolist()
    rows.append({'phase':label,'per_foot':result})
valid_phases=all(all(x>0 for x in row['per_foot']['count']) for row in rows)
result={'diagnostic_valid':valid_phases,'phase_mask':'Unscaled clock: swing progress 0.15 to 0.85; stance before duty boundary.',
    'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),'source_sha256':sha256(__file__),
    'runtime_sha256':cp['infos']['runtime_sha256'],'created_unix':time.time(),'num_envs':n,'horizon_s':a.seconds,
    'reference_strength':cfg.reference.strength,'mirror_policy':bool(config.get('mirror_policy',False)),'seed':947,'initial_joint_noise_rad':.015,'commands_m_s':commands.cpu().tolist(),
    'survival_horizon':float(alive.float().mean()),'mean_duration_s':float(duration.mean()),
    'median_dx_m':float((end[:,0]-start[:,0]).median()),'samples_after_first_second':observations,
    'mean_root_height_m':float(rootz/max(1,observations)),'init_root_height_m':cfg.init_state.pos[2],
    'raw_reward_negative_fraction':float(raw_negative/max(1,observations)),
    'weighted_reward_mean_per_second':{k:float(v/max(1,observations)) for k,v in terms.items()},
    'foot_order':[env.body_names[i] for i in env.feet_indices.cpu().tolist()],'joint_pairs':PAIRS,'phase_statistics':rows,
    'scope':'50Hz target and measured joint audit after first second, alive pre-terminal steps only. No optimization; not a substep peak torque or hardware test.'}
atomic_json(a.output,result);print('GUIDED_TRACKING_AUDIT',json.dumps(result),flush=True);env.gym.destroy_sim(env.sim)
if not valid_phases:raise SystemExit('Insufficient phase observations')
