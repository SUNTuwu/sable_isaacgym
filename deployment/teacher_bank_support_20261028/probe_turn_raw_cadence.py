"""Small deterministic two-turn probe with actual per-foot event timing."""
import os,sys,json,time,hashlib,importlib.util,statistics
from pathlib import Path
from isaacgym import gymapi,gymtorch
import numpy as np
import torch

p=Path(os.environ['SABLE_GYM_ROOT']);bank=Path(sys.argv[1]).resolve();runtime=bank/'runtime'
sys.path.insert(0,str(runtime));spec=importlib.util.spec_from_file_location('timing_core',runtime/'teacher_bank_core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
torch.set_num_threads(2)
ids=['04_TURN_LEFT','05_TURN_RIGHT'];parents=[];receipts=[]
for ident in ids:
 r=json.loads((bank/ident/'checkpoint.json').read_text());cp=Path(r['path'])
 assert bank/ident in cp.resolve().parents and hashlib.sha256(cp.read_bytes()).hexdigest()==r['sha256']
 x=torch.load(str(cp),map_location='cpu');assert x['teacher_id']==ident and x['observations']==89
 parents.append(x);receipts.append(r)
assert parents[0]['iteration']==parents[1]['iteration'];iteration=parents[0]['iteration']
seed=947+iteration;torch.manual_seed(seed);np.random.seed(seed)
reference=Path(parents[0]['reference']).parent;manifest=json.loads((reference/'manifest.json').read_text())
clips=[next(c for c in manifest['clips'] if c['id']==i) for i in ids]
state=json.loads((bank/'teacher_bank.json').read_text())
cfg=core.student_config(p/'runs/stand5_20261027_v44_stride80_all',stage='S3',num_envs=8)
cfg.migration.freeze_waist=False;cfg.migration.moving_start_training=False
cfg.migration.command_frontier_alpha=1.;cfg.migration.history_noise_level=0.
cfg.commands.resampling_time=60.;cfg.env.episode_length_s=24.
for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
cfg.rewards.scales.alive=1.
cfg.teacher_bank=dict(clips=clips,train_per=0,eval_per=4,reference=str(reference),
 control_mode='reference_plus_residual',reference_strength=1.,time_scale=1.,imitation_scale=.2,
 offset_limit_deg=20.,balance_first=True,adaptive_enabled=True,motion_reward_module_enabled=True,
 walk_event_weight_scale=.25,task_tracking_profile='turn90_lateral_tracking_v1',step_quality_profile='strict_steps_v1',translation_guard_enabled=True,turn_speed_scale=state.get('turn_speed_scale',1.))
started=time.time();env=None
try:
 env=core.create_env(cfg);policies=[]
 for i,parent in enumerate(parents):
  policy=core.Policy().to(env.device);policy.load_state_dict(parent['model_state_dict']);policy.eval();policies.append(policy)
  env.adaptive.load_teacher_state(i,parent['curriculum_state'])
 env.eval_reset_phase=0.;env.reset();alive=torch.ones(8,device=env.device,dtype=torch.bool);frames=[]
 with torch.no_grad():
  for step in range(200):
   assert time.time()-started<90,'Probe wall limit'
   obs=env.teacher_observations();assert obs.shape==(8,89)
   action=torch.cat([policy.actor(obs[i*4:(i+1)*4]) for i,policy in enumerate(policies)])
   env.step(action);d=env.pre_reset_teacher_state
   frames.append(dict(time_s=(step+1)*env.dt,root=d['root'].clone(),toe=d['toe'].clone(),
    center=env.foot_center_world.clone(),clearance=env.foot_clearance.clone(),
    normal=d['contact_force'].clone(),alive=alive.clone()))
   alive&=~d['fall']
 output={'bank':str(bank),'iteration':iteration,'dt':env.dt,'horizon_s':4.,'seed':seed,
         'checkpoints':receipts,'teacher_ids':ids,'frames':[],'survived':alive.cpu().tolist()}
 for f in frames:output['frames'].append({k:v.cpu().tolist() if torch.is_tensor(v) else v for k,v in f.items()})
 dest=Path(sys.argv[2]);dest.write_text(json.dumps(output));print(json.dumps({'output':str(dest),'wall_s':time.time()-started,'survived':output['survived']}),flush=True)
finally:
 if env is not None:env.gym.destroy_sim(env.sim)
