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
 x=torch.load(str(cp),map_location='cpu');assert x['teacher_id']==ident and x['observations']==87
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
 walk_event_weight_scale=.25,task_tracking_profile='turn90_lateral_tracking_v1',step_quality_profile='strict_steps_v1')
started=time.time();env=None
try:
 env=core.create_env(cfg);policies=[]
 for i,parent in enumerate(parents):
  policy=core.Policy().to(env.device);policy.load_state_dict(parent['model_state_dict']);policy.eval();policies.append(policy)
  env.adaptive.load_teacher_state(i,parent['curriculum_state'])
 env.eval_reset_phase=0.;env.reset();alive=torch.ones(8,device=env.device,dtype=torch.bool);events=[[],[]]
 with torch.no_grad():
  for step in range(300):
   assert time.time()-started<90,'Timing probe exceeded wall budget'
   obs=env.teacher_observations();assert obs.shape==(8,87)
   action=torch.cat([policy.actor(obs[i*4:(i+1)*4]) for i,policy in enumerate(policies)])
   env.step(action);d=env.pre_reset_teacher_state;m=d['motion_diagnostics']
   hit=m['quality_duration_landed']&alive[:,None]
   for row,foot in torch.nonzero(hit,as_tuple=False).tolist():
    events[row//4].append(dict(env=row%4,foot=['FL','FR','HL','HR'][foot],landing_s=(step+1)*env.dt,
       swing_ms=float(m['quality_swing_duration_s'][row,foot])*1000,
       valid=bool(m['quality_valid_landing'][row,foot])))
   alive&=~d['fall']
 def summary(values):return dict(count=len(values),mean_ms=statistics.mean(values) if values else None,
     median_ms=statistics.median(values) if values else None,min_ms=min(values) if values else None,max_ms=max(values) if values else None)
 reports=[]
 for i,ident in enumerate(ids):
  rows=events[i];intervals=[];stance=[];perfoot={}
  for foot in ['FL','FR','HL','HR']:
   cycles=[];swings=[]
   for env_index in range(4):
    selected=[x for x in rows if x['env']==env_index and x['foot']==foot]
    swings.extend(x['swing_ms'] for x in selected)
    for a,b in zip(selected,selected[1:]):
     cycles.append((b['landing_s']-a['landing_s'])*1000)
     stance.append((b['landing_s']-b['swing_ms']/1000-a['landing_s'])*1000)
   intervals.extend(cycles);perfoot[foot]=dict(swing=summary(swings),landing_to_landing=summary(cycles))
  reports.append(dict(teacher_id=ident,checkpoint=receipts[i],iteration=iteration,seed=seed,horizon_s=6.,
    environments=4,survival_fraction=float(alive[i*4:(i+1)*4].float().mean()),swing=summary([x['swing_ms'] for x in rows]),
    landing_to_landing=summary(intervals),stance_between_swings=summary(stance),per_foot=perfoot,events=rows,
    scope='Deterministic separate8env probe; v44 continuous2mm/60ms physical flights,20ms event resolution; intervals pair only the same env and foot, first episodes only. Counts are not task acceptance.'))
 out=dict(reports=reports,bank=str(bank),created_unix=time.time(),wall_s=time.time()-started)
 destination=bank/('turn_step_timing_%06d.json'%iteration);destination.write_text(json.dumps(out,indent=2))
 print(json.dumps({'result':str(destination),'reports':[{k:r[k] for k in ['teacher_id','iteration','survival_fraction','swing','landing_to_landing','stance_between_swings']} for r in reports]}),flush=True)
finally:
 if env is not None:env.gym.destroy_sim(env.sim)
