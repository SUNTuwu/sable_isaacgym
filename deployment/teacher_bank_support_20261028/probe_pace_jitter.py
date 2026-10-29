"""Nonlearning learned/reference/smoothed pace diagnostics at20ms."""
import os,sys,json,time,hashlib,importlib.util,math
from pathlib import Path
from isaacgym import gymapi,gymtorch
import torch,numpy as np
root=Path(os.environ['SABLE_GYM_ROOT']);bank=Path(sys.argv[1]).resolve();runtime=bank/'runtime';out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(runtime));spec=importlib.util.spec_from_file_location('pace_diagnostic_core',runtime/'teacher_bank_core.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
torch.set_num_threads(2);torch.manual_seed(5317);np.random.seed(5317)
ids=['02_STRAFE_LEFT','03_STRAFE_RIGHT'];parents={};receipts={}
for ident in ids:
 r=json.loads((bank/ident/'checkpoint.json').read_text());p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];parents[ident]=torch.load(str(p),map_location='cpu');receipts[ident]=r
assert len({p['iteration'] for p in parents.values()})==1
state=json.loads((bank/'teacher_bank.json').read_text());reference=Path(parents[ids[0]]['reference']).parent;manifest=json.loads((reference/'manifest.json').read_text())
cases=[(mode,ident) for mode in ['learned','zero_offset','smoothed'] for ident in ids];clips=[next(c for c in manifest['clips'] if c['id']==ident) for _,ident in cases]
cfg=core.student_config(root/'runs/stand5_20261027_v44_stride80_all',stage='S3',num_envs=12)
cfg.migration.freeze_waist=False;cfg.migration.moving_start_training=False;cfg.migration.command_frontier_alpha=1.;cfg.migration.history_noise_level=0.;cfg.commands.resampling_time=60.;cfg.env.episode_length_s=24.
for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
cfg.rewards.scales.alive=1.
cfg.teacher_bank=dict(clips=clips,train_per=0,eval_per=2,reference=str(reference),control_mode='reference_plus_residual',reference_strength=1.,time_scale=1.,imitation_scale=.2,offset_limit_deg=20.,balance_first=True,adaptive_enabled=True,motion_reward_module_enabled=True,walk_event_weight_scale=.25,task_tracking_profile=state['task_tracking_profile'],step_quality_profile=state['step_quality_profile'],translation_guard_enabled=True,turn_speed_scale=.5,turn_reference_time_scale=3.,turn_offset_rate_deg_s=50.,turn_lift_config=state['turn_lift_config'])
env=None;start=time.time()
try:
 env=core.create_env(cfg);policies={}
 for ident in ids:
  p=core.Policy().to(env.device);p.load_state_dict(parents[ident]['model_state_dict']);p.eval();policies[ident]=p
 for i,(_,ident) in enumerate(cases):env.adaptive.load_teacher_state(i,parents[ident]['curriculum_state'])
 env.eval_reset_phase=0.;env.reset();frames=[];alive=torch.ones(12,device=env.device,dtype=torch.bool)
 # Static comparison of the animation FK and the actual physics asset FK.
 fk_reference=[]
 for i,ident in enumerate(ids):
  q=torch.zeros(51,len(env.dof_names),device=env.device);q[:,env.active]=env.ref['q'][i];q[:,env.dependent]=-q[:,env.hip]
  pos,rot=env.mapping.forward(env.ref['root'][i,:,:3,3],env.ref['quat'][i],q,env.dof_names);tips=[]
  for body,points,radius in env.mapping.capsules:
   distal=int(points.square().sum(-1).argmax());axis=points[distal]-points[1-distal];tip=points[distal]+axis/axis.norm()*radius
   tips.append(pos[:,body]+core.quat_apply(rot[:,body],tip[None].expand(51,-1)))
  err=(torch.stack(tips,1)-env.ref['toe'][i]).norm(dim=-1)*1000
  fk_reference.append({'teacher_id':ident,'mean_mm':err.mean(0).cpu().tolist(),'max_mm':err.max(0)[0].cpu().tolist()})
 with torch.no_grad():
  for step in range(150):
   assert time.time()-start<105,'Diagnostic wall budget'
   obs=env.teacher_observations();ref_now=env.ref_sample()['q'];action=torch.zeros(12,15,device=env.device)
   for i,(mode,ident) in enumerate(cases):
    sl=slice(i*2,(i+1)*2);a=policies[ident].actor(obs[sl])
    if mode=='zero_offset':a=torch.zeros_like(a)
    elif mode=='smoothed' and step:
     previous=(env.executed_target[sl]-ref_now[sl]).clamp(-math.radians(20),math.radians(20));requested=(.35*a).clamp(-math.radians(20),math.radians(20))
     a=(previous+(requested-previous).clamp(-math.radians(45)*env.dt,math.radians(45)*env.dt))/.35
    action[sl]=a
   env.step(action);d=env.pre_reset_teacher_state
   frames.append(dict(time_s=(step+1)*env.dt,alive=alive.clone(),fall=d['fall'].clone(),root=d['root'].clone(),q=d['q'].clone(),native_q=env.dof_pos.clone(),qd=env.dof_vel[:,env.active].clone(),torque=env.torques[:,env.active].clone(),target=d['executed_target_full'].clone(),offset=d['offset_joint_rad'].clone(),refq=d['reference_joint_target'].clone(),toe=d['toe'].clone(),ref_toe=d['reference_toe'].clone(),contact=d['contact_force'].clone()))
   alive&=~d['fall']
 output={'bank':str(bank),'iteration':parents[ids[0]]['iteration'],'dt':env.dt,'cases':cases,'frames':[],'fk_reference':fk_reference,'physics':{'physics_dt':cfg.sim.dt,'decimation':cfg.control.decimation,'kp':env.p_gains.cpu().tolist(),'kd':env.d_gains.cpu().tolist(),'torque_limits':env.torque_limits.cpu().tolist(),'dof_names':env.dof_names,'action_names':list(core.ACTION_NAMES)}}
 for f in frames:output['frames'].append({k:v.cpu().tolist() if torch.is_tensor(v) else v for k,v in f.items()})
 # Compare native measured tip transforms against FK at actual q/root, excluding reset frames.
 allq=torch.stack([f['native_q'] for f in frames]).reshape(-1,len(env.dof_names));allroot=torch.stack([f['root'] for f in frames]).reshape(-1,13)
 pos,rot=env.mapping.forward(allroot[:,:3],allroot[:,3:7],allq,env.dof_names);tips=[]
 for body,points,radius in env.mapping.capsules:
  distal=int(points.square().sum(-1).argmax());axis=points[distal]-points[1-distal];tip=points[distal]+axis/axis.norm()*radius;tips.append(pos[:,body]+core.quat_apply(rot[:,body],tip[None].expand(len(pos),-1)))
 actual=torch.stack([f['toe'] for f in frames]).reshape(-1,4,3);mask=torch.stack([f['alive']&~f['fall'] for f in frames]).reshape(-1)
 error=(torch.stack(tips,1)-actual).norm(dim=-1)[mask]*1000;output['native_fk_max_error_mm']=float(error.max());output['native_fk_mean_error_mm']=float(error.mean())
 dest=out/'diagnostic.json';dest.write_text(json.dumps(output));print(json.dumps({'file':str(dest),'fk_reference':fk_reference,'native_fk_max_error_mm':output['native_fk_max_error_mm'],'wall_s':time.time()-start}),flush=True)
finally:
 if env is not None:env.gym.destroy_sim(env.sim)
