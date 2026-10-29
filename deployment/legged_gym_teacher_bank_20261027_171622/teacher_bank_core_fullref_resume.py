"""Independent from-scratch reference teachers sharing one Isaac Gym batch."""
import argparse,datetime,json,math,os,signal,time,hashlib
from pathlib import Path
from isaacgym import gymapi,gymtorch
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from migration_env import MigrationRobot,create_migration_env
from migration_config import student_config
from action_schema import ACTION_NAMES
from isaacgym.torch_utils import quat_rotate_inverse
from model_mapping import quat_apply

STOP=False
def stop(*args):
 global STOP;STOP=True
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix('.partial');part.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');part.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.detach().cpu())
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
 return a==b

class Policy(nn.Module):
 def __init__(self):
  super().__init__()
  def net(out):return nn.Sequential(nn.Linear(86,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,out))
  self.actor=net(15);self.critic=net(1);self.log_std=nn.Parameter(torch.ones(15)*math.log(.08))
  nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
 def distribution(self,obs):return Normal(self.actor(obs),self.log_std.exp().clamp(.02,.3))
 def value(self,obs):return self.critic(obs).squeeze(-1)

class TeacherRobot(MigrationRobot):
 def _init_buffers(self):
  super()._init_buffers()
  spec=self.cfg.teacher_bank
  self.teacher_count=len(spec['clips']);self.train_per=spec['train_per'];self.eval_per=spec['eval_per']
  types=[]
  for i in range(self.teacher_count):types.extend([i]*self.train_per)
  for i in range(self.teacher_count):types.extend([i]*self.eval_per)
  self.teacher_ids=torch.tensor(types,device=self.device,dtype=torch.long)
  self.training_env_count=self.train_per*self.teacher_count
  self.eval_ids=torch.arange(self.training_env_count,self.num_envs,device=self.device)
  self.ref_phase=torch.zeros(self.num_envs,device=self.device)
  self._teacher_reset_phase=torch.zeros(self.num_envs,device=self.device)
  self.eval_reset_phase=None
  all_data=[np.load(str(Path(spec['reference'])/(c['id']+'.npz')),allow_pickle=False) for c in spec['clips']]
  names={'q':'joint_position_policy_order','qd':'joint_velocity_policy_order','root':'root_transform_world',
         'velocity':'root_linear_velocity_world','omega':'root_angular_velocity_world','toe':'toe_actual_world','surface':'foot_capsule_min_z'}
  self.ref={key:torch.tensor(np.stack([d[field] for d in all_data]),device=self.device,dtype=torch.float32) for key,field in names.items()}
  self.reference_time_scale=float(spec.get('time_scale',1.))
  assert self.reference_time_scale>0
  self.ref_duration=torch.tensor([float(d['time_s'][-1])*self.reference_time_scale for d in all_data],device=self.device)
  for key in ('qd','velocity','omega'):self.ref[key]/=self.reference_time_scale
  quat=[]
  from scipy.spatial.transform import Rotation
  for d in all_data:quat.append(Rotation.from_matrix(d['root_transform_world'][:,:3,:3]).as_quat())
  self.ref['quat']=torch.tensor(np.stack(quat),device=self.device,dtype=torch.float32)
  # Quaternion hemisphere continuity is essential for interpolation.
  for i in range(1,51):
   flip=(self.ref['quat'][:,i-1]*self.ref['quat'][:,i]).sum(-1)<0
   self.ref['quat'][flip,i]*=-1
  commands=[]
  for c,d in zip(spec['clips'],all_data):
   r=d['root_transform_world'];p=r[-1,:3,3]-r[0,:3,3];yaw0=math.atan2(r[0,1,0],r[0,0,0]);yaws=np.unwrap(np.arctan2(r[:,1,0],r[:,0,0]));period=float(d['time_s'][-1])*self.reference_time_scale;cs,sn=math.cos(yaw0),math.sin(yaw0)
   commands.append([(cs*p[0]+sn*p[1])/period,(-sn*p[0]+cs*p[1])/period,(yaws[-1]-yaws[0])/period])
  self.teacher_commands=torch.tensor(commands,device=self.device,dtype=torch.float32)
  phases=[]
  for c in spec['clips']:
   ident=c['id'];phases.append([.5,.75,.25] if ident=='09_BACKWARD_WALK' else [.5,.25,.75] if ident in ('01_FORWARD_POLICY','04_TURN_LEFT','05_TURN_RIGHT') else [.5,.5,0.])
  self.teacher_gait_phases=torch.tensor(phases,device=self.device)
  self.teacher_terms={};self.pre_reset_teacher_obs=torch.zeros(self.num_envs,86,device=self.device)
  self.pre_reset_teacher_state={}
  self.reference_initializations=0
  self.teacher_control_mode=spec.get('control_mode','direct_full_target')
  self.reference_strength=float(spec.get('reference_strength',1.))
  self.imitation_scale=float(spec.get('imitation_scale',1.))
  self.offset_limit_rad=math.radians(float(spec.get('offset_limit_deg',10.)))
  self.current_residual_action=torch.zeros(self.num_envs,15,device=self.device)
  self.current_reference_target=self.default_dof_pos[:,self.active].expand(self.num_envs,-1).clone()
 def ref_sample(self,phase=None,ids=None):
  ids=torch.arange(self.num_envs,device=self.device) if ids is None else ids
  p=self.ref_phase[ids] if phase is None else phase
  typ=self.teacher_ids[ids];x=p.clamp(0,1)*50;i=x.long().clamp(0,49);w=x-i
  result={}
  for k,v in self.ref.items():
   a=v[typ,i];b=v[typ,i+1];weight=w.reshape((-1,)+(1,)*(a.ndim-1));result[k]=a+(b-a)*weight
  result['quat']=result['quat']/result['quat'].norm(dim=1,keepdim=True).clamp(min=1e-8)
  return result
 def _resample_commands(self,ids):
  if not hasattr(self,'teacher_commands'):return super()._resample_commands(ids)
  self.commands16[ids]=self.command_defaults
  self.commands16[ids,:3]=self.teacher_commands[self.teacher_ids[ids]]
  self.commands16[ids,5]=1./self.ref_duration[self.teacher_ids[ids]]
  self.commands16[ids,6:9]=self.teacher_gait_phases[self.teacher_ids[ids]]
  self.commands16[ids,9]=.6
  self.commands[ids,:3]=self.commands16[ids,:3];self.commands[ids,3]=0.
 def reset_idx(self,ids):
  if hasattr(self,'ref_phase') and len(ids):
   phase=torch.rand(len(ids),device=self.device)
   if self.eval_reset_phase is not None:
    mask=ids>=self.training_env_count;phase[mask]=self.eval_reset_phase
   self.ref_phase[ids]=phase
  super().reset_idx(ids)
  if hasattr(self,'ref') and len(ids):
   ref=self.ref_sample(ids=ids)
   self.executed_target[ids]=ref['q'];self.student_target[ids]=ref['q'];self.full_student_target[ids[:,None],self.active]=ref['q']
   self.clock.phase[ids]=self.ref_phase[ids]
   self.actions[ids]=0.;self.last_actions[ids]=0.;self.last_dof_vel[ids]=self.dof_vel[ids]
   self.history.reset(ids)
 def _reset_dofs(self,ids):
  if not hasattr(self,'ref'):return super()._reset_dofs(ids)
  ref=self.ref_sample(ids=ids);self.dof_pos[ids]=self.default_dof_pos;self.dof_pos[ids[:,None],self.active]=ref['q'];self.dof_vel[ids]=0.;self.dof_vel[ids[:,None],self.active]=ref['qd']
  self.dof_pos[ids,self.dependent]=-self.dof_pos[ids,self.hip];self.dof_vel[ids,self.dependent]=-self.dof_vel[ids,self.hip]
  actor=self.robot_actor_indices[ids];self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(actor),len(ids));self.reference_initializations+=len(ids)
 def _reset_root_states(self,ids):
  if not hasattr(self,'ref'):return super()._reset_root_states(ids)
  ref=self.ref_sample(ids=ids);self.root_states[ids]=self.base_init_state
  self.root_states[ids,:3]=ref['root'][:,:3,3]+self.env_origins[ids];self.root_states[ids,3:7]=ref['quat'];self.root_states[ids,7:10]=ref['velocity'];self.root_states[ids,10:13]=ref['omega']
  actor=self.robot_actor_indices[ids];self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_all),gymtorch.unwrap_tensor(actor),len(ids))
 def teacher_observations(self):
  self._refresh_proprio_state();ref=self.ref_sample()
  q=(ref['q']-self.default_dof_pos[:,self.active]);phase=self.ref_phase*2*math.pi
  return torch.cat((self._proprio(),self.base_lin_vel*2.,torch.sin(phase)[:,None],torch.cos(phase)[:,None],q,ref['qd']*.05),-1)
 def step(self,actions):
  self.ref_phase=torch.remainder(self.ref_phase+self.dt/self.ref_duration[self.teacher_ids],1.)
  self.current_residual_action=actions.clone()
  if self.teacher_control_mode=='reference_plus_residual':
   self.current_reference_target=self.ref_sample()['q']
   default=self.default_dof_pos[:,self.active]
   self.current_offset_rad=(.35*actions).clamp(-self.offset_limit_rad,self.offset_limit_rad)
   full=self.current_reference_target+self.current_offset_rad
   actions=(full-default)/.35
   self.reference_execution_calls+=1
  return super().step(actions)
 def check_termination(self):
  super().check_termination()
  if not hasattr(self,'ref'):return
  ref=self.ref_sample();q=self.dof_pos[:,self.active];qd=self.dof_vel[:,self.active]
  self.pre_reset_teacher_obs=self.teacher_observations().clone()
  # Task-space reference relative to each measured torso; root travel is free.
  local=torch.matmul((ref['toe']-ref['root'][:,:3,3,None].transpose(1,2)),ref['root'][:,:3,:3])
  tips=[]
  for index,points,radius in self.mapping.capsules:
   distal=int(points.square().sum(-1).argmax());axis=points[distal]-points[1-distal]
   tips.append(points[distal]+axis/axis.norm()*radius)
  tip=torch.stack(tips);feet=self.rigid_states[:,self.feet_indices]
  actual_toe=feet[:,:,:3]+quat_apply(feet[:,:,3:7],tip[None].expand(self.num_envs,-1,-1))
  actual_local=quat_rotate_inverse(self.base_quat[:,None,:].expand(-1,4,-1).reshape(-1,4),
      (actual_toe-self.root_states[:,:3,None].transpose(1,2)).reshape(-1,3)).reshape(-1,4,3)
  toe_error=(actual_local-local).square().sum(-1).mean(1)
  qe=(q-ref['q']).square().mean(1);qde=(qd-ref['qd']).square().mean(1)
  v=self.heading_linear_velocity;cmd=self.teacher_commands[self.teacher_ids]
  ve=(v[:,:2]-cmd[:,:2]).square().sum(1);we=(self.heading_yaw_rate-cmd[:,2]).square()
  gref=quat_rotate_inverse(ref['quat'],self.gravity_vec);ge=(self.projected_gravity-gref).square().sum(1)
  he=(self.root_states[:,2]-self.env_origins[:,2]-ref['root'][:,2,3]).square()
  contact=self.contact_forces[:,self.feet_indices,2]>1.
  clearance=(self.foot_clearance-ref['surface']).square().mean(1)
  slip=(self.rigid_states[:,self.feet_indices,7:9].square().sum(-1)*contact).mean(1)
  sigma_v=torch.maximum(torch.ones_like(ve)*.01,.4*cmd[:,:2].norm(dim=1))
  sigma_yaw=torch.maximum(torch.ones_like(we)*.04,.4*cmd[:,2].abs())
  terms={'q_imitation':8*torch.exp(-qe/.02),'qd_imitation':torch.exp(-qde/12.),
         'foot_imitation':4*torch.exp(-toe_error/.000025),'foot_clearance':2*torch.exp(-clearance/.000025),'velocity':3*torch.exp(-ve/sigma_v.square()),
         'yaw':2*torch.exp(-we/sigma_yaw.square()),'orientation':2*torch.exp(-ge/.1),'height':2*torch.exp(-he/.0016),
         'alive':torch.ones_like(qe),'slip':-.5*slip,'torque':-.00005*self.torques.square().sum(1),
         'action_rate':-.01*(self.actions-self.last_actions).square().mean(1)}
  for key in ('q_imitation','qd_imitation','foot_imitation','foot_clearance'):terms[key]*=self.imitation_scale
  if self.teacher_control_mode=='reference_plus_residual':terms['offset_magnitude']=-.5*(self.current_offset_rad/self.offset_limit_rad).square().mean(1)
  reward=sum(terms.values())*self.dt-3.*self.true_fall.float()
  self.teacher_terms=terms;self.teacher_reward=reward
  self.pre_reset_teacher_state={'q_rmse':qe.sqrt().clone(),'xy_error':ve.sqrt().clone(),'yaw_error':we.sqrt().clone(),
    'actual_velocity':torch.cat((v[:,:2],self.heading_yaw_rate[:,None]),-1).clone(),
    'q':q.clone(),'root':self.root_states.clone(),'toe':actual_toe.clone(),
    'reference_toe':(self.root_states[:,:3,None].transpose(1,2)+quat_apply(self.base_quat[:,None,:].expand(-1,4,-1),local)).clone(),'contact_force':self.contact_forces[:,self.feet_indices,2].clone(),
    'contact':contact.clone(),'phase':self.ref_phase.clone(),'fall':self.true_fall.clone(),
    'executed_target_full':self.executed_target.clone(),'residual_action':self.current_residual_action.clone(),'reference_joint_target':self.current_reference_target.clone(),
    'offset_joint_rad':self.current_offset_rad.clone() if self.teacher_control_mode=='reference_plus_residual' else torch.zeros_like(q)}
  tilt=(self.projected_gravity*self.reference_gravity).sum(1)
  terminal_contact=torch.any(self.contact_forces[:,self.termination_contact_indices].norm(dim=-1)>2,dim=1)
  coupling=(self.dof_pos[:,self.hip]+self.dof_pos[:,self.dependent]).abs()
  waist_error=(self.dof_pos[:,self.waist]+math.radians(40)).abs()
  self.pre_reset_teacher_state['fall_diagnostics']={'tilt_dot':tilt.clone(),'root_z_m':self.root_states[:,2].clone(),'coupling_error_rad':coupling.clone(),'waist_error_rad':waist_error.clone(),
   'tilt_failure':(tilt<.55),'height_failure':(self.root_states[:,2]<.055),'termination_contact_failure':terminal_contact,'coupling_failure':(coupling>.08),'waist_failure':(waist_error>.6),'knee_ground_failure':self.knee_ground_contact.any(1)&bool(self.cfg.termination.knee_contact_enabled)}
 def compute_reward(self):
  if hasattr(self,'teacher_reward'):self.rew_buf.copy_(self.teacher_reward)
  else:super().compute_reward()

def create_env(cfg):
 import migration_env
 original=migration_env.MigrationRobot;migration_env.MigrationRobot=TeacherRobot
 try:return create_migration_env(cfg,noise=False)
 finally:migration_env.MigrationRobot=original

def update(policy,opt,storage,last_value):
 obs,actions,oldlog,reward,done,values,timeout_values=storage
 values=torch.stack(values);rewards=torch.stack(reward);dones=torch.stack(done);timeouts=torch.stack(timeout_values)
 returns=torch.zeros_like(rewards);adv=torch.zeros_like(last_value);last=last_value
 for t in reversed(range(len(rewards))):
  delta=rewards[t]+.99*(last*(~dones[t])+timeouts[t])-values[t]
  adv=delta+.99*.95*(~dones[t])*adv;returns[t]=adv+values[t];last=values[t]
 advantages=(returns-values).flatten();advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
 obs=torch.stack(obs).flatten(0,1);actions=torch.stack(actions).flatten(0,1);oldlog=torch.stack(oldlog).flatten();target=returns.flatten();oldvalue=values.flatten()
 losses=[]
 for epoch in range(3):
  perm=torch.randperm(len(obs),device=obs.device)
  for ids in perm.chunk(2):
   dist=policy.distribution(obs[ids]);log=dist.log_prob(actions[ids]).sum(-1);ratio=(log-oldlog[ids]).exp();a=advantages[ids]
   surrogate=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean();v=policy.value(obs[ids]);vl=(v-target[ids]).square().mean()
   loss=surrogate+vl*.5-.002*dist.entropy().sum(-1).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step();policy.log_std.data.clamp_(math.log(.02),math.log(.3));losses.append([float(vl),float(surrogate)])
 return np.mean(losses,axis=0).tolist()

def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--duration',type=float,default=1200);p.add_argument('--deadline');p.add_argument('--resume-bank',type=Path);p.add_argument('--control-mode',choices=['direct_full_target','reference_plus_residual'],default='reference_plus_residual');p.add_argument('--reference-strength',type=float,default=1.);p.add_argument('--time-scale',type=float,default=1.);p.add_argument('--offset-limit-deg',type=float,default=10.);p.add_argument('--imitation-scale',type=float,default=.5);p.add_argument('--envs-per-teacher',type=int,default=256);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.control_mode=='reference_plus_residual' and (a.reference_strength!=1. or a.time_scale!=1.):raise ValueError('Authored reference amplitude and timing must remain1; teacher learns bounded joint offsets only')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 torch.set_num_threads(2);torch.manual_seed(741);np.random.seed(741)
 ids=['01_FORWARD_POLICY','04_TURN_LEFT','05_TURN_RIGHT','06_BACKWARD','07_LATERAL_TROT_LEFT','08_LATERAL_TROT_RIGHT','09_BACKWARD_WALK','10_FORWARD_TROT']
 manifest=json.loads((a.reference/'manifest.json').read_text());clips=[next(c for c in manifest['clips'] if c['id']==ident) for ident in ids]
 a.bank.mkdir(parents=True,exist_ok=True);project=Path(os.environ['SABLE_GYM_ROOT'])
 cfg=student_config(project/'runs/stand5_20261027_v44_stride80_all',stage='S3',num_envs=len(ids)*(a.envs_per_teacher+4))
 cfg.migration.freeze_waist=False;cfg.migration.moving_start_training=False;cfg.migration.command_frontier_alpha=1.;cfg.migration.history_noise_level=0.
 cfg.commands.resampling_time=60.;cfg.env.episode_length_s=24.
 for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
 cfg.rewards.scales.alive=1.
 cfg.teacher_bank={'clips':clips,'train_per':a.envs_per_teacher,'eval_per':4,'reference':str(a.reference),'control_mode':a.control_mode,'reference_strength':a.reference_strength,'time_scale':a.time_scale,'imitation_scale':a.imitation_scale,'offset_limit_deg':a.offset_limit_deg}
 env=create_env(cfg);env.reset();policies=[Policy().to(env.device) for _ in ids];opts=[torch.optim.Adam(x.parameters(),lr=3e-4) for x in policies]
 resume_iteration=0;resume_sources=[]
 if a.resume_bank:
  source=a.resume_bank.resolve();assert source.parent==project/'runs' and source.name.startswith('teacher_bank_fullref_')
  for i,ident in enumerate(ids):
   receipt=json.loads((source/ident/'checkpoint.json').read_text(encoding='utf8'));cp=Path(receipt['path']).resolve()
   assert source/ident in cp.parents and sha(cp)==receipt['sha256']
   parent=torch.load(str(cp),map_location=env.device)
   assert parent['teacher_id']==ident and parent['teacher_control_mode']=='reference_plus_residual'
   assert parent['reference_strength']==1. and parent['reference_time_scale']==1. and parent['offset_limit_deg']==a.offset_limit_deg
   assert parent['reference_sha256']==sha(a.reference/(ident+'.npz')) and parent['action_names']==list(ACTION_NAMES)
   policies[i].load_state_dict(parent['model_state_dict'],strict=True);opts[i].load_state_dict(parent['optimizer_state_dict'])
   assert state_equal(policies[i].state_dict(),parent['model_state_dict']) and state_equal(opts[i].state_dict(),parent['optimizer_state_dict'])
   resume_sources.append({'teacher_id':ident,'path':str(cp),'sha256':receipt['sha256'],'iteration':parent['iteration'],'policy_equal':True,'optimizer_equal':True})
  assert len(set(x['iteration'] for x in resume_sources))==1,'Refuse to mix teacher checkpoint iterations'
  resume_iteration=resume_sources[0]['iteration']
  atomic(a.bank/'resume_verification.json',{'passed':True,'source_bank':str(source),'sources':resume_sources,'initial_iteration':resume_iteration,'new_environments_reset':True,'old_rollout_storage_reused':False})
 start=time.time();deadline=datetime.datetime.fromisoformat(a.deadline).timestamp() if a.deadline else start+a.duration;assert deadline>start;iteration=resume_iteration;last_save=0.;eval_state=None;last_metrics=[]
 for ident in ids:(a.bank/ident/'checkpoints').mkdir(parents=True,exist_ok=True)
 bank={'status':'smoke' if a.smoke else 'training','run_dir':str(a.bank),'pid':os.getpid(),'started_unix':start,'deadline':datetime.datetime.fromtimestamp(deadline,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'updated_unix':start,'num_envs':env.num_envs,'training_envs':env.training_env_count,'reserved_evaluation_envs':32,'initialization':'random_no_checkpoint_loaded_zero_mean_offset','teacher_control_mode':a.control_mode,'reference_strength':a.reference_strength,'reference_time_scale':a.time_scale,'offset_limit_deg':a.offset_limit_deg,'imitation_reward_scale':a.imitation_scale,'reference_sampling':'Pre-action observation uses phase_t; env advances phase by dt before composing full q_ref(phase_t+dt)+bounded offset. q_default is only the lower PD-coordinate conversion origin.','trace_is_BC_dataset':False,'student_distillation_label':'Collect pre-action326D observation -> complete clipped executed_target_full, never residual_action alone','teachers':[{'id':c['id'],'title':c.get('title',c['id']),'run_dir':str(a.bank/c['id']),'reference_metadata':str(a.reference/(c['id']+'.json')),'status':'training'} for c in clips]}
 atomic(a.bank/'teacher_bank.json',bank)
 bank.update(initial_iteration=resume_iteration,resume_start_iteration=resume_iteration,resume_from_bank=str(a.resume_bank.resolve()) if a.resume_bank else None)
 if a.resume_bank:bank['initialization']='resume_current_full_reference_offset_teachers_with_Adam_verified'
 atomic(a.bank/'teacher_bank.json',bank)
 atomic(project/'results/teacher_bank/current.json',{'bank':str(a.bank),'deployment':str(Path(__file__).resolve().parent),'status':bank['status']})
 def save(reason):
  receipts=[]
  for i,ident in enumerate(ids):
   out=a.bank/ident;cp=out/'checkpoints'/('model_%06d.pt'%iteration)
   if not cp.exists():torch.save({'model_state_dict':policies[i].state_dict(),'optimizer_state_dict':opts[i].state_dict(),'iteration':iteration,'teacher_id':ident,'initialization':'random','observations':86,'actions':15,'action_names':list(ACTION_NAMES),'teacher_control_mode':a.control_mode,'reference_strength':a.reference_strength,'reference_time_scale':a.time_scale,'offset_limit_deg':a.offset_limit_deg,'imitation_reward_scale':a.imitation_scale,'action_definition':'q_target=clip(q_ref(next_phase)+clip(.35*offset_action,+/-offset_limit))' if a.control_mode=='reference_plus_residual' else 'q_target=clip(q_default+.35*action)','student_supervision':'pre-action326 observation -> complete clipped executed target, not residual alone; trace is visualization not BC data','reference':str(a.reference/(ident+'.npz')),'reference_sha256':sha(a.reference/(ident+'.npz')),'reason':reason},str(cp))
   receipt={'path':str(cp),'sha256':sha(cp),'iteration':iteration,'updated_unix':time.time(),'reason':reason};atomic(out/'checkpoint.json',receipt);receipts.append(receipt)
  return receipts
 receipts=save('initial')
 eval_policies=[Policy().to(env.device) for _ in ids]
 def begin_eval(receipts):
  nonlocal eval_state
  for e,t in zip(eval_policies,policies):e.load_state_dict(t.state_dict())
  env.eval_reset_phase=0.;env.reset_idx(env.eval_ids);env._refresh_proprio_state();env.compute_observations();env.eval_reset_phase=None
  generator=torch.Generator(device=env.device).manual_seed(947+iteration)
  perturb=(torch.rand((32,15),device=env.device,generator=generator)-.5)*.01
  env.dof_pos[env.eval_ids[:,None],env.active]+=perturb
  env.dof_pos[env.eval_ids,env.dependent]=-env.dof_pos[env.eval_ids,env.hip]
  actor=env.robot_actor_indices[env.eval_ids];env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(actor),32)
  env._refresh_proprio_state();env.compute_observations()
  eval_state={'step':0,'alive':torch.ones(32,device=env.device,dtype=torch.bool),'count':torch.zeros(32,device=env.device),'q':torch.zeros(32,device=env.device),'xy':torch.zeros(32,device=env.device),'yaw':torch.zeros(32,device=env.device),'actual':torch.zeros(32,3,device=env.device),'steps':torch.zeros(32,4,device=env.device),'last_contact':torch.zeros(32,4,device=env.device,dtype=torch.bool),'trace':[[] for _ in ids],'receipts':receipts,'iteration':iteration,'first_fall_diagnostics':[None]*32}
 def finish_eval():
  nonlocal eval_state
  es=eval_state
  for i,ident in enumerate(ids):
   sl=slice(i*4,(i+1)*4);n=es['count'][sl].clamp(min=1)
   report={'teacher_id':ident,'iteration':es['iteration'],'checkpoint_iteration':es['iteration'],'checkpoint':es['receipts'][i],
     'independent':True,'evaluation_seed':947+es['iteration'],'reset':'reference_phase0 plus seeded uniform joint perturbation +/-0.005rad; reserved environments never used by optimizer','horizon_s':16.,'num_envs':4,
     'survival_fraction':float(es['alive'][sl].float().mean()),'q_tracking_rmse_rad':float((es['q'][sl].sum()/n.sum()).sqrt()),'xy_rmse_m_s':float((es['xy'][sl].sum()/n.sum()).sqrt()),'yaw_rmse_rad_s':float((es['yaw'][sl].sum()/n.sum()).sqrt()),'mean_command':env.teacher_commands[i].tolist(),'mean_actual':(es['actual'][sl]/n[:,None]).mean(0).tolist(),'steps_per_foot':es['steps'][sl].sum(0).tolist(),'step_scope':'Observed normal-force contact onset; not yet genuine-air-step acceptance','samples':int(n.sum()),'accepted':False,'first_fall_diagnostics':es['first_fall_diagnostics'][i*4:(i+1)*4],'teacher_control_mode':a.control_mode,'reference_strength':a.reference_strength,'reference_time_scale':a.time_scale,'created_unix':time.time()}
   path=a.bank/ident/'evaluations'/('independent_%06d.json'%es['iteration']);atomic(path,report);atomic(a.bank/ident/'latest_evaluation.json',report)
   tracepath=a.bank/ident/'videos'/('trace_%06d.json'%es['iteration']);atomic(tracepath,{'teacher_id':ident,'iteration':es['iteration'],'joint_names':list(ACTION_NAMES),'type':'kinematic_trace','frames':es['trace'][i]})
   atomic(a.bank/ident/'video_receipt.json',{'type':'kinematic_trace','path':str(tracepath),'iteration':es['iteration'],'checkpoint_path':es['receipts'][i]['path'],'checkpoint_sha256':es['receipts'][i]['sha256'],'source':'Independent reserved-env deterministic rollout','created_unix':time.time()})
  eval_state=None
 def collect_eval():
  if eval_state is None:return
  es=eval_state;sl=slice(env.training_env_count,env.num_envs);state=env.pre_reset_teacher_state;active=es['alive'].float()
  es['count']+=active;es['q']+=state['q_rmse'][sl].square()*active;es['xy']+=state['xy_error'][sl].square()*active;es['yaw']+=state['yaw_error'][sl].square()*active;es['actual']+=state['actual_velocity'][sl]*active[:,None]
  newly_failed=torch.nonzero(es['alive']&state['fall'][sl],as_tuple=False).flatten()
  for k in newly_failed.tolist():
   j=env.training_env_count+k;es['first_fall_diagnostics'][k]={'time_s':(es['step']+1)*env.dt,**{name:value[j].item() for name,value in state['fall_diagnostics'].items()}}
  contact=state['contact'][sl];es['steps']+=((contact&~es['last_contact'])*es['alive'][:,None]).float();es['last_contact']=contact;es['alive']&=~state['fall'][sl]
  if es['step']%5==0:
   for i in range(len(ids)):
    j=env.training_env_count+i*4;root=state['root'][j];es['trace'][i].append({'time_s':(es['step']+1)*env.dt,'q':state['q'][j].tolist(),'root_position':(root[:3]-env.env_origins[j]).tolist(),'root_quaternion_xyzw':root[3:7].tolist(),'toe_world':(state['toe'][j]-env.env_origins[j]).tolist(),'reference_toe_world':(state['reference_toe'][j]-env.env_origins[j]).tolist(),'contact_normal_force':state['contact_force'][j].tolist(),'commands':env.teacher_commands[i].tolist(),'actual_velocity':state['actual_velocity'][j].tolist(),'active':bool(es['alive'][i*4]),'executed_target_full':state['executed_target_full'][j].tolist(),'residual_action':state['residual_action'][j].tolist(),'offset_joint_rad':state['offset_joint_rad'][j].tolist(),'reference_joint_target':state['reference_joint_target'][j].tolist(),'teacher_control_mode':a.control_mode})
  es['step']+=1
  if es['step']>=800:finish_eval()
 begin_eval(receipts)
 outcome='completed'
 try:
  while time.time()<deadline and not STOP and not (a.bank/'STOP').exists():
   storages=[[[],[],[],[],[],[],[]] for _ in ids];reward_sum=np.zeros(len(ids));begin=time.time()
   for step in range(24):
    obs=env.teacher_observations();action=torch.zeros(env.num_envs,15,device=env.device)
    with torch.no_grad():
     for i,(policy,store) in enumerate(zip(policies,storages)):
      sl=slice(i*a.envs_per_teacher,(i+1)*a.envs_per_teacher);dist=policy.distribution(obs[sl]);act=dist.sample();action[sl]=act
      store[0].append(obs[sl].clone());store[1].append(act);store[2].append(dist.log_prob(act).sum(-1));store[5].append(policy.value(obs[sl]))
     for i,policy in enumerate(eval_policies):action[env.training_env_count+i*4:env.training_env_count+(i+1)*4]=policy.actor(obs[env.training_env_count+i*4:env.training_env_count+(i+1)*4])
    _,_,reward,done,info=env.step(action)
    with torch.no_grad():
     for i,(policy,store) in enumerate(zip(policies,storages)):
      sl=slice(i*a.envs_per_teacher,(i+1)*a.envs_per_teacher);store[3].append(reward[sl].clone());store[4].append(done[sl].bool().clone());store[6].append(policy.value(env.pre_reset_teacher_obs[sl])*env.time_out_buf[sl]);reward_sum[i]+=float(reward[sl].mean())
    collect_eval()
   with torch.no_grad():last=env.teacher_observations()
   losses=[]
   for i,(policy,opt,store) in enumerate(zip(policies,opts,storages)):
    sl=slice(i*a.envs_per_teacher,(i+1)*a.envs_per_teacher)
    with torch.no_grad():last_value=policy.value(last[sl])
    losses.append(update(policy,opt,store,last_value))
   iteration+=1;elapsed=time.time()-start;updates_this_run=iteration-resume_iteration;throughput=updates_this_run*24*env.training_env_count/max(elapsed,1e-3)
   assert all(torch.isfinite(v).all() for policy in policies for v in policy.parameters())
   for i,ident in enumerate(ids):
    group=slice(i*a.envs_per_teacher,(i+1)*a.envs_per_teacher);offset=env.pre_reset_teacher_state['offset_joint_rad'][group]
    metrics={'mean_reward_per_control_step':float(reward_sum[i]/24),'value_loss':losses[i][0],'surrogate_loss':losses[i][1],'action_std':float(policies[i].log_std.exp().mean()),'elapsed_s':elapsed,'updates_this_run':updates_this_run,'environment_steps_per_second':throughput/len(ids),'iteration_time_s':time.time()-begin,'reference_execution_calls':env.reference_execution_calls,'teacher_control_mode':a.control_mode,'reference_strength':a.reference_strength,'offset_rms_rad':float(offset.square().mean().sqrt()),'offset_clip_fraction':float((offset.abs()>=env.offset_limit_rad-1e-6).float().mean())}
    atomic(a.bank/ident/'status.json',{'teacher_id':ident,'reference_id':ident,'status':'training','pid':os.getpid(),'iteration':iteration,'deadline':bank['deadline'],'metrics':metrics,'updated_unix':time.time(),'initialization':'random'})
   bank.update(iteration=iteration,updated_unix=time.time(),metrics={'environment_steps_per_second':throughput,'elapsed_s':elapsed,'updates_this_run':updates_this_run,'iteration_time_s':time.time()-begin});atomic(a.bank/'teacher_bank.json',bank)
   if iteration==1 or time.time()-last_save>=180:
    receipts=save('periodic');last_save=time.time()
    if eval_state is None:begin_eval(receipts)
   if a.smoke and iteration>=2:break
   if iteration%5==0:print(json.dumps({'iteration':iteration,'elapsed':elapsed,'samples_per_second':throughput}),flush=True)
  if not a.smoke and not STOP and (time.time()>=deadline or not (a.bank/'STOP').exists()):
   receipts=save('final');bank.update(status='final_evaluation',updated_unix=time.time());atomic(a.bank/'teacher_bank.json',bank);begin_eval(receipts)
   while eval_state is not None and not STOP:
    obs=env.teacher_observations();action=torch.zeros(env.num_envs,15,device=env.device)
    with torch.no_grad():
     for i,policy in enumerate(eval_policies):
      sl=slice(env.training_env_count+i*4,env.training_env_count+(i+1)*4);action[sl]=policy.actor(obs[sl])
    env.step(action);collect_eval()
  outcome='smoke_complete' if a.smoke else 'stopped_at_deadline' if time.time()>=deadline else 'stopped_requested'
 except BaseException as error:
  outcome='failed';bank['error']=type(error).__name__+': '+str(error);raise
 finally:
  save('failed' if outcome=='failed' else 'smoke_final' if a.smoke else 'final');bank.update(status=outcome,updated_unix=time.time(),iteration=iteration)
  atomic(a.bank/'teacher_bank.json',bank)
  for ident in ids:
   status=json.loads((a.bank/ident/'status.json').read_text()) if (a.bank/ident/'status.json').exists() else {};status.update(status=bank['status'],updated_unix=time.time());atomic(a.bank/ident/'status.json',status)
  env.gym.destroy_sim(env.sim)
 print(json.dumps({'status':bank['status'],'iteration':iteration,'bank':str(a.bank)}),flush=True)
if __name__=='__main__':main()
