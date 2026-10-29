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
import teacher_task_tracking as task_tracking
import teacher_reference_rewards_v2 as reference_v2
from teacher_yaw_integral import expand_checkpoint
from teacher_translation_guard import TranslationGuard,METADATA as GUARD_METADATA
from teacher_lift_limits import (LiftCountGuard,limit_turn_offset,CADENCE_CLEANUP_PROFILE,
                               REMOVED_CADENCE_TERMS,diagnostic_lift_config,cleanup_cadence_terms)
from teacher_event_symmetry import relative_xy
from teacher_pace_stability import PaceStability,METADATA as PACE_METADATA
OBSERVATIONS=94

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
  def net(out):return nn.Sequential(nn.Linear(94,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,out))
  self.actor=net(15);self.critic=net(1);self.log_std=nn.Parameter(torch.ones(15)*math.log(.08))
  nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
 def distribution(self,obs):return Normal(self.actor(obs),self.log_std.exp().clamp(getattr(self,'minimum_std',.02),getattr(self,'maximum_std',.3)))
 def value(self,obs):return self.critic(obs).squeeze(-1)

class AdaptiveTeacherCurriculum:
 """Bounded error tolerances and failure-aware circular phase sampling."""
 def __init__(self,env):
  self.env=env;self.device=env.device;self.count=env.teacher_count;self.bins=20;self.uniform_mix=.5
  command=env.teacher_commands;speed=command[:,:2].norm(dim=1);yaw=command[:,2].abs()
  self.lower=torch.stack((torch.full_like(speed,.005),torch.maximum(speed*.15,torch.full_like(speed,.005)),torch.maximum(yaw*.15,torch.full_like(yaw,.04))),1)
  self.upper=torch.stack((torch.full_like(speed,.025),torch.maximum(speed*.8,torch.full_like(speed,.04)),torch.maximum(yaw*1.5,torch.full_like(yaw,.35))),1)
  self.sigma=torch.stack((torch.full_like(speed,.010),torch.maximum(speed*.5,torch.full_like(speed,.020)),torch.maximum(yaw*.75,torch.full_like(yaw,.15))),1).clamp(self.lower,self.upper)
  self.deadband=torch.tensor([.003,.005,.025],device=self.device)
  self.error_ema=self.sigma.clone();self.progress_ema=torch.zeros(self.count,device=self.device);self.survival_ema=torch.zeros_like(self.progress_ema);self.success_cycle_ema=torch.zeros_like(self.progress_ema)
  self.failures=torch.zeros(self.count,self.bins,device=self.device);self.exposure=torch.zeros_like(self.failures);self.phase_prob=torch.ones_like(self.failures)/self.bins
  self.terminal_failures=torch.zeros_like(self.failures)
  self.genuine_events=torch.zeros(self.count,4,device=self.device);self.fall_count=torch.zeros(self.count,device=self.device);self.timeout_count=torch.zeros_like(self.fall_count);self.completed_cycles=torch.zeros_like(self.fall_count)
  self.walk_event_scale=torch.full((self.count,),.25,device=self.device)
  n=env.num_envs;self.air=torch.zeros(n,4,device=self.device);self.armed=torch.zeros(n,4,device=self.device,dtype=torch.bool);self.has_stance=self.armed.clone();self.start_xy=torch.zeros(n,4,2,device=self.device)
  self.cycle_events=torch.zeros(n,4,device=self.device);self.cycle_progress=torch.zeros(n,device=self.device);self.cycle_yaw_error=torch.zeros_like(self.cycle_progress);self.cycle_samples=torch.zeros_like(self.cycle_progress);self.last_cycle=torch.zeros(n,device=self.device,dtype=torch.long)
  self.generator=torch.Generator(device=self.device).manual_seed(7519);self.last_observed_step=-1
 def mean_by_teacher(self,value,mask=None):
  n=self.env.training_env_count;typ=self.env.teacher_ids[:n];v=value[:n]
  mask=torch.ones(n,device=self.device) if mask is None else mask[:n].float()
  total=torch.zeros(self.count,device=self.device);den=torch.zeros_like(total);total.index_add_(0,typ,v*mask);den.index_add_(0,typ,mask)
  return total/den.clamp(min=1),den
 def reset(self,ids):
  self.air[ids]=0;self.armed[ids]=False;self.has_stance[ids]=False;self.start_xy[ids]=0;self.cycle_events[ids]=0;self.cycle_progress[ids]=0;self.cycle_yaw_error[ids]=0;self.cycle_samples[ids]=0;self.last_cycle[ids]=0
 def sample_phase(self,ids):
  probabilities=self.phase_prob[self.env.teacher_ids[ids]]
  chosen=torch.multinomial(probabilities,1,replacement=True,generator=self.generator).squeeze(1)
  return (chosen.float()+torch.rand(len(ids),device=self.device,generator=self.generator))/self.bins
 def observe(self,toe_error,velocity_error,yaw_error):
  env=self.env;n=env.training_env_count
  if not n or env.common_step_counter==self.last_observed_step:return
  self.last_observed_step=env.common_step_counter;typ=env.teacher_ids[:n];phase_bin=(env.ref_phase[:n]*self.bins).long().remainder(self.bins)
  self.exposure.view(-1).index_add_(0,typ*self.bins+phase_bin,torch.ones(n,device=self.device))
  errors=torch.stack((toe_error.sqrt(),velocity_error.sqrt(),yaw_error.sqrt()),1)
  for k in range(3):
   avg,_=self.mean_by_teacher(errors[:,k]);self.error_ema[:,k].lerp_(avg,.01)
  cmd=env.teacher_commands[env.teacher_ids];actual=torch.cat((env.heading_linear_velocity[:,:2],env.heading_yaw_rate[:,None]),1);translation=cmd[:,:2].norm(dim=1)>.001
  progress=torch.where(translation,(actual[:,:2]*cmd[:,:2]).sum(1)/cmd[:,:2].square().sum(1).clamp(min=1e-6),actual[:,2]*cmd[:,2]/cmd[:,2].square().clamp(min=1e-6)).clamp(-1,2)
  avg,_=self.mean_by_teacher(progress);self.progress_ema.lerp_(avg,.01)
  contact=env.contact_forces[:,env.feet_indices,2]>1.;flight=(~contact)&(env.foot_clearance>.002)
  xy=env.foot_center_world[:,:,:2];starting=flight&(self.air<=0)&self.has_stance
  self.start_xy=torch.where(starting[:,:,None],xy,self.start_xy);self.air=torch.where(flight,self.air+env.dt,torch.zeros_like(self.air));self.armed|=(self.air>=.06)&self.has_stance
  genuine=contact&self.armed&((xy-self.start_xy).norm(dim=2)>=.005);self.armed&=~contact;self.has_stance|=contact
  quality=getattr(getattr(env,'motion_rewards',None),'step_quality',None)
  if quality is not None:genuine=quality.valid
  self.cycle_events+=genuine.float();self.genuine_events.index_add_(0,typ,genuine[:n].float())
  self.cycle_progress+=progress;self.cycle_yaw_error+=yaw_error.sqrt();self.cycle_samples+=1
  period=env.ref_duration[env.teacher_ids];cycle=(env.episode_length_buf*env.dt/period).long();finished=(cycle>self.last_cycle)&(~env.true_fall)
  moving_ok=self.cycle_progress/self.cycle_samples.clamp(min=1)>.5
  yaw_ok=self.cycle_yaw_error/self.cycle_samples.clamp(min=1)<=torch.maximum(cmd[:,2].abs()*.35,torch.ones_like(period)*.1)
  successful=finished&moving_ok&yaw_ok&(self.cycle_events>=1).all(1)
  survived,number=self.mean_by_teacher(finished.float(),finished|env.true_fall)
  event_mask=number>0;self.survival_ema[event_mask]=.9*self.survival_ema[event_mask]+.1*survived[event_mask]
  success,number=self.mean_by_teacher(successful.float(),finished);mask=number>0;self.success_cycle_ema[mask]=.9*self.success_cycle_ema[mask]+.1*success[mask]
  self.completed_cycles.index_add_(0,typ,finished[:n].float());self.fall_count.index_add_(0,typ,env.true_fall[:n].float());self.timeout_count.index_add_(0,typ,env.time_out_buf[:n].float())
  failed=env.true_fall[:n];ft=typ[failed];fb=phase_bin[failed]
  self.terminal_failures.view(-1).index_add_(0,ft*self.bins+fb,torch.ones_like(fb,dtype=torch.float32))
  for offset,weight in [(0,1.),(1,.8),(2,.5),(3,.25)]:
   self.failures.view(-1).index_add_(0,ft*self.bins+(fb-offset).remainder(self.bins),torch.ones_like(fb,dtype=torch.float32)*weight)
  risk=(self.failures+1)/(self.exposure/self.exposure.mean(1,keepdim=True).clamp(min=1)+1)
  risk=(risk.roll(1,1)+2*risk+risk.roll(-1,1))/4.;risk=risk.pow(.75);risk=risk/risk.sum(1,keepdim=True).clamp(min=1e-6)
  self.phase_prob=self.uniform_mix/self.bins+(1-self.uniform_mix)*risk
  gate=(self.survival_ema>=.85)&(self.success_cycle_ema>=.6)&(self.progress_ema>=.5)
  desired=(1.5*(self.error_ema-self.deadband).clamp(min=0)).clamp(self.lower,self.upper)
  allowed=torch.where((desired<self.sigma)&(~gate[:,None]),self.sigma,desired)
  self.sigma.lerp_(allowed,.003);self.sigma.clamp_(self.lower,self.upper)
  self.walk_event_scale=torch.minimum(torch.ones_like(self.walk_event_scale),self.walk_event_scale+gate.float()*.0002)
  if finished.any():
   self.cycle_events[finished]=0;self.cycle_progress[finished]=0;self.cycle_yaw_error[finished]=0;self.cycle_samples[finished]=0
  self.last_cycle=cycle
 def teacher_state(self,i):
  names=['sigma','error_ema','progress_ema','survival_ema','success_cycle_ema','failures','terminal_failures','exposure','phase_prob','genuine_events','fall_count','timeout_count','completed_cycles','walk_event_scale']
  return {'version':1,'teacher_index':i,'bins':self.bins,'uniform_mix':self.uniform_mix,'lower_bounds':self.lower[i].cpu().clone(),'upper_bounds':self.upper[i].cpu().clone(),'deadbands':self.deadband.cpu().clone(),'values':{n:getattr(self,n)[i].detach().cpu().clone() for n in names},'phase_rng_state':self.generator.get_state().cpu()}
 def load_teacher_state(self,i,state):
  assert state['version']==1 and state['bins']==self.bins
  for name,value in state['values'].items():getattr(self,name)[i].copy_(value.to(self.device))
  self.generator.set_state(state['phase_rng_state'].cpu())
 def report(self,i):
  return {'adaptive_tolerances':dict(zip(['toe_position_m','velocity_m_s','yaw_rad_s'],self.sigma[i].tolist())),
   'adaptive_tolerance_bounds':{'lower':dict(zip(['toe_position_m','velocity_m_s','yaw_rad_s'],self.lower[i].tolist())),'upper':dict(zip(['toe_position_m','velocity_m_s','yaw_rad_s'],self.upper[i].tolist()))},
   'tracking_error_ema':dict(zip(['toe_position_m','velocity_m_s','yaw_rad_s'],self.error_ema[i].tolist())),
   'tracking_deadbands':dict(zip(['toe_position_m','velocity_m_s','yaw_rad_s'],self.deadband.tolist())),
   'tolerance_tightening_gate':{'survival_ema':float(self.survival_ema[i]),'genuine_cycle_success_ema':float(self.success_cycle_ema[i]),'direction_progress_ema':float(self.progress_ema[i])},
   'failure_phase_bins':self.terminal_failures[i].tolist(),'failure_phase_window_weights':self.failures[i].tolist(),'phase_sampling_probabilities':self.phase_prob[i].tolist(),'phase_exposure_samples':self.exposure[i].tolist(),
   'phase_bin_definition':'20 circular bins on reference phase[0,1),failure and3preceding bins weighted;50%uniform mixture;training resets only',
   'genuine_landings_per_foot':self.genuine_events[i].tolist(),'genuine_step_scope':'Contact after>=2mm continuous>=60ms flight,>=5mm capsule-center XY travel, prioractualstance; no reset/timeout asfall',
   'walk_event_weight_scale':float(self.walk_event_scale[i])}

class TeacherRobot(MigrationRobot):
 def _init_buffers(self):
  super()._init_buffers()
  spec=self.cfg.teacher_bank
  self.reference_v2=spec.get('reward_cleanup_profile')==reference_v2.PROFILE
  if self.reference_v2:reference_v2.validate_config(spec)
  self.cadence_cleanup=spec.get('reward_cleanup_profile','legacy')==CADENCE_CLEANUP_PROFILE
  if self.cadence_cleanup or self.reference_v2:
   spec['turn_lift_config']=diagnostic_lift_config(spec.get('turn_lift_config'),[c['id'] for c in spec['clips']])
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
  for i in range(1,self.ref['quat'].shape[1]):
   flip=(self.ref['quat'][:,i-1]*self.ref['quat'][:,i]).sum(-1)<0
   self.ref['quat'][flip,i]*=-1
  self.native_ref_duration=self.ref_duration.clone()
  self.turn_reference_time_scale=float(spec.get('turn_reference_time_scale',1.))
  for i,c in enumerate(spec['clips']):
   if 'TURN' in c['id']:
    self.ref_duration[i]*=self.turn_reference_time_scale
    for key in ('qd','velocity','omega'):self.ref[key][i]/=self.turn_reference_time_scale
  commands=[]
  for c,d in zip(spec['clips'],all_data):
   r=d['root_transform_world'];p=r[-1,:3,3]-r[0,:3,3];yaw0=math.atan2(r[0,1,0],r[0,0,0]);yaws=np.unwrap(np.arctan2(r[:,1,0],r[:,0,0]));period=float(d['time_s'][-1])*self.reference_time_scale;cs,sn=math.cos(yaw0),math.sin(yaw0)
   commands.append([(cs*p[0]+sn*p[1])/period,(-sn*p[0]+cs*p[1])/period,(yaws[-1]-yaws[0])/period])
  self.teacher_commands=torch.tensor(commands,device=self.device,dtype=torch.float32)
  self.raw_reference_commands=self.teacher_commands.clone()
  for i,c in enumerate(spec['clips']):
   if 'TURN' in c['id']:self.raw_reference_commands[i]/=self.turn_reference_time_scale
  for i,c in enumerate(spec['clips']):
   ident=c['id']
   if ident in ('01_FORWARD_POLICY','06_BACKWARD','09_BACKWARD_WALK','10_FORWARD_TROT'):self.teacher_commands[i,1:]=0.
   elif ident in ('07_LATERAL_TROT_LEFT','08_LATERAL_TROT_RIGHT'):self.teacher_commands[i,0]=0.;self.teacher_commands[i,2]=0.
   else:self.teacher_commands[i,:2]=0.
  self.task_tracking_profile=spec.get('task_tracking_profile','legacy')
  self.teacher_commands,self.velocity_tracking_weight,self.yaw_tracking_weight=task_tracking.configure(
      [c['id'] for c in spec['clips']],self.raw_reference_commands,self.native_ref_duration,
      self.task_tracking_profile,bool(spec.get('balance_first',False)))
  self.turn_speed_scale=float(spec.get('turn_speed_scale',1.))
  assert 0<self.turn_speed_scale<=1.
  for i,c in enumerate(spec['clips']):
   if 'TURN' in c['id']:self.teacher_commands[i,2]*=self.turn_speed_scale
  phases=[]
  for c in spec['clips']:
   ident=c['id'];phases.append([.5,.75,.25] if ident=='09_BACKWARD_WALK' else [.5,.25,.75] if ident in ('01_FORWARD_POLICY','04_TURN_LEFT','05_TURN_RIGHT') else [.5,0.,.5] if ident in ('02_STRAFE_LEFT','03_STRAFE_RIGHT') else [.5,.5,0.])
  self.teacher_gait_phases=torch.tensor(phases,device=self.device)
  self.translation_guard=TranslationGuard(self.num_envs,self.device,self.teacher_commands[self.teacher_ids],enabled=bool(spec.get('translation_guard_enabled',False)))
  self.turn_lift_guard=LiftCountGuard(self.num_envs,self.device,self.dt,self.teacher_ids,[c['id'] for c in spec['clips']],spec.get('turn_lift_config'))
  self.turn_offset_rate_rad_s=math.radians(float(spec.get('turn_offset_rate_deg_s',0.)))
  self.event_anchor_indices=torch.tensor([[self.body_names.index(n) for n in names] for names in [('L_shoulder_pitch','R_shoulder_pitch'),('L_hip_pitch','R_hip_pitch')]],device=self.device)
  self.event_frame_indices=torch.tensor([self.body_names.index('ROOT'),self.body_names.index('spine_roll')],device=self.device)
  self.teacher_terms={};self.pre_reset_teacher_obs=torch.zeros(self.num_envs,94,device=self.device)
  self.pre_reset_teacher_state={}
  self.reference_initializations=0
  self.teacher_control_mode=spec.get('control_mode','direct_full_target')
  self.reference_strength=float(spec.get('reference_strength',1.))
  self.imitation_scale=float(spec.get('imitation_scale',1.))
  self.balance_first=bool(spec.get('balance_first',False))
  self.offset_limit_rad=math.radians(float(spec.get('offset_limit_deg',10.)))
  self.current_residual_action=torch.zeros(self.num_envs,15,device=self.device)
  self.current_reference_target=self.default_dof_pos[:,self.active].expand(self.num_envs,-1).clone()
  self.pace_stability=PaceStability(self,bool(spec.get("pace_stability_enabled",False)))
  self.motion_rewards=None
  if spec.get('motion_reward_module_enabled',False):
   if not hasattr(self,'contact_foot_com'):
    properties=self.gym.get_actor_rigid_body_properties(self.envs[0],self.actor_handles[0])
    self.contact_foot_com=torch.tensor([[properties[i].com.x,properties[i].com.y,properties[i].com.z] for i in self.feet_indices.tolist()],device=self.device)
   from teacher_motion_rewards import TeacherMotionRewards
   self.motion_rewards=TeacherMotionRewards(self);self.teacher_commands.copy_(self.motion_rewards.teacher_commands)
  self.adaptive=AdaptiveTeacherCurriculum(self) if spec.get('adaptive_enabled',False) else None
  if self.reference_v2:
   assert ACTION_NAMES[-1]=='spine_pitch'
   assert torch.allclose(self.ref['q'][:,:,-1],self.ref['q'][0,0,-1].expand_as(self.ref['q'][:,:,-1]),atol=1e-6)
   reference_v2.initialize_env(self)
 def ref_sample(self,phase=None,ids=None):
  ids=torch.arange(self.num_envs,device=self.device) if ids is None else ids
  p=self.ref_phase[ids] if phase is None else phase
  typ=self.teacher_ids[ids];segments=self.ref['q'].shape[1]-1;x=p.clamp(0,1)*segments;i=x.long().clamp(0,segments-1);w=x-i
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
   if hasattr(self,'reference_rewards_v2'):
    train_mask=ids<self.training_env_count;phase[train_mask]=reference_v2.sample_phase(self,ids[train_mask])
   if getattr(self,'adaptive',None) is not None:
    train_mask=ids<self.training_env_count;phase[train_mask]=self.adaptive.sample_phase(ids[train_mask])
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
   if getattr(self,'adaptive',None) is not None:self.adaptive.reset(ids)
   if getattr(self,'motion_rewards',None) is not None:self.motion_rewards.reset(ids)
   self.turn_lift_guard.reset(ids)
   self.pace_stability.reset(ids)
   if hasattr(self,'reference_rewards_v2'):self.reference_rewards_v2.reset(ids)
   if hasattr(self,'turn_cycles'):self.turn_cycles.reset(ids)
   if hasattr(self,'turn_boundary'):self.turn_boundary.reset(ids)
 def _reset_dofs(self,ids):
  if not hasattr(self,'ref'):return super()._reset_dofs(ids)
  ref=self.ref_sample(ids=ids);self.dof_pos[ids]=self.default_dof_pos;self.dof_pos[ids[:,None],self.active]=ref['q'];self.dof_vel[ids]=0.;self.dof_vel[ids[:,None],self.active]=ref['qd']
  self.dof_pos[ids,self.dependent]=-self.dof_pos[ids,self.hip];self.dof_vel[ids,self.dependent]=-self.dof_vel[ids,self.hip]
  actor=self.robot_actor_indices[ids];self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(actor),len(ids));self.reference_initializations+=len(ids)
 def _reset_root_states(self,ids):
  if not hasattr(self,'ref'):return super()._reset_root_states(ids)
  ref=self.ref_sample(ids=ids);self.root_states[ids]=self.base_init_state
  self.root_states[ids,:3]=ref['root'][:,:3,3]+self.env_origins[ids];self.root_states[ids,3:7]=ref['quat'];self.root_states[ids,7:10]=ref['velocity'];self.root_states[ids,10:13]=ref['omega']
  if hasattr(self,'turn_curriculum'):self.root_states[ids,12]=self.teacher_commands[self.teacher_ids[ids],2]
  if hasattr(self,'translation_guard'):self.translation_guard.reset(ids,self.root_states)
  actor=self.robot_actor_indices[ids];self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_all),gymtorch.unwrap_tensor(actor),len(ids))
 def teacher_observations(self):
  self._refresh_proprio_state();ref=self.ref_sample()
  q=(ref['q']-self.default_dof_pos[:,self.active]);phase=self.ref_phase*2*math.pi
  quality=getattr(getattr(self,'motion_rewards',None),'step_quality',None)
  integral=quality.yaw_integral.observation() if quality is not None else torch.zeros_like(phase)
  path_observation=self.translation_guard.observations(self.root_states)
  events=self.turn_lift_guard.observations(self.episode_length_buf*self.dt)
  if hasattr(self,'reference_rewards_v2'):
   if hasattr(self,'turn_cycles'):
    c=self.turn_cycles;turn=self.teacher_commands[self.teacher_ids,2].abs()>.001;goal=(self.teacher_commands[self.teacher_ids,2].abs()*self.ref_duration[self.teacher_ids]).clamp(min=.01)
    path_observation=torch.where(turn[:,None],torch.stack((c.net/goal,c.max_drift/.075),1).clamp(-3,3),path_observation)
   engine=self.reference_rewards_v2;integral=(engine.yaw_integral/math.radians(10)).clamp(-5,5)
   events=torch.cat((engine.air/engine.swing_time.clamp(min=.04),engine.cycle_count.mean(1,keepdim=True)),1).clamp(max=5)
  return torch.cat((self._proprio(),self.base_lin_vel*2.,torch.sin(phase)[:,None],torch.cos(phase)[:,None],q,ref['qd']*.05,integral[:,None],path_observation,events),-1)
 def step(self,actions):
  previous_offset=(self.executed_target-self.ref_sample()['q']).clamp(-self.offset_limit_rad,self.offset_limit_rad)
  self.ref_phase=torch.remainder(self.ref_phase+self.dt/self.ref_duration[self.teacher_ids],1.)
  self.current_residual_action=actions.clone()
  if self.teacher_control_mode=='reference_plus_residual':
   self.current_reference_target=self.ref_sample()['q']
   default=self.default_dof_pos[:,self.active]
   self.current_offset_rad=(.35*actions).clamp(-self.offset_limit_rad,self.offset_limit_rad)
   self.current_offset_rad[:,-1]=0.;self.current_residual_action[:,-1]=0.  # spine_pitch held at constant authored angle
   if self.turn_offset_rate_rad_s>0:
    self.current_offset_rad=limit_turn_offset(previous_offset,self.current_offset_rad,self.turn_offset_rate_rad_s*self.dt,
        self.teacher_commands[self.teacher_ids,2].abs()>.001,self.episode_length_buf==0)
   self.current_offset_rad=self.pace_stability.filter(previous_offset,self.current_offset_rad,self.episode_length_buf==0)
   full=self.current_reference_target+self.current_offset_rad
   actions=(full-default)/.35
   self.reference_execution_calls+=1
  return super().step(actions)
 def check_termination(self):
  if hasattr(self,'reference_rewards_v2'):
   reference_v2.check_env(self);return
  super().check_termination()
  if not hasattr(self,'ref'):return
  task_failure=self.translation_guard.evaluate(self.root_states)
  self.reset_buf|=task_failure;self.true_fall|=task_failure;self.time_out_buf&=~task_failure
  ref=self.ref_sample();q=self.dof_pos[:,self.active];qd=self.dof_vel[:,self.active]
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
  self.reference_endpoint_error_vector_m=actual_local-local
  self.reference_endpoint_error_m=self.reference_endpoint_error_vector_m.norm(dim=-1)
  lift_failure=self.turn_lift_guard.update(actual_toe[:,:,2],self.episode_length_buf*self.dt)
  self.reset_buf|=lift_failure;self.true_fall|=lift_failure;self.time_out_buf&=~lift_failure
  self.pre_reset_teacher_obs=self.teacher_observations().clone()
  anchor=self.rigid_states[:,self.event_anchor_indices,:3].mean(2)
  self.event_local_xy=relative_xy(self.foot_contact_point,anchor,self.rigid_states[:,self.event_frame_indices,3:7])
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
  sigma_toe=torch.full_like(ve,.010)
  if self.adaptive is not None:
   if self.cfg.teacher_bank.get('step_quality_profile','legacy')=='legacy':self.adaptive.observe(toe_error,ve,we)
   sigma=self.adaptive.sigma[self.teacher_ids];sigma_toe,sigma_v,sigma_yaw=sigma[:,0],sigma[:,1],sigma[:,2]
  q_excess=(q-ref['q']).abs().sub(.05).clamp(min=0).square().mean(1);qd_excess=(qd-ref['qd']).abs().sub(.3).clamp(min=0).square().mean(1)
  toe_excess=(actual_local-local).norm(dim=-1).sub(.003).clamp(min=0).square().mean(1)
  clearance_excess=(self.foot_clearance-ref['surface']).abs().sub(.002).clamp(min=0).square().mean(1)
  terms={'q_imitation':8*torch.exp(-q_excess/.02),'qd_imitation':torch.exp(-qd_excess/12.),
         'foot_imitation':4*torch.exp(-toe_excess/sigma_toe.square()),'foot_clearance':2*torch.exp(-clearance_excess/sigma_toe.square()),'velocity':3*torch.exp(-ve.sqrt().sub(.005).clamp(min=0).square()/sigma_v.square()),
         'yaw':2*torch.exp(-we.sqrt().sub(.025).clamp(min=0).square()/sigma_yaw.square()),'orientation':2*torch.exp(-ge/.1),'height':2*torch.exp(-he/.0016),
         'alive':torch.ones_like(qe),'slip':-.5*slip,'torque':-.00005*self.torques.square().sum(1),
         'action_rate':-.01*(self.actions-self.last_actions).square().mean(1)}
  for key in ('q_imitation','qd_imitation','foot_imitation','foot_clearance'):terms[key]*=self.imitation_scale
  if self.balance_first:
   terms['velocity']*=.5;terms['yaw']*=.5;terms['alive']*=5.
  terms.update(task_tracking.tracking_terms(v[:,:2],self.heading_yaw_rate,cmd,sigma_v,sigma_yaw,
      self.velocity_tracking_weight[self.teacher_ids],self.yaw_tracking_weight[self.teacher_ids]))
  if self.teacher_control_mode=='reference_plus_residual':terms['offset_magnitude']=-.5*(self.current_offset_rad/self.offset_limit_rad).square().mean(1)
  if self.motion_rewards is not None:
   if self.adaptive is not None:self.motion_rewards.set_walk_event_scale(self.adaptive.walk_event_scale)
   terms=self.motion_rewards.apply(terms,ref)
   if self.adaptive is not None and self.motion_rewards.step_quality is not None:self.adaptive.observe(toe_error,ve,we)
  if self.pace_stability.enabled:
   terms=self.pace_stability.apply(terms,self.reference_endpoint_error_vector_m,self.executed_target-self.current_reference_target)
  extra_fall=(8. if self.balance_first else 3.)*self.true_fall.float()
  if self.motion_rewards is not None:extra_fall*=~self.motion_rewards.walk
  terms['nonwalk_fall']=-extra_fall/self.dt
  if self.cadence_cleanup:terms=cleanup_cadence_terms(terms)
  reward=sum(terms.values())*self.dt
  self.teacher_terms=terms;self.teacher_reward=reward
  self.pre_reset_teacher_state={'q_rmse':qe.sqrt().clone(),'xy_error':ve.sqrt().clone(),'yaw_error':we.sqrt().clone(),
    'actual_velocity':torch.cat((v[:,:2],self.heading_yaw_rate[:,None]),-1).clone(),
    'q':q.clone(),'root':self.root_states.clone(),'toe':actual_toe.clone(),
    'reference_toe':(self.root_states[:,:3,None].transpose(1,2)+quat_apply(self.base_quat[:,None,:].expand(-1,4,-1),local)).clone(),'contact_force':self.contact_forces[:,self.feet_indices,2].clone(),
    'contact':contact.clone(),'phase':self.ref_phase.clone(),'fall':self.true_fall.clone(),
    'executed_target_full':self.executed_target.clone(),'residual_action':self.current_residual_action.clone(),'reference_joint_target':self.current_reference_target.clone(),
    'reference_endpoint_error_vector_m':self.reference_endpoint_error_vector_m.clone(),
    'offset_joint_rad':self.current_offset_rad.clone() if self.teacher_control_mode=='reference_plus_residual' else torch.zeros_like(q)}
  tilt=(self.projected_gravity*self.reference_gravity).sum(1)
  terminal_contact=torch.any(self.contact_forces[:,self.termination_contact_indices].norm(dim=-1)>2,dim=1)
  coupling=(self.dof_pos[:,self.hip]+self.dof_pos[:,self.dependent]).abs()
  waist_error=(self.dof_pos[:,self.waist]+math.radians(40)).abs()
  self.pre_reset_teacher_state['fall_diagnostics']={'tilt_dot':tilt.clone(),'root_z_m':self.root_states[:,2].clone(),'coupling_error_rad':coupling.clone(),'waist_error_rad':waist_error.clone(),
   'tilt_failure':(tilt<.55),'height_failure':(self.root_states[:,2]<.055),'termination_contact_failure':terminal_contact,'coupling_failure':(coupling>.08),'waist_failure':(waist_error>.6),'knee_ground_failure':self.knee_ground_contact.any(1)&bool(self.cfg.termination.knee_contact_enabled)}
  self.pre_reset_teacher_state['fall_diagnostics'].update(task_heading_drift_failure=self.translation_guard.heading_failure.clone(),task_cross_track_failure=self.translation_guard.cross_failure.clone(),task_heading_drift_deg=torch.rad2deg(self.translation_guard.heading_error).clone(),task_cross_track_cm=(100*self.translation_guard.cross_track).clone())
  self.pre_reset_teacher_state['fall_diagnostics'].update(direct_lift_frequency_failure=self.turn_lift_guard.violation.clone(),direct_lift_interval_failure=self.turn_lift_guard.interval_failure.clone(),direct_lift_window_failure=self.turn_lift_guard.window_failure.clone(),direct_lift_total_hz=(self.turn_lift_guard.count.sum(1)/self.turn_lift_guard.window).clone())
  if self.motion_rewards is not None:self.pre_reset_teacher_state['motion_diagnostics']=self.motion_rewards.diagnostics
 def compute_reward(self):
  if hasattr(self,'teacher_reward'):self.rew_buf.copy_(self.teacher_reward)
  else:super().compute_reward()

def create_env(cfg):
 import migration_env
 original=migration_env.MigrationRobot;migration_env.MigrationRobot=TeacherRobot
 try:return create_migration_env(cfg,noise=False)
 finally:migration_env.MigrationRobot=original

def update(policy,opt,storage,last_value,minibatches=2):
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
  for ids in perm.chunk(minibatches):
   dist=policy.distribution(obs[ids]);log=dist.log_prob(actions[ids])[...,:14].sum(-1);ratio=(log-oldlog[ids]).exp();a=advantages[ids]
   surrogate=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean();v=policy.value(obs[ids]);vl=(v-target[ids]).square().mean()
   loss=surrogate+vl*.5-.002*dist.entropy()[...,:14].sum(-1).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step();policy.log_std.data.clamp_(math.log(getattr(policy,'minimum_std',.02)),math.log(getattr(policy,'maximum_std',.3)));losses.append([float(vl),float(surrogate)])
 return np.mean(losses,axis=0).tolist()

def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--duration',type=float,default=1200);p.add_argument('--deadline');p.add_argument('--resume-bank',type=Path);p.add_argument('--minibatches',type=int,default=2);p.add_argument('--control-mode',choices=['direct_full_target','reference_plus_residual'],default='reference_plus_residual');p.add_argument('--reference-strength',type=float,default=1.);p.add_argument('--time-scale',type=float,default=1.);p.add_argument('--offset-limit-deg',type=float,default=10.);p.add_argument('--imitation-scale',type=float,default=.5);p.add_argument('--envs-per-teacher',type=int,default=256);p.add_argument('--smoke',action='store_true');p.add_argument('--balance-first',action='store_true');p.add_argument('--exploration-std',type=float,default=.08);p.add_argument('--minimum-action-std',type=float,default=.02);p.add_argument('--allow-objective-change',action='store_true');p.add_argument('--adaptive-curriculum',action='store_true');p.add_argument('--motion-reward-module',action='store_true');p.add_argument('--task-tracking-profile',choices=['legacy',task_tracking.PROFILE],default='legacy');p.add_argument('--step-quality-profile',choices=['legacy','strict_steps_v1'],default='legacy');p.add_argument('--translation-guard',action='store_true');p.add_argument('--turn-speed-scale',type=float,default=1.);p.add_argument('--turn-reference-time-scale',type=float,default=1.);p.add_argument('--turn-offset-rate-deg-s',type=float,default=0.);p.add_argument('--turn-lift-config',type=Path);p.add_argument('--pace-stability',action='store_true');p.add_argument('--reward-cleanup-profile',choices=['legacy',CADENCE_CLEANUP_PROFILE,reference_v2.PROFILE],default=reference_v2.PROFILE);a=p.parse_args()
 if a.control_mode=='reference_plus_residual' and (a.reference_strength!=1. or a.time_scale!=1.):raise ValueError('Authored reference amplitude and timing must remain1; teacher learns bounded joint offsets only')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 torch.set_num_threads(2);torch.manual_seed(741);np.random.seed(741)
 ids=['01_FORWARD_POLICY','04_TURN_LEFT','05_TURN_RIGHT','06_BACKWARD','02_STRAFE_LEFT','03_STRAFE_RIGHT','09_BACKWARD_WALK','10_FORWARD_TROT']
 manifest=json.loads((a.reference/'manifest.json').read_text());clips=[next(c for c in manifest['clips'] if c['id']==ident) for ident in ids]
 a.bank.mkdir(parents=True,exist_ok=True);project=Path(os.environ['SABLE_GYM_ROOT'])
 base_config=Path(__file__).resolve().parent/'bootstrap'
 if not (base_config/'config.json').exists():base_config=project/'runs/stand5_20261027_v44_stride80_all'
 cfg=student_config(base_config,stage='S3',num_envs=len(ids)*(a.envs_per_teacher+4))
 cfg.migration.freeze_waist=False;cfg.migration.moving_start_training=False;cfg.migration.command_frontier_alpha=1.;cfg.migration.history_noise_level=0.
 cfg.commands.resampling_time=60.;cfg.env.episode_length_s=24.
 for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
 cfg.rewards.scales.alive=1.
 cfg.teacher_bank={'clips':clips,'train_per':a.envs_per_teacher,'eval_per':4,'reference':str(a.reference),'control_mode':a.control_mode,'reference_strength':a.reference_strength,'time_scale':a.time_scale,'imitation_scale':a.imitation_scale,'offset_limit_deg':a.offset_limit_deg,'balance_first':a.balance_first,'adaptive_enabled':a.adaptive_curriculum,'motion_reward_module_enabled':a.motion_reward_module,'walk_event_weight_scale':.25,'task_tracking_profile':a.task_tracking_profile,'step_quality_profile':a.step_quality_profile,'translation_guard_enabled':a.translation_guard,'turn_speed_scale':a.turn_speed_scale,'turn_reference_time_scale':a.turn_reference_time_scale,'turn_offset_rate_deg_s':a.turn_offset_rate_deg_s,'turn_lift_config':json.loads(a.turn_lift_config.read_text()) if a.turn_lift_config else None,'pace_stability_enabled':a.pace_stability}
 cfg.teacher_bank['reward_cleanup_profile']=a.reward_cleanup_profile
 env=create_env(cfg);env.reset();policies=[Policy().to(env.device) for _ in ids];opts=[torch.optim.Adam(x.parameters(),lr=3e-4) for x in policies]
 for policy in policies:policy.minimum_std=a.minimum_action_std;policy.maximum_std=.6;policy.log_std.data.fill_(math.log(a.exploration_std))
 resume_iteration=0;resume_sources=[];fresh_teacher_ids=[];pace_origin=None
 if a.resume_bank:
  source=a.resume_bank.resolve();assert source.parent==project/'runs' and source.name.startswith('teacher_bank_fullref_')
  assert source!=a.bank.resolve(),'Refuse to overwrite source run'
  for i,ident in enumerate(ids):
   if ident in ('02_STRAFE_LEFT','03_STRAFE_RIGHT') and not (source/ident/'checkpoint.json').exists():
    with torch.no_grad():policies[i].log_std.fill_(math.log(a.exploration_std))
    assert not policies[i].actor[-1].weight.any() and not policies[i].actor[-1].bias.any() and not opts[i].state
    fresh_teacher_ids.append(ident)
    continue
   receipt=json.loads((source/ident/'checkpoint.json').read_text(encoding='utf8'));cp=Path(receipt['path']).resolve()
   assert source/ident in cp.parents and sha(cp)==receipt['sha256']
   parent=torch.load(str(cp),map_location=env.device)
   if parent.get('reward_cleanup_profile','legacy')!=a.reward_cleanup_profile:
    raise ValueError('Reward cleanup changed the objective. Start fresh teachers; do not resume a legacy checkpoint/optimizer with this entrypoint.')
   assert parent['teacher_id']==ident and parent['teacher_control_mode']=='reference_plus_residual'
   assert parent['reference_strength']==1. and parent['reference_time_scale']==1.
   if not a.allow_objective_change:
    assert parent['offset_limit_deg']==a.offset_limit_deg and parent.get('imitation_reward_scale')==a.imitation_scale
   assert parent['reference_sha256']==sha(a.reference/(ident+'.npz')) and parent['action_names']==list(ACTION_NAMES)
   parent,observation_migration=expand_checkpoint(parent,[name for name,_ in policies[i].named_parameters()])
   policies[i].load_state_dict(parent['model_state_dict'],strict=True);opts[i].load_state_dict(parent['optimizer_state_dict'])
   assert state_equal(policies[i].state_dict(),parent['model_state_dict']) and state_equal(opts[i].state_dict(),parent['optimizer_state_dict'])
   intentional_change=None
   if a.allow_objective_change:
    with torch.no_grad():policies[i].log_std.fill_(math.log(a.exploration_std))
    retained={k:v for k,v in policies[i].state_dict().items() if k!='log_std'};old_retained={k:v for k,v in parent['model_state_dict'].items() if k!='log_std'}
    assert state_equal(retained,old_retained) and state_equal(opts[i].state_dict(),parent['optimizer_state_dict'])
    intentional_change={'parameter':'log_std','new_normalized_std':a.exploration_std,'new_physical_std_deg':math.degrees(.35*a.exploration_std),'minimum_normalized_std':a.minimum_action_std,'offset_limit_deg':a.offset_limit_deg,'imitation_scale':a.imitation_scale,'objective':'balance_first' if a.balance_first else 'unchanged'}
   saved_origin=parent.get('pace_stability_origin_iteration')
   if saved_origin is not None:
    assert pace_origin is None or pace_origin==saved_origin
    pace_origin=saved_origin
   if env.reference_v2:
    env.turn_curriculum.load(i,parent.get('turn_curriculum'))
    env.turn_boundary.load(i,parent.get('turn_boundary'))
   course=parent.get('curriculum_state')
   curriculum_command_change=None
   if course is not None and env.adaptive is not None:
    env.adaptive.load_teacher_state(i,course)
    curriculum_command_change=task_tracking.migrate_curriculum_command(env.adaptive,i,parent.get('training_command',env.teacher_commands[i].tolist()),env.teacher_commands[i])
   resume_sources.append({'teacher_id':ident,'path':str(cp),'sha256':receipt['sha256'],'iteration':parent['iteration'],'policy_equal_before_intentional_change':True,'policy_equal_except_log_std_after_change':True,'optimizer_equal':True,'intentional_change':intentional_change,'observation_migration':observation_migration,'curriculum_command_change':curriculum_command_change,'curriculum_state_restored':course is not None and env.adaptive is not None})
  assert len(set(x['iteration'] for x in resume_sources))==1,'Refuse to mix teacher checkpoint iterations'
  resume_iteration=resume_sources[0]['iteration']
  atomic(a.bank/'resume_verification.json',{'passed':True,'source_bank':str(source),'sources':resume_sources,'initial_iteration':resume_iteration,'new_environments_reset':True,'old_rollout_storage_reused':False})
  if env.reference_v2:env.turn_curriculum.apply(env)
  env.reset()
 pace_origin=resume_iteration if pace_origin is None else pace_origin
 env.pace_stability.set_strength(min(1.,max(0.,(resume_iteration-pace_origin)/60.)))
 start=time.time();deadline=datetime.datetime.fromisoformat(a.deadline).timestamp() if a.deadline else start+a.duration;assert deadline>start;iteration=resume_iteration;last_save=0.;eval_state=None;last_metrics=[]
 for ident in ids:(a.bank/ident/'checkpoints').mkdir(parents=True,exist_ok=True)
 bank={'status':'smoke' if a.smoke else 'training','run_dir':str(a.bank),'pid':os.getpid(),'started_unix':start,'deadline':datetime.datetime.fromtimestamp(deadline,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'updated_unix':start,'num_envs':env.num_envs,'training_envs':env.training_env_count,'reserved_evaluation_envs':32,'initialization':'random_no_checkpoint_loaded_zero_mean_offset','reward_cleanup_profile':a.reward_cleanup_profile,'removed_cadence_terms':list(REMOVED_CADENCE_TERMS) if a.reward_cleanup_profile==CADENCE_CLEANUP_PROFILE else [],'teacher_control_mode':a.control_mode,'reference_strength':a.reference_strength,'reference_time_scale':a.time_scale,'offset_limit_deg':a.offset_limit_deg,'imitation_reward_scale':a.imitation_scale,'reference_sampling':'Pre-action observation uses phase_t; env advances phase by dt before composing full q_ref(phase_t+dt)+bounded offset. q_default is only the lower PD-coordinate conversion origin.','trace_is_BC_dataset':False,'student_distillation_label':'Collect pre-action326D observation -> complete clipped executed_target_full, never residual_action alone','teachers':[{'id':c['id'],'title':c.get('title',c['id']),'run_dir':str(a.bank/c['id']),'reference_metadata':str(a.reference/(c['id']+'.json')),'status':'training'} for c in clips]}
 atomic(a.bank/'teacher_bank.json',bank)
 bank.update(initial_iteration=resume_iteration,resume_start_iteration=resume_iteration,resume_from_bank=str(a.resume_bank.resolve()) if a.resume_bank else None)
 bank['ppo_minibatches']=a.minibatches
 bank.update(pace_stability_enabled=a.pace_stability,pace_stability_metadata=PACE_METADATA if a.pace_stability else None,pace_stability_origin_iteration=pace_origin)
 bank.update(training_objective='balance_first' if a.balance_first else 'reference_and_task_tracking',exploration_normalized_std_initial=a.exploration_std,minimum_action_std=a.minimum_action_std,exploration_physical_std_deg_initial=math.degrees(.35*a.exploration_std),intentional_std_change_on_resume=a.allow_objective_change)
 bank.update(turn_slew_definition='Turn residual offset changes by at most turn_offset_rate_deg_s*dt, inferred from previous actually executed target and previous reference; first reset command warm-starts the existing balance correction',turn_reference_time_scale=a.turn_reference_time_scale,turn_offset_rate_deg_s=a.turn_offset_rate_deg_s,turn_lift_config=cfg.teacher_bank['turn_lift_config'],teacher_reference_periods_s=env.ref_duration.tolist(),teacher_observations=94,added_observation='original89D plus four per-foot cooldown availability values and remaining total direct-lift budget; zero for translations',turn_speed_scale=a.turn_speed_scale,translation_guard_enabled=a.translation_guard,translation_guard_metadata=GUARD_METADATA if a.translation_guard else None,step_quality_profile=a.step_quality_profile,task_tracking_profile=a.task_tracking_profile,task_target_scope='User-requested task targets; hand-K joint amplitude/timing and raw root motion preserved',reference_guidance='loose_tracking',adaptive_tolerances_enabled=a.adaptive_curriculum,motion_reward_module_enabled=a.motion_reward_module,walk_event_weight_scale=.25,
  raw_reference_commands=env.raw_reference_commands.tolist(),training_commands=env.teacher_commands.tolist(),reward_metadata=env.motion_rewards.metadata if env.motion_rewards is not None else None,
  policy_resume_from=str(a.resume_bank) if a.resume_bank else None,reward_profile_source='v44-9470 reward mechanisms only, forward/backward direction-adapted' if a.motion_reward_module else None)
 if a.resume_bank:bank['initialization']='six_current_teachers_resumed_with_Adam_two_fresh_zero_mean_offset_pace_teachers' if fresh_teacher_ids else 'all_eight_current_teachers_resumed_with_Adam';bank['fresh_teacher_ids']=fresh_teacher_ids;bank['gait_replacement']={'02_STRAFE_LEFT':'07_LATERAL_TROT_LEFT','03_STRAFE_RIGHT':'08_LATERAL_TROT_RIGHT'}
 if env.reference_v2:bank.update(waist_pitch_locked=True,waist_pitch_target_deg=float(env.ref['q'][0,0,-1])*180/math.pi,exploration_active_joints=14,reward_metadata=reference_v2.METADATA,added_observation='yaw integral + contact swing ages and cycle counts; no lift budget',task_target_scope='explicit source commands; body-relative end effectors; no world reference XY/yaw tracking')
 atomic(a.bank/'teacher_bank.json',bank)
 if not a.smoke:atomic(project/'results/teacher_bank/current.json',{'bank':str(a.bank),'deployment':str(Path(__file__).resolve().parent),'status':bank['status']})
 def save(reason):
  receipts=[]
  for i,ident in enumerate(ids):
   out=a.bank/ident;cp=out/'checkpoints'/('model_%06d.pt'%iteration)
   if not cp.exists():torch.save({'model_state_dict':policies[i].state_dict(),'optimizer_state_dict':opts[i].state_dict(),'iteration':iteration,'teacher_id':ident,'initialization':'fresh_random_pace_zero_mean_offset' if ident in fresh_teacher_ids else 'current_teacher_policy_and_Adam_resumed' if a.resume_bank else 'random','observations':94,'actions':15,'action_names':list(ACTION_NAMES),'reward_cleanup_profile':a.reward_cleanup_profile,'removed_cadence_terms':list(REMOVED_CADENCE_TERMS) if a.reward_cleanup_profile==CADENCE_CLEANUP_PROFILE else [],'teacher_control_mode':a.control_mode,'reference_strength':a.reference_strength,'reference_time_scale':a.time_scale,'offset_limit_deg':a.offset_limit_deg,'imitation_reward_scale':a.imitation_scale,'minimum_action_std':a.minimum_action_std,'training_objective':'balance_first' if a.balance_first else 'reference_and_task_tracking','adaptive_tolerances_enabled':a.adaptive_curriculum,'motion_reward_module_enabled':a.motion_reward_module,'walk_event_weight_scale':float(env.adaptive.walk_event_scale[i]) if env.adaptive is not None else .25,'curriculum_state':env.adaptive.teacher_state(i) if env.adaptive is not None else None,'motion_reward_metadata':env.motion_rewards.metadata if env.motion_rewards is not None else None,'training_command':env.teacher_commands[i].tolist(),'raw_reference_command':env.raw_reference_commands[i].tolist(),'task_tracking_profile':a.task_tracking_profile,'step_quality_profile':a.step_quality_profile,'action_definition':'q_target=clip(q_ref(next_phase)+clip(.35*offset_action,+/-offset_limit))' if a.control_mode=='reference_plus_residual' else 'q_target=clip(q_default+.35*action)','student_supervision':'pre-action326 observation -> complete clipped executed target, not residual alone; trace is visualization not BC data','reference':str(a.reference/(ident+'.npz')),'reference_sha256':sha(a.reference/(ident+'.npz')),'pace_stability_origin_iteration':pace_origin if a.pace_stability else None,'pace_stability_strength':float(env.pace_stability.strength[i*a.envs_per_teacher]) if a.pace_stability else None,'turn_boundary':env.turn_boundary.state(i) if env.reference_v2 else None,'turn_curriculum':env.turn_curriculum.state(i) if env.reference_v2 else None,'reason':reason},str(cp))
   receipt={'path':str(cp),'sha256':sha(cp),'iteration':iteration,'updated_unix':time.time(),'reason':reason};atomic(out/'checkpoint.json',receipt);receipts.append(receipt)
  return receipts
 receipts=save('initial')
 evaluation_history=[[] for _ in ids]
 eval_policies=[Policy().to(env.device) for _ in ids]
 def begin_eval(receipts):
  nonlocal eval_state
  for e,t in zip(eval_policies,policies):e.load_state_dict(t.state_dict())
  env.eval_reset_phase=0.;env.reset_idx(env.eval_ids);env._refresh_proprio_state();env.compute_observations();env.eval_reset_phase=None
  generator=torch.Generator(device=env.device).manual_seed(947 if env.reference_v2 else 947+iteration)
  perturb=(torch.rand((32,15),device=env.device,generator=generator)-.5)*.01
  env.dof_pos[env.eval_ids[:,None],env.active]+=perturb
  env.dof_pos[env.eval_ids,env.dependent]=-env.dof_pos[env.eval_ids,env.hip]
  actor=env.robot_actor_indices[env.eval_ids];env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(actor),32)
  env._refresh_proprio_state();env.compute_observations()
  env.pace_stability.set_strength(float(env.pace_stability.strength[0]),slice(env.training_env_count,env.num_envs))
  eval_state={'step':0,'alive':torch.ones(32,device=env.device,dtype=torch.bool),'count':torch.zeros(32,device=env.device),'q':torch.zeros(32,device=env.device),'xy':torch.zeros(32,device=env.device),'yaw':torch.zeros(32,device=env.device),'actual':torch.zeros(32,3,device=env.device),'steps':torch.zeros(32,4,device=env.device),'last_contact':torch.zeros(32,4,device=env.device,dtype=torch.bool),'trace':[[] for _ in ids],'receipts':receipts,'iteration':iteration,'first_fall_diagnostics':[None]*32,'pace_dense':[[] for _ in ids],'pace_strength':float(env.pace_stability.strength[0])}
  if env.reference_v2:eval_state.update(turn_cycle_sums=torch.zeros(32,8,device=env.device),contact_agreement=torch.zeros(32,device=env.device),gravity_square=torch.zeros(32,device=env.device),foot_square=torch.zeros(32,4,device=env.device),liftoffs=torch.zeros(32,4,device=env.device),slip_square=torch.zeros(32,device=env.device),saturated=torch.zeros(32,device=env.device))
 def finish_eval():
  nonlocal eval_state
  es=eval_state
  for i,ident in enumerate(ids):
   sl=slice(i*4,(i+1)*4);n=es['count'][sl].clamp(min=1)
   report={'teacher_id':ident,'iteration':es['iteration'],'checkpoint_iteration':es['iteration'],'checkpoint':es['receipts'][i],
     'independent':True,'evaluation_seed':947 if env.reference_v2 else 947+es['iteration'],'reset':'reference_phase0 plus seeded uniform joint perturbation +/-0.005rad; reserved environments never used by optimizer','horizon_s':16.,'num_envs':4,
     'survival_fraction':float(es['alive'][sl].float().mean()),'q_tracking_rmse_rad':float((es['q'][sl].sum()/n.sum()).sqrt()),'xy_rmse_m_s':float((es['xy'][sl].sum()/n.sum()).sqrt()),'yaw_rmse_rad_s':float((es['yaw'][sl].sum()/n.sum()).sqrt()),'mean_command':env.teacher_commands[i].tolist(),'mean_actual':(es['actual'][sl]/n[:,None]).mean(0).tolist(),'steps_per_foot':es['steps'][sl].sum(0).tolist(),'step_scope':'Observed normal-force contact onset; not yet genuine-air-step acceptance','samples':int(n.sum()),'accepted':False,'first_fall_diagnostics':es['first_fall_diagnostics'][i*4:(i+1)*4],'reward_cleanup_profile':a.reward_cleanup_profile,'removed_cadence_terms':list(REMOVED_CADENCE_TERMS) if a.reward_cleanup_profile==CADENCE_CLEANUP_PROFILE else [],'teacher_control_mode':a.control_mode,'reference_strength':a.reference_strength,'reference_time_scale':a.time_scale,'created_unix':time.time()}
   if env.pace_stability.enabled and ident in ('02_STRAFE_LEFT','03_STRAFE_RIGHT'):
    from teacher_pace_stability import dense_report,acceptance
    report['executed_simulation_s']=es['step']*env.dt
    report['pace_dense_diagnostics']=dense_report(es['pace_dense'][i],env.dt,es['pace_strength'])
    report['pace_acceptance']=acceptance(report,report['pace_dense_diagnostics'])
    report['accepted']=report['pace_acceptance']['passed']
   if env.reference_v2:
    from teacher_reference_evaluation_v2 import summarize
    report['contact_phase_agreement']=float(es['contact_agreement'][sl].sum()/n.sum())
    report['gravity_rmse_deg']=math.degrees(float((es['gravity_square'][sl].sum()/n.sum()).sqrt()))
    report.update(summarize(report,es['foot_square'][sl].sum(0).tolist(),float(n.sum()),es['liftoffs'][sl].sum(0).tolist(),float(es['slip_square'][sl].sum()),float(es['saturated'][sl].sum()),evaluation_history[i]))
   if env.reference_v2:
    report['accepted']=report['accepted'] and report['gravity_rmse_deg']<=5.
    report['evaluation_limits']['gravity_rmse_deg']=5.
    if 'TURN' in ident:
     from teacher_turn_cycles import cycle_report
     totals=es['turn_cycle_sums'][sl].sum(0).tolist();report['turn_cycles']=cycle_report(*totals[:7]);report['turn_cycles']['mean_signed_angle_deg']=totals[7]/max(1.,totals[0])
     report['accepted']=report['accepted'] and totals[0]>=4 and report['turn_cycles']['pass_fraction']>=.8
    env.turn_boundary.observe(i,es['alive'][sl].tolist());report['turn_boundary']=env.turn_boundary.state(i)
    env.turn_curriculum.observe(i,report);report['turn_curriculum']=env.turn_curriculum.state(i)
   path=a.bank/ident/'evaluations'/('independent_%06d.json'%es['iteration']);atomic(path,report);atomic(a.bank/ident/'latest_evaluation.json',report)
   tracepath=a.bank/ident/'videos'/('trace_%06d.json'%es['iteration']);atomic(tracepath,{'teacher_id':ident,'iteration':es['iteration'],'joint_names':list(ACTION_NAMES),'type':'kinematic_trace','frames':es['trace'][i]})
   atomic(a.bank/ident/'video_receipt.json',{'type':'kinematic_trace','path':str(tracepath),'iteration':es['iteration'],'checkpoint_path':es['receipts'][i]['path'],'checkpoint_sha256':es['receipts'][i]['sha256'],'source':'Independent reserved-env deterministic rollout','created_unix':time.time()})
  eval_state=None
 def collect_eval():
  if eval_state is None:return
  es=eval_state;sl=slice(env.training_env_count,env.num_envs);state=env.pre_reset_teacher_state;active=es['alive'].float()
  es['count']+=active;es['q']+=state['q_rmse'][sl].square()*active;es['xy']+=state['xy_error'][sl].square()*active;es['yaw']+=state['yaw_error'][sl].square()*active;es['actual']+=state['actual_velocity'][sl]*active[:,None]
  if env.reference_v2:
   diag=state['v2_diagnostics'];es['foot_square']+=diag['foot_error_m'][sl].square()*active[:,None];es['liftoffs']+=diag['liftoff'][sl]*active[:,None];es['saturated']+=diag['foot_saturated_fraction'][sl]*active
   es['slip_square']+=diag['contact_slip_squared'][sl]*active
   es['contact_agreement']+=diag['contact_phase_agreement'][sl]*active
   if 'turn_cycles' in diag:
    c=diag['turn_cycles'];complete=c['complete'][sl].float()*active
    values=torch.stack((torch.ones_like(complete),c['passed'][sl].float(),c['net_angle_error_deg'][sl],c['displacement_m'][sl],c['max_drift_m'][sl],c['contact_agreement'][sl],c['cadence_error'][sl],c['signed_angle_deg'][sl]),1)
    es['turn_cycle_sums']+=values*complete[:,None]
   es['gravity_square']+=diag['gravity_error_rad'][sl].square()*active
  if env.pace_stability.enabled:
   for i,ident in enumerate(ids):
    if ident in ('02_STRAFE_LEFT','03_STRAFE_RIGHT'):
     group=slice(env.training_env_count+i*4,env.training_env_count+(i+1)*4)
     es['pace_dense'][i].append((state['reference_endpoint_error_vector_m'][group].cpu().numpy(),state['offset_joint_rad'][group].cpu().numpy(),es['alive'][i*4:(i+1)*4].cpu().numpy().copy()))
  newly_failed=torch.nonzero(es['alive']&state['fall'][sl],as_tuple=False).flatten()
  for k in newly_failed.tolist():
   j=env.training_env_count+k;es['first_fall_diagnostics'][k]={'time_s':(es['step']+1)*env.dt,**{name:value[j].item() for name,value in state['fall_diagnostics'].items()}}
  contact=state['contact'][sl];es['steps']+=((contact&~es['last_contact'])*es['alive'][:,None]).float();es['last_contact']=contact;es['alive']&=~state['fall'][sl]
  if es['step']%5==0:
   for i in range(len(ids)):
    j=env.training_env_count+i*4;root=state['root'][j];es['trace'][i].append({'time_s':(es['step']+1)*env.dt,'q':state['q'][j].tolist(),'root_position':(root[:3]-env.env_origins[j]).tolist(),'root_quaternion_xyzw':root[3:7].tolist(),'toe_world':(state['toe'][j]-env.env_origins[j]).tolist(),'reference_toe_world':(state['reference_toe'][j]-env.env_origins[j]).tolist(),'contact_normal_force':state['contact_force'][j].tolist(),'commands':env.teacher_commands[i].tolist(),'actual_velocity':state['actual_velocity'][j].tolist(),'active':bool(es['alive'][i*4]),'executed_target_full':state['executed_target_full'][j].tolist(),'residual_action':state['residual_action'][j].tolist(),'offset_joint_rad':state['offset_joint_rad'][j].tolist(),'reference_joint_target':state['reference_joint_target'][j].tolist(),'reward_cleanup_profile':a.reward_cleanup_profile,'removed_cadence_terms':list(REMOVED_CADENCE_TERMS) if a.reward_cleanup_profile==CADENCE_CLEANUP_PROFILE else [],'teacher_control_mode':a.control_mode})
  es['step']+=1
  if es['step']>=800:finish_eval()
 begin_eval(receipts)
 outcome='completed'
 try:
  while time.time()<deadline and not STOP and not (a.bank/'STOP').exists():
   import night_tuning
   night_tuning.refresh(env)
   if env.reference_v2 and eval_state is None:
    changed=sorted(set(env.turn_curriculum.apply(env)+env.turn_boundary.apply(env)))
    if changed:
     last_save=0.;bank['training_commands']=env.teacher_commands.tolist()
     for i in changed:evaluation_history[i].clear()
   env.pace_stability.set_strength(min(1.,max(0.,(iteration-pace_origin)/60.)),slice(0,env.training_env_count))
   pace_sums=None
   storages=[[[],[],[],[],[],[],[]] for _ in ids];reward_sum=torch.zeros(len(ids),device=env.device);begin=time.time();term_sums=None;term_names=None;actual_sum=torch.zeros(len(ids),3,device=env.device);direction_sums=torch.zeros(len(ids),4,device=env.device);quality_sums=None
   for step in range(24):
    obs=env.teacher_observations();action=torch.zeros(env.num_envs,15,device=env.device)
    with torch.no_grad():
     for i,(policy,store) in enumerate(zip(policies,storages)):
      sl=slice(i*a.envs_per_teacher,(i+1)*a.envs_per_teacher);dist=policy.distribution(obs[sl]);act=dist.sample();act[:,-1]=0.;action[sl]=act
      store[0].append(obs[sl].clone());store[1].append(act);store[2].append(dist.log_prob(act)[...,:14].sum(-1));store[5].append(policy.value(obs[sl]))
     for i,policy in enumerate(eval_policies):action[env.training_env_count+i*4:env.training_env_count+(i+1)*4]=policy.actor(obs[env.training_env_count+i*4:env.training_env_count+(i+1)*4])
    _,_,reward,done,info=env.step(action)
    quality=getattr(env.motion_rewards,'step_quality',None)
    if quality is not None:
     sample=quality.statistics(a.envs_per_teacher,len(ids));quality_sums=sample if quality_sums is None else quality_sums+sample
    if env.pace_stability.enabled:
     ps=env.pace_stability.statistics(a.envs_per_teacher,len(ids));pace_sums=ps if pace_sums is None else pace_sums+ps
    actual_sum+=env.pre_reset_teacher_state['actual_velocity'][:env.training_env_count].reshape(len(ids),a.envs_per_teacher,3).mean(1)
    measured=env.pre_reset_teacher_state['actual_velocity'][:env.training_env_count];desired=env.teacher_commands[env.teacher_ids[:env.training_env_count]];speed=desired[:,:2].norm(dim=1);turn=desired[:,2].abs()>.001
    correct=torch.where(turn,measured[:,2]*desired[:,2].sign()>.1*desired[:,2].abs(),(measured[:,:2]*desired[:,:2]).sum(1)>.1*speed.square())
    reverse=turn&(measured[:,2]*desired[:,2]<0);unwanted=(~turn)&(measured[:,2].abs()>.05);yaw_sq=(measured[:,2]-desired[:,2]).square()
    direction_sums+=torch.stack((correct.float(),reverse.float(),unwanted.float(),yaw_sq),1).reshape(len(ids),a.envs_per_teacher,4).mean(1)
    names=list(env.teacher_terms)
    if term_names is None:term_names=names;term_sums=torch.zeros(len(ids),len(names),device=env.device)
    assert names==term_names
    term_sums+=torch.stack([env.teacher_terms[name][:env.training_env_count] for name in names],1).reshape(len(ids),a.envs_per_teacher,len(names)).mean(1)
    reward_sum+=reward[:env.training_env_count].reshape(len(ids),a.envs_per_teacher).mean(1)
    with torch.no_grad():
     for i,(policy,store) in enumerate(zip(policies,storages)):
      sl=slice(i*a.envs_per_teacher,(i+1)*a.envs_per_teacher);store[3].append(reward[sl].clone());store[4].append(done[sl].bool().clone());store[6].append(policy.value(env.pre_reset_teacher_obs[sl])*env.time_out_buf[sl])
    collect_eval()
   with torch.no_grad():last=env.teacher_observations()
   losses=[]
   for i,(policy,opt,store) in enumerate(zip(policies,opts,storages)):
    sl=slice(i*a.envs_per_teacher,(i+1)*a.envs_per_teacher)
    with torch.no_grad():last_value=policy.value(last[sl])
    losses.append(update(policy,opt,store,last_value,a.minibatches))
   iteration+=1;elapsed=time.time()-start;updates_this_run=iteration-resume_iteration;throughput=updates_this_run*24*env.training_env_count/max(elapsed,1e-3)
   assert all(torch.isfinite(v).all() for policy in policies for v in policy.parameters())
   for i,ident in enumerate(ids):
    group=slice(i*a.envs_per_teacher,(i+1)*a.envs_per_teacher);offset=env.pre_reset_teacher_state['offset_joint_rad'][group]
    actual_std=policies[i].log_std.exp().clamp(policies[i].minimum_std,policies[i].maximum_std)[:14]
    metrics={'mean_reward_per_control_step':float(reward_sum[i]/24),'value_loss':losses[i][0],'surrogate_loss':losses[i][1],'action_std':float(actual_std.mean()),'action_std_deg':float(actual_std.mean()*.35*180/math.pi),'minimum_action_std':a.minimum_action_std,'training_objective':'balance_first' if a.balance_first else 'reference_and_task_tracking','elapsed_s':elapsed,'updates_this_run':updates_this_run,'environment_steps_per_second':throughput/len(ids),'iteration_time_s':time.time()-begin,'reference_execution_calls':env.reference_execution_calls,'reward_cleanup_profile':a.reward_cleanup_profile,'removed_cadence_terms':list(REMOVED_CADENCE_TERMS) if a.reward_cleanup_profile==CADENCE_CLEANUP_PROFILE else [],'teacher_control_mode':a.control_mode,'reference_strength':a.reference_strength,'offset_rms_rad':float(offset.square().mean().sqrt()),'offset_clip_fraction':float((offset.abs()>=env.offset_limit_rad-1e-6).float().mean())}
    metrics.update(target_command=env.teacher_commands[i].tolist(),actual_velocity=(actual_sum[i]/24).tolist(),weighted_rewards_per_second=dict(zip(term_names,(term_sums[i]/24).tolist())),reward_aggregation_scope='Weighted persecond before single dt integration;all pre-reset training transitions includingstartup/failures, notindependent evaluation',reference_guidance='loose_tracking')
    metrics['reference_time_scale']=float(env.ref_duration[i]/env.native_ref_duration[i]);metrics.update(task_tracking.metrics(a.task_tracking_profile,env.teacher_commands[i],env.raw_reference_commands[i],env.ref_duration[i],(env.velocity_tracking_weight[i],env.yaw_tracking_weight[i]),actual_sum[i]/24))
    if env.reference_v2:metrics.update(turn_boundary=env.turn_boundary.state(i),turn_curriculum=env.turn_curriculum.state(i),reward_metadata=reference_v2.METADATA,foot_tolerance_xyz_m=env.v2_sigma[i].tolist(),failure_phase_counts=env.v2_failures[i].tolist())
    if env.adaptive is not None:metrics.update(env.adaptive.report(i))
    if quality_sums is not None:
     from teacher_step_quality import statistics_report
     metrics.update(statistics_report(quality_sums[i]));metrics['genuine_step_scope']='v44 detector plus signed>=20mm, actual task progress and no fall; profile strict_steps_v1'
    if pace_sums is not None:
     from teacher_pace_stability import statistics_report as pace_report
     metrics.update(pace_report(pace_sums[i],float(env.pace_stability.strength[i*a.envs_per_teacher])))
    metrics.update(direction_correct_fraction=float(direction_sums[i,0]/24),reverse_yaw_fraction=float(direction_sums[i,1]/24),unwanted_yaw_fraction=float(direction_sums[i,2]/24),yaw_tracking_rmse_rad_s=float((direction_sums[i,3]/24).sqrt()),direction_metric_scope='All pre-reset training transitions;correct requires signed response>10%target,not task acceptance')
    metrics['night_reward_revision']=getattr(env,'night_control',{}).get('revision',0)
    with (a.bank/ident/'training_history.jsonl').open('a') as history:history.write(json.dumps({'time':time.time(),'iteration':iteration,'metrics':metrics})+'\n')
    atomic(a.bank/ident/'status.json',{'teacher_id':ident,'reference_id':ident,'status':'training','pid':os.getpid(),'iteration':iteration,'deadline':bank['deadline'],'metrics':metrics,'updated_unix':time.time(),'initialization':'fresh_random_pace_zero_mean_offset' if ident in fresh_teacher_ids else 'resumed_current_fullref_teachers_with_Adam_verified' if a.resume_bank else 'random_zero_mean_offset'})
   bank.update(iteration=iteration,updated_unix=time.time(),pace_stability_strength=float(env.pace_stability.strength[0]),metrics={'environment_steps_per_second':throughput,'elapsed_s':elapsed,'updates_this_run':updates_this_run,'iteration_time_s':time.time()-begin});atomic(a.bank/'teacher_bank.json',bank)
   if iteration==1 or time.time()-last_save>=180:
    receipts=save('periodic');last_save=time.time()
    if eval_state is None:begin_eval(receipts)
   if a.smoke and iteration-resume_iteration>=1:break
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
