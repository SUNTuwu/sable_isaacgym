"""0.5ms physical diagnostic of force-sensor and rigid-velocity conventions."""
import argparse,json
from pathlib import Path
from isaacgym import gymapi,gymtorch
import torch
from migration_config import student_config
from migration_env import create_migration_env,MigrationRobot
from history_policy import HistoryActorCritic
from model_mapping import quat_apply
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,sha256


def main():
 from sable_config import MODEL
 def configure(self,asset):
  props=gymapi.ForceSensorProperties();props.enable_forward_dynamics_forces=False;props.enable_constraint_solver_forces=True;props.use_world_frame=True
  for name in MODEL['foot_bodies']:self.gym.create_asset_force_sensor(asset,self.body_names.index(name),gymapi.Transform(),props)
 original_init=MigrationRobot._init_buffers
 def init(self):
  original_init(self);self.terminal_force_sensors=gymtorch.wrap_tensor(self.gym.acquire_force_sensor_tensor(self.sim)).view(self.num_envs,4,6)
 MigrationRobot._configure_force_sensors=configure;MigrationRobot._init_buffers=init
 p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.manual_seed(947);cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',2);cfg.control.decimation=1;cfg.commands.resampling_time=1000.;cfg.env.episode_length_s=20.
 env=create_migration_env(cfg,noise=False);env.capture_eval=True;env.reset();cmd=env.commands16.clone();cmd[:,:3]=cmd.new_tensor([.105,0,0]);env.set_commands(cmd)
 policy=HistoryActorCritic().to(env.device).eval();policy.load_state_dict(torch.load(a.checkpoint,map_location=env.device)['model_state_dict'])
 props=env.gym.get_actor_rigid_body_properties(env.envs[0],env.actor_handles[0]);mass=cmd.new_tensor([p.mass for p in props]);com=cmd.new_tensor([[p.com.x,p.com.y,p.com.z]for p in props]);total=mass.sum()
 previous=None;alive=torch.ones(2,dtype=torch.bool,device=env.device);rows=[]
 nonfeet=[i for i in range(env.num_bodies)if i not in env.feet_indices.tolist()]
 with torch.no_grad():
  for step in range(2000):
   if step%40==0:action=policy.act_inference(env.get_observations())
   _,_,_,done,_=env.step(action);state=env.rigid_states.clone();offset=quat_apply(state[...,3:7],com[None].expand(2,-1,-1));center=state[...,:3]+offset
   velocity=state[...,7:10];v_com_if_origin=velocity+torch.cross(state[...,10:13],offset,dim=-1)
   momentum_bare=(mass[None,:,None]*velocity).sum(1);momentum_corrected=(mass[None,:,None]*v_com_if_origin).sum(1)
   env.gym.refresh_force_sensor_tensor(env.sim)
   sensor=env.terminal_force_sensors[...,:3].sum(1).clone();normal=env.contact_forces[:,env.feet_indices].sum(1).clone()
   clean=(env.contact_forces[:,nonfeet].norm(dim=-1)<.1).all(1)
   if previous is not None:
    old,old_center,old_bare,old_corrected,old_sensor,old_normal,mask=previous;valid=alive&~done&mask&clean
    if bool(valid.any()):
     origin_fd=(state[...,:3]-old[...,:3])/env.dt;com_fd=(center-old_center)/env.dt;average=(velocity+old[...,7:10])*.5
     gravity=cmd.new_tensor([0.,0.,-9.81])*total
     inertial_bare=(momentum_bare-old_bare)/env.dt-gravity
     inertial_corrected=(momentum_corrected-old_corrected)/env.dt-gravity
     rows.append(dict(origin_velocity_error=(origin_fd-average)[valid][:,env.feet_indices].cpu(),com_velocity_error=(com_fd-average)[valid][:,env.feet_indices].cpu(),
      force_error_bare=(sensor-inertial_bare)[valid].cpu(),force_error_corrected=(sensor-inertial_corrected)[valid].cpu(),normal_error_bare=(normal-inertial_bare)[valid].cpu(),
      sensor_force=sensor[valid].cpu(),normal_force=normal[valid].cpu()))
   alive&=~done;previous=(state,center,momentum_bare,momentum_corrected,sensor,normal,alive&clean)
 data={k:torch.cat([r[k]for r in rows])for k in rows[0]}
 result=dict(samples=len(data['sensor_force']),control_sample_s=env.dt,nominal_weight_n=float(total*9.81),
  vector_RMSE={k:float(v.square().sum(-1).mean().sqrt())for k,v in data.items()if'error'in k},
  component_RMSE={k:v.square().mean(0).sqrt().tolist()for k,v in data.items()if k.startswith(('force_error','normal_error'))},
  sensor_force_mean_n=data['sensor_force'].mean(0).tolist(),normal_force_mean_n=data['normal_force'].mean(0).tolist(),
  sensor_horizontal_RMS_n=float(data['sensor_force'][:,:2].square().sum(-1).mean().sqrt()),normal_horizontal_RMS_n=float(data['normal_force'][:,:2].square().sum(-1).mean().sqrt()),
  scope='Diagnostic uses0.5ms sampling, same assets and PD with action held40substeps. Not a locomotion or policy acceptance test. Only original episodes without other-body contact; force balance compares total momentum change against four solver sensors and ordinary normal contact tensor.')
 atomic_json(a.output,result);print(json.dumps(result));env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
