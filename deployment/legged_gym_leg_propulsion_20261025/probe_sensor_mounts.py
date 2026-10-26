"""Independent sensor-location diagnostic; source assets remain unchanged."""
import argparse,json,types,time
from pathlib import Path
from isaacgym import gymapi,gymtorch
import torch
from preflight import create_env
from sable_config import SableCfg
from run_io import atomic_json,sha256
p=argparse.ArgumentParser();p.add_argument('--mount',choices=['feet','lower','upper'],required=True);a=p.parse_args()
cfg=SableCfg()
mounts={'feet':['L_wrist_fixed','R_wrist_fixed','L_ankle_fixed','R_ankle_fixed'],
        'lower':['L_elbow_pitch','R_elbow_pitch','L_knee_pitch','R_knee_pitch'],
        'upper':['L_shoulder_pitch','R_shoulder_pitch','L_hip_pitch','R_hip_pitch']}
cfg.propulsion.sensor_bodies=mounts[a.mount]
env=create_env(3,noise=False,cfg=cfg);env.reset_noise=0.;env.fixed_command=0.;env.reset()
load=torch.tensor([[0.,0.,0.],[.3,0.,0.],[0.,.3,0.]],device=env.device)
force=torch.zeros(3,env.num_bodies,3,device=env.device);force[:,0]=load
original=env._compute_torques
def actuate(self,actions):
    self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(force),None,gymapi.ENV_SPACE)
    return original(actions)
env._compute_torques=types.MethodType(actuate,env)
history=[];normal=[]
for k in range(200):
    env.step(torch.zeros(3,12,device=env.device))
    if k>=150:
        history.append(env.leg_propulsion.force.clone());normal.append(env.contact_forces[:,env.feet_indices].clone())
report={'mount':a.mount,'sensor_bodies':cfg.propulsion.sensor_bodies,'applied_force_world_n':load.cpu().tolist(),
        'mean_sensor_world_n':torch.stack(history).mean(0).cpu().tolist(),'mean_normal_world_n':torch.stack(normal).mean(0).cpu().tolist(),
        'created_unix':time.time(),'source_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ['sable_env.py','leg_propulsion.py','probe_sensor_mounts.py']}}
atomic_json(Path(__file__).with_name('verification')/('sensor_mount_'+a.mount+'.json'),report)
print(json.dumps(report));env.gym.destroy_sim(env.sim)
