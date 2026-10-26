"""Inspect legacy contact fields on remote GPU PhysX with CPU readback only."""
import json,types,time
from pathlib import Path
from isaacgym import gymapi,gymtorch
import torch
from preflight import create_env
from sable_config import SableCfg
from run_io import atomic_json
cfg=SableCfg();cfg.propulsion.enabled=False;cfg.rewards.scales.leg_propulsion=0.
env=create_env(3,cfg=cfg,cpu_pipeline=True);env.reset_noise=0.;env.fixed_command=0.;env.reset()
force=torch.zeros(3,env.num_bodies,3);force[1,0,0]=.3;force[2,0,1]=.3
original=env._compute_torques
def actuate(self,actions):
    self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(force),None,gymapi.ENV_SPACE)
    return original(actions)
env._compute_torques=types.MethodType(actuate,env)
for k in range(150):env.step(torch.zeros(3,12))
report={'created_unix':time.time(),'gpu_physics':True,'cpu_pipeline':True,'foot_ids':env.feet_indices.tolist(),'contacts':[]}
for i,e in enumerate(env.envs):
    contacts=env.gym.get_env_rigid_contacts(e)
    print('FIELDS',contacts.dtype,flush=True)
    rows=[]
    for contact in contacts:
        rows.append({name:str(contact[name]) for name in contacts.dtype.names})
    report['contacts'].append({'env':i,'rows':rows})
atomic_json(Path(__file__).with_name('verification')/'cpu_contact_fields.json',report)
print(json.dumps(report));env.gym.destroy_sim(env.sim)
