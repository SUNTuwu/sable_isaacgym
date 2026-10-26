"""Known horizontal load on the standing robot; no policy or PPO involved."""
import json,time,types
from pathlib import Path
from isaacgym import gymapi,gymtorch
import torch
from preflight import create_env
from run_io import atomic_json,sha256

env=create_env(5,noise=False);env.reset_noise=0.;env.fixed_command=0.;env.reset()
loads=torch.tensor([[0.,0.,0.],[.3,0.,0.],[-.3,0.,0.],[0.,.3,0.],[0.,-.3,0.]],device=env.device)
forces=torch.zeros((5,env.num_bodies,3),device=env.device)
original=env._compute_torques
def torque_with_external(self,actions):
    forces[:,0]=loads
    # Native Gym ENV_SPACE axes equal world axes (environments only translate).
    self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(forces),None,gymapi.ENV_SPACE)
    return original(actions)
env._compute_torques=types.MethodType(torque_with_external,env)
observed=[];normals=[];speeds=[];falls=0;t0=time.time()
for k in range(400):
    _,_,_,done,_=env.step(torch.zeros(5,12,device=env.device))
    falls+=int(env.true_fall.sum())
    if k>=300:
        observed.append(env.leg_propulsion.force.sum(1).clone())
        normals.append(env.contact_forces[:,env.feet_indices].sum(1).clone())
        speeds.append(env.root_states[:,7:10].clone())
sensor=torch.stack(observed).mean(0);normal=torch.stack(normals).mean(0)
residual=sensor[:,:2]+loads[:,:2]
checks={'no_falls':falls==0,
        'known_horizontal_load_balanced':bool(residual.abs().max()<.04),
        'world_xy_axes_and_sign':bool(sensor[1,0]<-.25 and sensor[2,0]>.25 and sensor[3,1]<-.25 and sensor[4,1]>.25),
        'weight_supported':bool((sensor[:,2]-env.import_mass*9.81).abs().max()<.2),
        'approximately_stationary':bool(torch.stack(speeds).abs().max()<.01),
        'net_contact_tensor_lacks_horizontal_reaction':bool(normal[:,:2].abs().max()<1e-4)}
here=Path(__file__).resolve().parent
report={'passed':all(checks.values()),'checks':checks,'wall_seconds':time.time()-t0,
        'applied_force_world_n':loads.cpu().tolist(),'measured_feet_force_world_n':sensor.cpu().tolist(),
        'last_step_net_contact_force_world_n':normal.cpu().tolist(),
        'horizontal_balance_max_error_n':float(residual.abs().max()),
        'source_sha256':{n:sha256(here/n) for n in ['sable_env.py','leg_propulsion.py','check_propulsion_force_sensor.py']},
        'scope':'Remote simulation known-load check; external forces only in this diagnostic, not the training environment.'}
atomic_json(here/'verification/propulsion_force_sensor.json',report);print(json.dumps(report));env.gym.destroy_sim(env.sim)
assert report['passed']
