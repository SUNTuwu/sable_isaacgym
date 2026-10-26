"""Separate CPU-PhysX known-load diagnostic; never a training backend change."""
import sys,json,time,types,hashlib
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
SOURCE=ROOT/'deployment/legged_gym_leg_propulsion_20261025'
sys.path.insert(0,str(SOURCE))
from isaacgym import gymapi,gymtorch
import torch
from sable_config import SableCfg
from sable_env import SableRobot
cfg=SableCfg();cfg.env.num_envs=5;cfg.asset.fix_base_link=False
cfg.noise.add_noise=False;cfg.domain_rand.randomize_friction=False
cfg.propulsion.enabled=True;cfg.rewards.scales.leg_propulsion=0.
params=gymapi.SimParams();params.dt=cfg.sim.dt;params.substeps=cfg.sim.substeps
params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=4
params.physx.solver_type=1;params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=2
params.physx.contact_offset=.0005;params.physx.rest_offset=0.
params.physx.max_depenetration_velocity=.5;params.physx.bounce_threshold_velocity=.1
params.physx.contact_collection=gymapi.ContactCollection.CC_ALL_SUBSTEPS
env=SableRobot(cfg,params,gymapi.SIM_PHYSX,'cpu',True)
env.reset_noise=0.;env.fixed_command=0.;env.reset()
loads=torch.tensor([[0.,0.,0.],[.3,0.,0.],[-.3,0.,0.],[0.,.3,0.],[0.,-.3,0.]])
forces=torch.zeros(5,env.num_bodies,3);original=env._compute_torques
def torque(self,actions):
    forces[:,0]=loads
    self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(forces),None,gymapi.ENV_SPACE)
    return original(actions)
env._compute_torques=types.MethodType(torque,env)
sensors=[];nets=[];speeds=[];falls=0;t0=time.time()
for k in range(400):
    env.step(torch.zeros(5,12));falls+=int(env.true_fall.sum())
    if k>=300:
        sensors.append(env.leg_propulsion.force.sum(1).clone())
        nets.append(env.contact_forces[:,env.feet_indices].sum(1).clone())
        speeds.append(env.root_states[:,7:10].clone())
    if k%100==0:print('CPU_FORCE_PROGRESS',k,flush=True)
sensor=torch.stack(sensors).mean(0);net=torch.stack(nets).mean(0)
checks={}
for name,value in [('sensor',sensor),('net_contact',net)]:
    checks[name]={'horizontal_balance':bool((value[:,:2]+loads[:,:2]).abs().max()<.04),
        'weight_supported':bool((value[:,2]-env.import_mass*9.81).abs().max()<.2)}
report={'created_unix':time.time(),'wall_seconds':time.time()-t0,'backend':'CPU PhysX, diagnostic only',
    'checks':checks,'falls':falls,'max_speed_m_s':float(torch.stack(speeds).abs().max()),
    'loads_n':loads.tolist(),'sensor_n':sensor.tolist(),'net_contact_n':net.tolist(),
    'passed':falls==0 and bool(torch.stack(speeds).abs().max()<.01) and any(all(c.values()) for c in checks.values()),
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_environment_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')},
    'scope':'A CPU-backend check cannot by itself validate GPU-backend per-foot force observations.'}
(ROOT/'results/legged_gym/cpu_ground_force_calibration.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report));env.gym.destroy_sim(env.sim)
