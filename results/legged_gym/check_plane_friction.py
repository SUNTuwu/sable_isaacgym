"""Measure native-plane effective dynamic friction from free sliding deceleration."""
import json,time,argparse
from pathlib import Path
from isaacgym import gymapi,gymtorch
import numpy as np
import torch
args=argparse.ArgumentParser();args.add_argument('--plane',type=float,default=1.);args.add_argument('--output',type=Path,default=Path(__file__).with_name('native_plane_friction_calibration.json'));a=args.parse_args()
assert not a.output.exists(),'Preserve earlier friction calibration'
g=gymapi.acquire_gym();p=gymapi.SimParams();p.dt=.0005;p.substeps=1
p.up_axis=gymapi.UP_AXIS_Z;p.gravity=gymapi.Vec3(0,0,-9.81);p.use_gpu_pipeline=True;p.physx.use_gpu=True
p.physx.num_position_iterations=8;p.physx.num_velocity_iterations=2;p.physx.contact_offset=.0005;p.physx.rest_offset=0.
sim=g.create_sim(0,-1,gymapi.SIM_PHYSX,p)
plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);plane.static_friction=a.plane;plane.dynamic_friction=a.plane;plane.restitution=0.
g.add_ground(sim,plane)
opt=gymapi.AssetOptions();opt.angular_damping=0.;opt.linear_damping=0.;opt.density=5000.
asset=g.create_box(sim,.1,.1,.02,opt);coefficients=[0.,.02,.05,.1,.2,.5,1.,2.]
for i,mu in enumerate(coefficients):
    e=g.create_env(sim,gymapi.Vec3(0,0,0),gymapi.Vec3(0,0,0),3)
    pose=gymapi.Transform();pose.p=gymapi.Vec3(i*.4,0,.011)
    actor=g.create_actor(e,asset,pose,'slider',i,0)
    shapes=g.get_actor_rigid_shape_properties(e,actor)
    for s in shapes:s.friction=mu;s.restitution=0.;s.rolling_friction=0.;s.torsion_friction=0.
    g.set_actor_rigid_shape_properties(e,actor,shapes)
g.prepare_sim(sim);root=gymtorch.wrap_tensor(g.acquire_actor_root_state_tensor(sim))
for k in range(600):g.simulate(sim)
g.refresh_actor_root_state_tensor(sim);root[:,7]=.3;root[:,8:]=0.
g.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(root));history=[]
for k in range(1500):
    g.simulate(sim);g.refresh_actor_root_state_tensor(sim)
    history.append(root[:,7].cpu().numpy().copy())
v=np.asarray(history);t=(np.arange(len(v))+1)*p.dt;rows=[]
for i,mu in enumerate(coefficients):
    mask=(v[:,i]>.05)&(v[:,i]<.25)&(t>.002)
    measured=-np.polyfit(t[mask],v[mask,i],1)[0]/9.81 if mask.sum()>5 else None
    rows.append({'shape_friction':mu,'effective_dynamic_mu_from_deceleration':measured,'fit_samples':int(mask.sum()),'final_velocity_m_s':float(v[-1,i])})
result={'created_unix':time.time(),'plane_friction':a.plane,'rows':rows,'scope':'Isolated rigid slider on the same native PhysX plane; dynamic friction measurement only, not robot gait evidence.'}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result));g.destroy_sim(sim)
