"""One short physics check, never training; run with --device gpu after GPU boot."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--device',choices=['cpu','gpu'],default='cpu')
args=p.parse_args()
root=Path('/root/gpufree-data/projects/sable-isaacgym')
assert root.is_dir() and sys.platform=='linux'
result={'device_requested':args.device,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'python':sys.version,'prefix':sys.prefix,'no_training':True,'passed':False}
destination=root/'verification'/('smoke_'+args.device+'.json')
try:
    try:
        ctypes.CDLL('libcuda.so.1')
    except OSError as exc:
        result['status']='blocked_driver: CUDA driver library unavailable; switch server to GPU mode'
        raise RuntimeError(result['status']+': '+str(exc))
    # Isaac Gym must be imported before torch.
    from isaacgym import gymapi,gymtorch
    import torch
    import numpy as np
    result.update(torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
                  gym_imported=True,tensor_bridge_imported=True)
    if args.device=='gpu' and not torch.cuda.is_available():
        result['status']='pending_gpu: CUDA device not available'
        raise RuntimeError(result['status'])
    use_gpu=args.device=='gpu'
    gym=gymapi.acquire_gym()
    params=gymapi.SimParams()
    params.dt=1./120.; params.substeps=2
    params.up_axis=gymapi.UP_AXIS_Z; params.gravity=gymapi.Vec3(0,0,-9.81)
    params.physx.use_gpu=use_gpu; params.use_gpu_pipeline=use_gpu
    params.physx.num_threads=2
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
    if sim is None: raise RuntimeError('create_sim returned None')
    try:
        plane=gymapi.PlaneParams(); plane.normal=gymapi.Vec3(0,0,1)
        gym.add_ground(sim,plane)
        env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,2),1)
        asset=gym.create_sphere(sim,.05,gymapi.AssetOptions())
        pose=gymapi.Transform(); pose.p=gymapi.Vec3(0,0,.5)
        actor=gym.create_actor(env,asset,pose,'installation_check',0,0)
        gym.prepare_sim(sim)
        state=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        for step in range(240):
            gym.simulate(sim); gym.fetch_results(sim,True)
        gym.refresh_actor_root_state_tensor(sim)
        values=state.detach().cpu().numpy()
        result.update(steps=240,simulated_seconds=2.,state_device=str(state.device),final_root_state=values.tolist(),
                      physics_engine='PhysX',gpu_pipeline=use_gpu)
        assert values.shape==(1,13) and np.isfinite(values).all(), 'Invalid actor state'
        assert .035<float(values[0,2])<.075, 'Sphere did not settle on the ground'
        assert np.linalg.norm(values[0,7:10])<.1, 'Sphere still moving unexpectedly'
        if use_gpu: assert state.is_cuda, 'GPU tensor pipeline not active'
        result.update(passed=True,status='passed',gpu_validation='passed' if use_gpu else 'pending_gpu_boot')
    finally:
        gym.destroy_sim(sim)
except Exception as exc:
    result['error']=str(exc)
    raise
finally:
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
