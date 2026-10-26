"""Single fixed-base SDK quadruped: exercise CUDA state, torque and reset APIs.

This is an installation/interface check with gravity disabled, not locomotion,
policy training, controller tuning or validation of the user's robot.
"""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path('/root/gpufree-data/projects/sable-isaacgym')
assert sys.platform == 'linux' and ROOT.is_dir()
result = {
    'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'passed': False, 'no_training': True, 'num_envs': 1,
    'asset': 'urdf/anymal_b_simple_description/urdf/anymal.urdf',
    'fixed_base': True, 'gravity_enabled': False,
    'purpose': 'GPU articulation, torque tensor and indexed reset interface check',
}
gym = sim = None
try:
    from isaacgym import gymapi, gymtorch
    import numpy as np
    import torch

    assert torch.cuda.is_available(), 'CUDA unavailable'
    result.update(gpu_name=torch.cuda.get_device_name(0), torch=torch.__version__,
                  torch_cuda=torch.version.cuda)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    # Explicit torque feedback needs a faster update than a passive sphere drop.
    # Physics substeps alone do not recompute the external damping torque.
    params.dt = 0.001
    params.substeps = 1
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.physx.use_gpu = True
    params.use_gpu_pipeline = True
    params.physx.num_position_iterations = 4
    params.physx.num_velocity_iterations = 1
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    assert sim is not None, 'create_sim failed'
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    options.collapse_fixed_joints = True
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    asset = gym.load_asset(sim, str(ROOT / 'vendor/isaacgym/assets'), result['asset'], options)
    assert asset is not None, 'load_asset failed'
    names = gym.get_asset_dof_names(asset)
    assert len(names) == 12, 'SDK quadruped should have 12 DOFs'
    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 4), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 2)
    actor = gym.create_actor(env, asset, pose, 'sdk_quadruped_check', 0, 1)
    props = gym.get_actor_dof_properties(env, actor)
    props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
    props['stiffness'].fill(0)
    props['damping'].fill(0)
    gym.set_actor_dof_properties(env, actor, props)
    nominal_np = np.clip(np.zeros(12), props['lower'] + 0.2, props['upper'] - 0.2).astype(np.float32)
    initial = np.zeros(12, dtype=gymapi.DofState.dtype)
    initial['pos'] = nominal_np
    gym.set_actor_dof_states(env, actor, initial, gymapi.STATE_ALL)
    gym.prepare_sim(sim)
    state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    bodies = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    contacts = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    assert state.shape == (12, 2) and state.is_cuda
    assert roots.is_cuda and bodies.is_cuda and contacts.is_cuda
    gym.refresh_dof_state_tensor(sim)
    nominal = torch.tensor(nominal_np, device='cuda:0')
    effort_limit = torch.tensor(np.minimum(props['effort'], 30), device='cuda:0')
    assert bool((effort_limit > 0).all())
    errors, positions = [], []
    torque = torch.zeros(12, dtype=torch.float32, device='cuda:0')
    for step in range(4000):
        target = nominal + 0.10 * math.sin(2 * math.pi * 0.5 * step * params.dt)
        torque.copy_(torch.clamp(80 * (target - state[:, 0]) - 4 * state[:, 1],
                                min=-effort_limit, max=effort_limit))
        assert gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_dof_state_tensor(sim)
        if step >= 1000:
            errors.append((state[:, 0] - target).clone())
            positions.append(state[:, 0].clone())
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    assert all(bool(torch.isfinite(t).all()) for t in [state, roots, bodies, contacts, torque])
    history = torch.stack(positions)
    excursion = history.max(dim=0).values - history.min(dim=0).values
    rms = torch.stack(errors).square().mean().sqrt().item()
    result.update(steps=4000, simulated_seconds=4.0, control_dt_seconds=params.dt,
                  kp=80, kd=4, torque_cap_nm=30, dof_count=12, dof_names=names,
                  rigid_body_count=bodies.shape[0], state_device=str(state.device),
                  torque_device=str(torque.device), all_state_tensors_cuda=True,
                  joint_excursion_rad=excursion.cpu().tolist(), tracking_rmse_rad=rms,
                  tracking_rmse_by_joint_rad=torch.stack(errors).square().mean(dim=0).sqrt().cpu().tolist(),
                  final_dof_state=state.cpu().tolist(), nominal_rad=nominal.cpu().tolist(),
                  joint_mean_rad=history.mean(dim=0).cpu().tolist(),
                  physics_engine='PhysX', gpu_pipeline=True)
    assert bool((excursion > 0.02).all()), 'One or more joints did not respond to CUDA torques'
    assert rms < 0.15, 'Unexpectedly large interface-test tracking error'
    state[:, 0] = nominal
    state[:, 1] = 0
    actor_ids = torch.tensor([0], dtype=torch.int32, device='cuda:0')
    assert gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(state),
                                          gymtorch.unwrap_tensor(actor_ids), 1)
    gym.refresh_dof_state_tensor(sim)
    reset_error = (state[:, 0] - nominal).abs().max().item()
    assert reset_error < 1e-5 and state[:, 1].abs().max().item() < 1e-5
    torch.cuda.synchronize()
    result.update(indexed_reset_max_error_rad=reset_error, passed=True, status='passed')
except Exception as exc:
    result['error'] = str(exc)
    raise
finally:
    if sim is not None:
        gym.destroy_sim(sim)
    (ROOT / 'verification/articulation_gpu.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
