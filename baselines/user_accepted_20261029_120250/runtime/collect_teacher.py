"""Collect teacher rollouts with exact executed end-of-interval target labels."""
import argparse
import json
import time
from pathlib import Path
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse
import torch
import numpy as np
from preflight import create_env
from migration_config import teacher_config
from equivariant_policy import SymmetricActorCritic
from teacher_adapter import verify_teacher_manifest
from action_schema import ActionSpec
from command_schema import CommandSpec
from observation_schema import HistoryBuffer, proprioception, pack
from gait_clock import clocks
from model_mapping import ModelMapping
from run_io import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--seconds', type=float, default=40.)
    parser.add_argument('--seed', type=int, default=42)
    a = parser.parse_args()
    assert not a.output.exists(), 'Refusing to overwrite a dataset'
    a.output.mkdir(parents=True)
    manifest = verify_teacher_manifest(a.teacher)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = teacher_config(manifest['run'], a.num_envs)
    env = create_env(a.num_envs, noise=True, cfg=cfg)
    config = json.loads((Path(manifest['run']) / 'config.json').read_text())
    teacher = SymmetricActorCritic(58, 58, 14, **config['train']['policy']).to(env.device).eval()
    payload = torch.load(manifest['checkpoint'], map_location=env.device)
    teacher.load_state_dict(payload['model_state_dict'], strict=True)
    teacher.requires_grad_(False)
    active = ActionSpec.indices(env.dof_names, env.device)
    default = env.default_dof_pos[:, active]
    executed = default.expand(a.num_envs, -1).clone()
    original_torques = env._compute_torques
    substep_count = [0]

    def instrument(actions):
        torque = original_torques(actions)
        alpha = min(1., env.motion_reference.tick / cfg.control.decimation)
        reference = env.motion_reference.start + alpha * (env.motion_reference.end - env.motion_reference.start)
        target = env.default_dof_pos.expand(env.num_envs, -1).clone()
        target[:, env.active] += actions * cfg.control.action_scale
        target += reference
        target = torch.maximum(torch.minimum(target, env.hard_limits[:, 1]), env.hard_limits[:, 0])
        executed.copy_(target[:, active])
        substep_count[0] += 1
        return torque

    env._compute_torques = instrument
    mapping = ModelMapping(cfg.asset.file, Path(cfg.asset.file).with_name('physical_manifest.json'))
    measurements = mapping.default_measurements()
    commands = torch.tensor(CommandSpec.defaults(measurements['shoulder_height_m'], measurements['hip_height_m'],
                            measurements['width0_m'], float(default[0, -1])), device=env.device).repeat(a.num_envs, 1)
    history = HistoryBuffer(a.num_envs, env.device)

    def current_proprio():
        # Recompute reset kinematics from the reset state, not stale pre-reset caches.
        quat = env.root_states[:, 3:7]
        ang = quat_rotate_inverse(quat, env.root_states[:, 10:13])
        gravity = quat_rotate_inverse(quat, env.gravity_vec)
        return proprioception(ang, gravity, env.dof_pos[:, active], env.dof_vel[:, active], default, executed)

    fields = {k: [] for k in ('raw', 'target', 'velocity', 'next_proprio', 'done')}
    shard_paths = []
    started = time.time()
    # Include the true initial decision at t=0. BaseTask.reset() performs a hidden
    # zero-action physics interval, which omitted the student's deployment start.
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.base_quat = env.root_states[:, 3:7]
    env.base_lin_vel = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.compute_observations()
    obs = env.get_observations()
    episodes = 0
    with torch.no_grad():
        for step in range(round(a.seconds / env.dt)):
            p = current_proprio()
            history.sample(p, env.common_step_counter)
            commands[:, :3] = env.commands[:, :3]
            phase = torch.remainder(env.gait_signals.phase(env) + .75, 1.)
            raw = pack(history.data, commands, clocks(phase, commands))
            velocity = quat_rotate_inverse(env.root_states[:, 3:7], env.root_states[:, 7:10]).clone()
            action = teacher.act_inference(obs)
            obs, _, _, done, _ = env.step(action)
            label = ActionSpec.offsets(executed, default).clone()
            ids = done.nonzero(as_tuple=False).flatten()
            episodes += len(ids)
            executed[ids] = default
            history.reset(ids)
            values = (raw, label, velocity, current_proprio(), done.clone())
            for k, v in zip(fields, values):
                fields[k].append(v.cpu())
            if len(fields['raw']) == 100 or step == round(a.seconds / env.dt) - 1:
                shard = a.output / ('shard_%04d.pt' % len(shard_paths))
                torch.save({k: torch.cat(v) for k, v in fields.items()}, str(shard))
                shard_paths.append({'path': str(shard), 'sha256': sha256(shard)})
                fields = {k: [] for k in fields}
                atomic_json(a.output / 'status.json', {'status': 'collecting', 'steps': step+1,
                    'samples': (step+1)*a.num_envs, 'teacher_iteration': manifest['iteration'], 'updated_unix': time.time()})
    atomic_json(a.output / 'dataset.json', {'schema': 'sable-complete-teacher-targets/1',
        'teacher_manifest': str(a.teacher), 'teacher_sha256': manifest['sha256'],
        'teacher_iteration': manifest['iteration'], 'seed': a.seed, 'num_envs': a.num_envs,
        'seconds_per_env': a.seconds, 'substep_target_observations': substep_count[0],
        'samples': round(a.seconds/env.dt)*a.num_envs, 'shards': shard_paths,
        'reset_count': episodes, 'wall_seconds': time.time()-started, 'measurements': measurements,
        'initial_observation': 'true_reset_state_before_first_physics_interval',
        'supervision': 'same-time ROOT velocity; complete clamped final-substep target; next clean proprioception masked at resets'})
    atomic_json(a.output / 'status.json', {'status': 'complete', 'samples': round(a.seconds/env.dt)*a.num_envs,
                                         'updated_unix': time.time()})
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
