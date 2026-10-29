"""Short GPU structural/control probe; explicitly not a survival acceptance run."""
import argparse
import json
from pathlib import Path
from isaacgym import gymapi
import torch
import numpy as np
from migration_config import student_config
from migration_env import create_migration_env
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-teacher', action='store_true')
    a = p.parse_args()
    manifest = verify_teacher_manifest(a.teacher)
    cfg = student_config(manifest['run'], num_envs=6)
    if a.no_teacher:
        cfg.reference.file = '/deliberately_missing_reference_table.json'
    torch.manual_seed(947)
    np.random.seed(947)
    env = create_migration_env(cfg, noise=False)
    obs, critic = env.reset()
    if not a.no_teacher:
        env.attach_teacher(manifest['checkpoint'], manifest['sha256'], 0.)
    env.record_substep_targets = True
    expected_waist = env.default_dof_pos[0, env.waist].item()
    target_error = 0.
    passive_torque = 0.
    fk_error = 0.
    for _ in range(100):
        assert obs.shape == (6, 326) and critic.shape == (6, 92)
        actions = torch.zeros(6, 15, device=env.device)
        obs, critic, reward, done, infos = env.step(actions)
        assert torch.isfinite(obs).all() and torch.isfinite(critic).all() and torch.isfinite(reward).all()
        assert infos['terminal_critic'].shape == (6, 92)
        assert infos['next_clean_proprio'].shape == (6, 51)
        assert len(env.substep_targets) == 40
        if env.teacher is not None and not done.any():
            for substep, target in enumerate(env.substep_targets):
                expected = env.teacher.target((substep+1)/40.)[:2]
                target_error = max(target_error, float((target-expected).abs().max()))
        passive_torque = max(passive_torque, float(env.torques[:, env.dependent].abs().max()))
        if not done.any():
            pos, _ = env.mapping.forward(env.root_states[:, :3], env.base_quat, env.dof_pos, env.dof_names)
            body_ids = [env.body_names.index(n) for n in env.mapping.body_names]
            fk_error = max(fk_error, float((pos-env.rigid_states[:, body_ids, :3]).abs().max()))
        assert abs(float(env.executed_target[:, -1].mean())-expected_waist) < 1e-6
    assert target_error < 1e-6 and passive_torque == 0.
    assert fk_error < 1e-4, fk_error
    if a.no_teacher:
        assert env.reference_execution_calls == 0 and env.motion_reference is None
    result = {'passed': True, 'seconds': 2., 'environments': 6,
        'teacher_attached': env.teacher is not None, 'raw_dimension': 326, 'critic_dimension': 92,
        'action_dimension': 15, 'full_substep_target_error_rad': target_error,
        'passive_torque_abs_max_nm': passive_torque, 'fk_vs_physics_origin_max_error_m': fk_error,
        'reference_execution_calls': env.reference_execution_calls,
        'scope': 'Structural2s probe, not16s or long-horizon stability evidence',
        'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(a.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}), flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
