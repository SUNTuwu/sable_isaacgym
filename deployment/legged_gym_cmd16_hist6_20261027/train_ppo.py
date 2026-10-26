"""Teacher-free on-policy training with explicit history auxiliary labels."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time
from isaacgym import gymapi
import numpy as np
import torch
from migration_env import create_migration_env
from migration_config import student_config
from history_policy import HistoryActorCritic
from ppo_history import HistoryPPO
from curriculum import next_frequency_range
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json, append_json, sha256

STOP = False


def stop(*_):
    global STOP
    STOP = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--stage', choices=['S2', 'S3', 'S5'], required=True)
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--rollout-steps', type=int, default=24)
    p.add_argument('--max-updates', type=int, default=100000000)
    p.add_argument('--critic-warmup-updates', type=int, default=10)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=44)
    a = p.parse_args()
    deadline = datetime.datetime.fromisoformat(a.deadline).timestamp()
    assert deadline > time.time()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    teacher = verify_teacher_manifest(a.teacher)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = student_config(teacher['run'], stage=a.stage, num_envs=a.num_envs)
    cfg.migration.moving_start_training = a.stage == 'S3'
    if cfg.migration.moving_start_training:
        # Sample transitions explicitly before policy inference, rather than
        # changing the commanded target inside a physics/reward interval.
        cfg.commands.resampling_time = cfg.env.episode_length_s+1.
    cfg.reference.file = '/intentionally_absent_ppo_reference.json'
    env = create_migration_env(cfg, noise=True)
    policy = HistoryActorCritic().to(env.device)
    parent = torch.load(str(a.resume), map_location=env.device)
    policy.load_state_dict(parent['model_state_dict'], strict=True)
    policy.novel_command_std_floor=cfg.migration.novel_command_std_floor
    with torch.no_grad():
        policy.std[:14].clamp_(min=cfg.migration.minimum_action_std)
    if a.stage == 'S5' and parent.get('stage') == 'S5' and 'frequency_range' in parent:
        previous = parent['frequency_range']
        if list(previous) not in ([.9,1.1],[.8,1.2]):
            raise ValueError('Unrecognized previous frequency curriculum bounds')
        cfg.migration.command_ranges['frequency'] = list(previous)
    algorithm = HistoryPPO(policy, device=env.device, num_learning_epochs=4, num_mini_batches=4,
        clip_param=.15, gamma=.99, lam=.95, value_loss_coef=1., entropy_coef=.001,
        learning_rate=a.learning_rate, max_grad_norm=1., use_clipped_value_loss=True,
        schedule='fixed', desired_kl=None, auxiliary_learning_rate=2e-5,
        minimum_action_std=cfg.migration.minimum_action_std)
    algorithm.init_storage(a.num_envs, a.rollout_steps, [326], [92], [15])
    a.run.mkdir(parents=True, exist_ok=True)
    (a.run/'checkpoints').mkdir(exist_ok=True)
    atomic_json(a.run/'schema.json', parent['schema'])
    config = dict(stage=a.stage, deadline=a.deadline, num_envs=a.num_envs, rollout_steps=a.rollout_steps,
        learning_rate=a.learning_rate, seed=a.seed, teacher_sha256=teacher['sha256'],
        parent_checkpoint=str(a.resume), parent_sha256=sha256(a.resume),
        algorithm='on-policy PPO; no teacher/reference execution; separate auxiliary encoder updates',
        critic_warmup_updates=a.critic_warmup_updates, optimizer_restore=False,
        command_ranges=cfg.migration.command_ranges, reward_weights=vars(cfg.rewards.scales),
        velocity_error_scale_m_s=cfg.migration.velocity_error_scale_m_s,
        unclipped_gait_costs=cfg.migration.unclipped_gait_costs,
        minimum_action_std=cfg.migration.minimum_action_std,
        novel_command_std_floor=cfg.migration.novel_command_std_floor,
        exploration_scope='Floor only for moving commands with vx<.03m/s; forward/stop retain learned noise. No change to deterministic actor.',
        command_sampling='25% old forward,10%stop,15%purelateral abs.02-.04,15%pureyaw abs.15-.35,15%purereverse -.04to-.02,20%combined' if a.stage in ('S3','S5') else 'old forward and stop',
        moving_start_curriculum=dict(enabled=cfg.migration.moving_start_training,
            fraction=cfg.migration.moving_start_fraction,hold_s=cfg.migration.moving_start_hold_s,
            blend_s=cfg.migration.moving_start_blend_s,direct_start_fraction=.25,
            evaluation_uses_direct_starts=True),
        auxiliary_labels='same-time ROOT velocity, pre-reset next clean proprioception, done mask',
        timeout_bootstrap='pre-reset terminal-state critic value; exactly once')
    atomic_json(a.run/'config.json', config)
    start = time.time()
    status = dict(status='training', stage=a.stage+'_PPO', pid=os.getpid(), started_unix=start,
        deadline=a.deadline, teacher_iteration=teacher['iteration'], accepted_locomotion=False,
        parent_checkpoint=dict(path=str(a.resume), sha256=sha256(a.resume), iteration=parent['iter']))
    iteration, control_steps = 0, 0
    last_report, last_save = 0., 0.
    metrics = {}

    def save(reason):
        cp = a.run/'checkpoints'/('model_%06d.pt'%iteration)
        if not cp.exists():
            part = cp.with_suffix('.partial')
            torch.save(dict(model_state_dict=policy.state_dict(), optimizer_state_dict=algorithm.optimizer.state_dict(),
                auxiliary_optimizer_state_dict=algorithm.auxiliary_optimizer.state_dict(),
                iter=iteration, schema=parent['schema'], teacher_sha256=teacher['sha256'],
                stage=a.stage, frequency_range=list(cfg.migration.command_ranges['frequency']), reason=reason, metrics=metrics), str(part))
            part.replace(cp)
        atomic_json(a.run/'checkpoint.json', dict(path=str(cp.resolve()), sha256=sha256(cp),
            iteration=iteration, reason=reason, updated_unix=time.time()))

    obs, critic = env.reset()
    env.last_episode_records = []
    save('initial')
    try:
        while time.time() < deadline and not STOP and not (a.run/'STOP').exists() and iteration < a.max_updates:
            warmup = iteration < a.critic_warmup_updates
            for parameter in policy.actor.parameters():
                parameter.requires_grad_(not warmup)
            policy.std.requires_grad_(not warmup)
            reward_mean = 0.
            with torch.no_grad():
                for _ in range(a.rollout_steps):
                    env.prepare_training_commands()
                    obs, critic = env.get_observations(), env.get_privileged_observations()
                    velocity = env.base_lin_vel.clone()
                    action = algorithm.act(obs, critic)
                    obs, critic, rewards, done, info = env.step(action)
                    algorithm.observe_auxiliary(velocity, info['next_clean_proprio'], done)
                    algorithm.process_env_step(rewards, done, info)
                    reward_mean += float(rewards.mean())
                    control_steps += 1
                algorithm.compute_returns(critic)
            value_loss, surrogate_loss = algorithm.update()
            iteration += 1
            assert env.teacher is None and env.reference_execution_calls == 0
            if not all(torch.isfinite(v).all() for v in policy.parameters()):
                raise RuntimeError('Nonfinite PPO parameters')
            if time.time()-last_report >= 30 or iteration == a.max_updates:
                rows = env.last_episode_records[-512:]
                survival = 1.-float(np.mean([r[1] for r in rows])) if rows else None
                # Keep the larger completed-episode window for curriculum decisions;
                # a short window can be biased by a synchronized timeout cohort.
                curriculum_rows = env.last_episode_records
                curriculum_survival = 1.-float(np.mean([r[1] for r in curriculum_rows])) if curriculum_rows else None
                frequency = next_frequency_range(a.stage,cfg.migration.command_ranges['frequency'],
                    len(curriculum_rows),curriculum_survival,time.time()-start,max(256,env.num_envs))
                if frequency != cfg.migration.command_ranges['frequency']:
                    cfg.migration.command_ranges['frequency'] = frequency
                    append_json(a.run/'events.jsonl',dict(event='frequency_range_expanded',range_hz=frequency,
                        time=time.time(),iteration=iteration,completed_episodes=len(curriculum_rows),survival_fraction=curriculum_survival,
                        independent_acceptance=False))
                metrics = dict(value_loss=value_loss, surrogate_loss=surrogate_loss,
                    mean_reward_per_control_step=reward_mean/a.rollout_steps,
                    control_steps=control_steps, environment_steps=control_steps*a.num_envs,
                    completed_episode_survival=survival, recent_completed_episodes=len(rows),
                    critic_warmup=warmup, reference_execution_calls=env.reference_execution_calls,
                    learning_rate=algorithm.learning_rate, mean_std=float(policy.std[:14].mean()),
                    novel_command_std_floor=policy.novel_command_std_floor,
                    mean_std_scope='Learned base parameters; low-forward moving-command distribution also applies the explicit novel-command floor',
                    frequency_range_hz=list(cfg.migration.command_ranges['frequency']),
                    **algorithm.last_auxiliary_metrics)
                status.update(iteration=iteration, metrics=metrics, updated_unix=time.time())
                atomic_json(a.run/'status.json', status)
                append_json(a.run/'metrics.jsonl', dict(iteration=iteration, time=time.time(), **metrics))
                print(json.dumps(status), flush=True)
                last_report = time.time()
            if time.time()-last_save >= 180:
                save('periodic')
                last_save = time.time()
        status['status'] = ('iterations_complete' if iteration >= a.max_updates else
            'stopped_at_deadline' if time.time() >= deadline else 'stopped_requested')
    except BaseException as error:
        status.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        save('final')
        status.update(iteration=iteration, updated_unix=time.time())
        atomic_json(a.run/'status.json', status)
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
