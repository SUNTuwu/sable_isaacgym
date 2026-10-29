"""Explicit task targets and positive tracking weights, separate from hand-K pose.

The turn90 profile expresses the user's requested motion, not an assertion that
the authored root trajectory or the current policy already performs 90 degrees.
"""
import math
import torch

PROFILE = 'turn90_lateral_tracking_v1'


def configure(clip_ids, raw_commands, periods, profile='legacy', balance_first=False):
    commands = raw_commands.clone()
    velocity = torch.full_like(periods, 1.5 if balance_first else 3.)
    yaw = torch.full_like(periods, 1. if balance_first else 2.)
    if profile not in ('legacy', PROFILE):
        raise ValueError('Unknown task tracking profile: ' + profile)
    for i, ident in enumerate(clip_ids):
        if 'TURN' in ident:
            commands[i, :2] = 0.
            if profile == PROFILE:
                commands[i, 2] = (1 if ident == '04_TURN_LEFT' else -1) * math.pi / (2 * periods[i])
                velocity[i], yaw[i] = 4., 8.
        elif 'LATERAL' in ident or 'STRAFE' in ident:
            commands[i, 0] = 0.; commands[i, 2] = 0.
            if profile == PROFILE:
                velocity[i], yaw[i] = 8., 4.
        else:
            commands[i, 1:] = 0.
    return commands, velocity, yaw


def tracking_terms(actual_xy, actual_yaw, commands, sigma_velocity, sigma_yaw,
                   velocity_weights, yaw_weights):
    ve = ((actual_xy - commands[:, :2]).norm(dim=1) - .005).clamp(min=0)
    we = ((actual_yaw - commands[:, 2]).abs() - .025).clamp(min=0)
    return {'velocity': velocity_weights * torch.exp(-(ve / sigma_velocity).square()),
            'yaw': yaw_weights * torch.exp(-(we / sigma_yaw).square())}


def migrate_curriculum_command(curriculum, index, old_command, new_command):
    """Keep failure history; recalibrate only statistics whose goal changed."""
    old = torch.as_tensor(old_command, device=curriculum.device, dtype=new_command.dtype)
    changed = (old - new_command).abs() > 1e-6
    if not bool(changed.any()):
        return None
    axes = []
    if bool(changed[:2].any()):
        speed = new_command[:2].norm()
        initial = torch.maximum(speed * .5, speed.new_tensor(.020))
        curriculum.sigma[index, 1] = initial.clamp(curriculum.lower[index, 1], curriculum.upper[index, 1])
        curriculum.error_ema[index, 1] = curriculum.sigma[index, 1]
        axes.append('velocity_m_s')
    if bool(changed[2]):
        initial = torch.maximum(new_command[2].abs() * .75, new_command.new_tensor(.15))
        curriculum.sigma[index, 2] = initial.clamp(curriculum.lower[index, 2], curriculum.upper[index, 2])
        curriculum.error_ema[index, 2] = curriculum.sigma[index, 2]
        axes.append('yaw_rad_s')
    curriculum.progress_ema[index] = 0.
    curriculum.success_cycle_ema[index] = 0.
    return dict(old_command=old.tolist(), new_command=new_command.tolist(),
                recalibrated_axes=axes, phase_failure_history_preserved=True,
                reason='Task command changed; reset command-dependent progress gates and reward bandwidth, not policy or Adam')


def metrics(profile, command, raw_command, period, weights, actual=None):
    result = dict(task_tracking_profile=profile, reference_period_s=float(period),
                  raw_reference_command=raw_command.tolist(),
                  task_target_yaw_per_cycle_deg=math.degrees(float(command[2] * period)),
                  task_target_xy_per_cycle_m=(command[:2] * period).tolist(),
                  tracking_reward_max_per_second=dict(velocity=float(weights[0]), yaw=float(weights[1])),
                  cycle_response_scope='Mean heading-frame velocity times reference period; equivalent response, not measured individual completed cycles')
    if actual is not None:
        result.update(equivalent_yaw_per_cycle_deg=math.degrees(float(actual[2] * period)),
                      equivalent_xy_per_cycle_m=(actual[:2] * period).tolist())
    return result
