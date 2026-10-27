"""Rehydrate the frozen teacher configuration without mutating shared classes."""
import json
from pathlib import Path
from types import SimpleNamespace


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def teacher_config(teacher_run, num_envs=256):
    data = json.loads((Path(teacher_run) / 'config.json').read_text())
    # A few fields are mappings consumed by .items(), not config namespaces.
    cfg = namespace(data['env'])
    cfg.init_state.default_joint_angles = dict(data['env']['init_state']['default_joint_angles'])
    cfg.control.stiffness = dict(data['env']['control']['stiffness'])
    cfg.control.damping = dict(data['env']['control']['damping'])
    assets = Path(__file__).resolve().parent / 'assets'
    cfg.asset.file = str(assets / 'sable.xml')
    cfg.reference.file = str(assets / 'gait_reference.json')
    cfg.env.num_envs = num_envs
    return cfg


def student_config(teacher_run, stage='S2', num_envs=1024):
    cfg = teacher_config(teacher_run, num_envs)
    cfg.env.num_observations = 326
    cfg.env.num_privileged_obs = 92
    cfg.env.num_actions = 15
    cfg.reference.enabled = False
    cfg.normalization.clip_actions = 100.  # Physical joint limits are authoritative.
    cfg.noise.noise_level = 0.
    cfg.migration = SimpleNamespace(stage=stage, freeze_waist=True, history_noise_level=0.,
        command_ranges={'vx': [-.04, .12], 'vy': [-.04, .04], 'yaw_rate': [-.35, .35],
                        'frequency': [.9, 1.1] if stage=='S5' else [1., 1.]}, target_interpolation='hold_full_target',
        command_foot_velocity_scale_floor_m_s=.04,
        command_frontier_alpha=1., training_frontier_alpha=.25,
        velocity_error_scale_m_s=.05, moving_start_training=False,
        moving_start_fraction=.75, moving_start_hold_s=2., moving_start_blend_s=2.)
    # The full-target student can otherwise trade increased lateral oscillation
    # for the old stride/clearance costs when positive reward is clipped to zero.
    # This direct command error survives that clipping, as the yaw error does.
    cfg.rewards.scales.velocity_error = -2.
    cfg.rewards.scales.command_foot_velocity = 0.
    cfg.rewards.scales.target_pose_clearance = 0.
    cfg.migration.unclipped_gait_costs = stage in ('S3','S5')
    if stage in ('S3', 'S5'):
        cfg.commands.heading_command = False
        # Direction-dependent teacher preferences conflict with omni motion.
        cfg.rewards.scales.forward_speed = 0.
        cfg.rewards.scales.stride_progress = 0.
        cfg.rewards.scales.stride_shortfall = 0.
        cfg.rewards.scales.stride_variance = 0.
        cfg.rewards.scales.ankle_event_angle = 0.
        cfg.rewards.scales.tracking_lin_vel = 8.
        cfg.rewards.scales.tracking_ang_vel = 3.
        cfg.rewards.scales.yaw_rate_error = -2.
        # Keep per-step survival preferable to an early terminal transition
        # while restoring the clearance signal previously lost to clipping.
        cfg.rewards.scales.command_foot_velocity = -2.
        # Scalar hypothetical-pose shaping did not resolve novel-command gait;
        # use actual physical clearance while teaching a measured command path.
        cfg.rewards.scales.target_pose_clearance = 0.
        # The added bounded swing cost is about4/s for stationary novel-command
        # feet. Preserve a survival incentive while adding its directional signal.
        cfg.rewards.scales.alive = 7.
    cfg.migration.minimum_action_std = .02
    cfg.migration.novel_command_std_floor = .04 if stage=='S3' else 0.
    cfg.migration.reverse_hind_std_floor = .08 if stage=='S3' else 0.
    return cfg
