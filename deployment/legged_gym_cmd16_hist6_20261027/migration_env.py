"""Sable physics with the new deployment interface and complete-target control.

The original asset, gains, torque-speed envelope and passive tendon are retained.
Teacher/reference queries exist only in the explicit distillation attachment.
"""
import math
from pathlib import Path
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.base.legged_robot import LeggedRobot
from sable_env import SableRobot
from gait_support import GaitSignals
from command_schema import CommandSpec
from action_schema import ActionSpec
from observation_schema import HistoryBuffer, proprioception, pack
from gait_clock import GaitClock, local_phases
from model_mapping import ModelMapping, heading_velocity, quat_apply, quat_inverse
from curriculum import sample_velocity_commands


class TaskGaitSignals(GaitSignals):
    def phase(self, env):
        return torch.remainder(env.clock.phase - .75, 1.)

    def update(self, env):
        super().update(env)  # Geometry from unchanged capsule endpoints.
        c = env.commands16
        local = local_phases(env.clock.phase, c)
        beta = c[:, 9:10]
        progress = ((local-beta)/(1-beta)).clamp(0, 1)
        age = env.episode_length_buf * env.dt
        ramp = (age / env.cfg.reference.ramp_s).clamp(0, 1)
        ramp = ramp.square() * (3-2*ramp)
        activation = (env.task_speed/.03).clamp(0, 1) * ramp
        if env.omni:
            activation *= env.clock.moving
        swing = (local >= beta) & (env.task_speed[:, None] > .001)
        env.gait_swing_weight = (progress/.15).clamp(0, 1)*((1-progress)/.15).clamp(0, 1)*swing*activation[:, None]
        env.gait_target_clearance = torch.sin(math.pi*progress).square()*c[:, 10:11]*swing*activation[:, None]
        env.task_contact_activation = activation


class MigrationRobot(SableRobot):
    def _get_noise_scale_vec(self, cfg):
        self.add_noise = cfg.noise.add_noise
        return torch.zeros(326, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self.active = ActionSpec.indices(self.dof_names, self.device)
        self.omni = self.cfg.migration.stage in ('S3', 'S5')
        self.mapping = ModelMapping(self.cfg.asset.file, Path(self.cfg.asset.file).with_name('physical_manifest.json'), self.device)
        self.nominal_measurements = self.mapping.default_measurements()
        m = self.nominal_measurements
        self.command_defaults = torch.tensor(CommandSpec.defaults(m['shoulder_height_m'], m['hip_height_m'],
            m['width0_m'], float(self.default_dof_pos[0, self.waist])), device=self.device)
        self.commands16 = self.command_defaults.repeat(self.num_envs, 1)
        self.command_override = None
        self.clock = GaitClock(self.num_envs, self.device, self.dt)
        self.gait_signals = TaskGaitSignals(self)
        self.task_speed = torch.zeros(self.num_envs, device=self.device)
        self.history = HistoryBuffer(self.num_envs, self.device)
        self.executed_target = self.default_dof_pos[:, self.active].expand(self.num_envs, -1).clone()
        self.student_target = self.executed_target.clone()
        self.teacher = None
        self.student_fraction = 1.
        self.reference_execution_calls = 0
        self.substep_index = 0
        self.record_substep_targets = False
        self.substep_targets = []
        self.clean_proprio = torch.zeros(self.num_envs, 51, device=self.device)
        self.terminal_critic = torch.zeros(self.num_envs, 92, device=self.device)
        self.terminal_proprio = self.clean_proprio.clone()
        self.terminal_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.teacher_target_label = torch.zeros(self.num_envs, 15, device=self.device)

    def attach_teacher(self, checkpoint, sha256, student_fraction=0.):
        if self.omni:
            raise ValueError('The forward teacher cannot supervise the new omnidirectional curriculum')
        from teacher_adapter import FrozenTeacher
        self.teacher = FrozenTeacher(self, checkpoint, sha256)
        self.student_fraction = float(student_fraction)

    def detach_teacher(self):
        self.teacher = None
        self.student_fraction = 1.

    def _refresh_proprio_state(self):
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.heading_linear_velocity, self.heading_yaw_rate = heading_velocity(self.base_quat,
            self.root_states[:, 7:10], self.root_states[:, 10:13])

    def _proprio(self):
        return proprioception(self.base_ang_vel, self.projected_gravity, self.dof_pos[:, self.active],
            self.dof_vel[:, self.active], self.default_dof_pos[:, self.active], self.executed_target,
            self.obs_scales.ang_vel, self.obs_scales.dof_vel)

    def _critic(self, clean):
        heights, roll, feet, lowest = self.mapping.task_measurements(self.root_states[:, :3],
            self.base_quat, self.dof_pos, self.dof_names)
        self.task_heights = heights
        self.task_chest_roll = roll
        self.task_foot_centers = feet
        force = quat_apply(quat_inverse(self.base_quat)[:, None].expand(-1, 4, -1),
                           self.contact_forces[:, self.feet_indices])
        return torch.cat((clean, CommandSpec.normalize(self.commands16), self.clock.values(self.commands16, self.omni),
                          self.base_lin_vel, heights, lowest, force.flatten(1)), -1)

    def compute_observations(self):
        self._refresh_proprio_state()
        if not self.omni:
            self.commands16[:, :3] = self.commands[:, :3]
        self.clean_proprio = self._proprio()
        current = self.clean_proprio.clone()
        level = self.cfg.migration.history_noise_level
        if self.add_noise and level:
            scales = self.cfg.noise.noise_scales
            amplitude = torch.zeros(51, device=self.device)
            amplitude[:3] = scales.ang_vel*self.obs_scales.ang_vel
            amplitude[3:6] = scales.gravity
            amplitude[6:21] = scales.dof_pos*self.obs_scales.dof_pos
            amplitude[21:36] = scales.dof_vel*self.obs_scales.dof_vel
            current += (2*torch.rand_like(current)-1)*amplitude*level
        self.history.sample(current, self.common_step_counter)
        self.obs_buf = pack(self.history.data, self.commands16, self.clock.values(self.commands16, self.omni))
        self.privileged_obs_buf = self._critic(self.clean_proprio)

    def _resample_commands(self, ids):
        if not hasattr(self, 'commands16'):
            return super()._resample_commands(ids)
        if self.command_override is not None:
            self.commands16[ids] = self.command_override[ids]
        elif self.fixed_command is not None:
            self.commands16[ids] = self.command_defaults
            self.commands16[ids, 0] = self.fixed_command
        else:
            sample_velocity_commands(self.commands16, ids, self.cfg.migration.stage, self.cfg.migration.command_ranges)
        self.commands[ids, :3] = self.commands16[ids, :3]
        self.commands[ids, 3] = 0.

    def set_commands(self, commands):
        CommandSpec.validate(commands)
        if commands.shape != self.commands16.shape:
            raise ValueError('Expected commands for every environment')
        self.command_override = commands.clone()
        self.commands16.copy_(commands)
        self.commands[:, :3] = commands[:, :3]
        # Caller can change frequency without touching the integrated phase.
        self.compute_observations()

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if not len(ids) or not hasattr(self, 'history'):
            return
        self.executed_target[ids] = self.default_dof_pos[:, self.active]
        self.task_speed[ids] = 0.
        self.clock.reset_from_teacher(ids, self.gait_phase_offset[ids])
        self.history.reset(ids)
        if self.teacher is not None:
            self.teacher.reset(ids)

    def _reset_dofs(self, ids):
        active = self.active
        if self.cfg.migration.freeze_waist:
            self.active = active[:14]
        try:
            super()._reset_dofs(ids)
        finally:
            self.active = active

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh_proprio_state()
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions):
        if self.teacher is not None:
            self.teacher_target_label = self.teacher.prepare()
        self.student_target = ActionSpec.target(actions, self.default_dof_pos[:, self.active],
            self.hard_limits[self.active], self.cfg.migration.freeze_waist)
        wanted = self.commands[:, 0].clamp(0, .09)
        if self.omni:
            wanted = torch.maximum(self.commands16[:, :2].norm(dim=1), self.commands16[:, 2].abs()*.06)
        self.task_speed += (1-math.exp(-self.dt/self.cfg.reference.speed_filter_tau_s))*(wanted-self.task_speed)
        self.clock.advance(self.commands16, self.omni)
        self.substep_index = 0
        self.substep_targets = []
        return super().step(actions)

    def _compute_torques(self, actions):
        self.substep_index += 1
        target = self.default_dof_pos.expand(self.num_envs, -1).clone()
        target[:, self.active] = self.student_target
        if self.teacher is not None and self.student_fraction < 1.:
            teacher_target = self.teacher.target(self.substep_index/self.cfg.control.decimation)
            target = (1-self.student_fraction)*teacher_target + self.student_fraction*target
            self.reference_execution_calls += 1
        target = torch.maximum(torch.minimum(target, self.hard_limits[:, 1]), self.hard_limits[:, 0])
        self.executed_target.copy_(target[:, self.active])
        if self.record_substep_targets:
            self.substep_targets.append(target[:2].clone())
        if self.teacher is not None and self.substep_index == self.cfg.control.decimation:
            self.teacher.finish(self.executed_target, self.student_fraction == 0.)
        wanted = self.p_gains*(target-self.dof_pos)-self.d_gains*self.dof_vel
        v = self.dof_vel
        rpm = v.abs()*60/(2*math.pi)
        large_limit = torch.where(rpm <= 245.1, 1.95-(1.95-.78)*rpm/245.1,
            .78*(421.3-rpm)/(421.3-245.1)).clamp(min=0)
        small_limit = self.torque_limits*(1-v.abs()/self.max_speed).clamp(min=0)
        driving_limit = torch.where(self.large, large_limit, small_limit)
        limit = torch.where(wanted*v < 0, self.torque_limits, driving_limit)
        torque = torch.maximum(torch.minimum(wanted, limit), -limit)
        torque[:, self.dependent] = 0.
        return torque

    def _post_physics_step_callback(self):
        LeggedRobot._post_physics_step_callback(self)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._refresh_proprio_state()
        if not self.omni:
            self.commands16[:, :3] = self.commands[:, :3]
        self.waist_height_acc.update(self.rigid_states[:, self.waist_body_index, 2])
        self.gait_signals.update(self)
        self.foot_pair_distance = torch.stack([(self.foot_center_world[:, 0, :2]-self.foot_center_world[:, 1, :2]).norm(dim=1),
            (self.foot_center_world[:, 2, :2]-self.foot_center_world[:, 3, :2]).norm(dim=1)], 1)
        local = local_phases(self.clock.phase, self.commands16)
        eligible = (self.task_speed > .02) & (self.episode_length_buf*self.dt >= self.cfg.reference.ramp_s)
        target = self.commands16[:, 10:11].expand(-1, 4)
        self.swing_peak_balance_cost = self.peak_balance.update(self.foot_clearance,
            local >= self.commands16[:, 9:10], eligible, target)
        q = self.base_quat
        x, y, z, w = q.unbind(1)
        forward = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z)), 1)
        forward /= forward.norm(dim=1, keepdim=True).clamp(min=1e-6)
        age = self.episode_length_buf*self.dt
        moving_command = self.commands[:, 0]
        if self.omni:
            moving_command = torch.maximum(self.commands16[:, :2].norm(dim=1), self.commands16[:, 2].abs()*.06)
        self.stride_progress.update(self.foot_center_world[:, :, :2], self.contact_forces[:, self.feet_indices, 2] > 1.,
            self.foot_clearance, forward, self.commands[:, 0], age, self.base_lin_vel[:, 0])
        axis = self.foot_long_axis_world[:, 2:]
        self.hind_foot_angle_deg = torch.acos((axis[:, :, :2]*forward[:, None]).sum(-1).clamp(-1, 1))*180/math.pi
        self.hind_landing_forward_m = ((self.foot_contact_point[:, 2:, :2]-self.rigid_states[:, self.hind_hip_bodies, :2])*forward[:, None]).sum(-1)
        contact = (self.contact_forces[:, self.feet_indices[2:], 2] > 1.) & (self.foot_clearance[:, 2:] <= .001)
        self.ankle_stance.update(self.dof_pos[:, self.hind_ankle_indices], contact, self.foot_clearance[:, 2:],
            self.hind_foot_angle_deg, moving_command, self.base_lin_vel[:, 0], age)
        self.ankle_events.update(contact, self.foot_clearance[:, 2:], self.hind_foot_angle_deg,
            self.foot_center_world[:, 2:, :2], moving_command, age)
        self.hind_width.update(self.foot_contact_point[:, 2:, :2], contact, self.stride_progress.hind_landing,
            forward, moving_command, age)
        self.knee_contact_detector.update()
        velocity = self.heading_linear_velocity if self.omni else self.base_lin_vel[:, :2]
        self.ep_velocity_error += (velocity[:, 0]-self.commands[:, 0]).square()
        self.ep_steps += 1
        feet_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.ep_touchdowns += ((~self.ep_last_contact) & feet_contact).float()
        self.ep_last_contact.copy_(feet_contact)
        self.ep_peak_coupling = torch.maximum(self.ep_peak_coupling, (self.dof_pos[:, self.hip]+self.dof_pos[:, self.dependent]).abs())

    def check_termination(self):
        super().check_termination()
        self.terminal_proprio = self._proprio().clone()
        self.terminal_velocity = self.base_lin_vel.clone()
        self.terminal_critic = self._critic(self.terminal_proprio).clone()
        self.extras['terminal_critic'] = self.terminal_critic
        self.extras['next_clean_proprio'] = self.terminal_proprio
        self.extras['next_velocity'] = self.terminal_velocity

    def _reward_target_limits(self):
        raw = self.default_dof_pos[:, self.active] + self.actions*.35
        if self.cfg.migration.freeze_waist:
            raw[:, -1] = self.default_dof_pos[:, self.waist]
        limits = self.hard_limits[self.active]
        excess = (limits[:, 0]-raw).clamp(min=0)+(raw-limits[:, 1]).clamp(min=0)
        return (excess/.35).square().sum(1)

    def _reward_gait_clearance(self):
        return ((self.foot_clearance-self.gait_target_clearance)/self.commands16[:, 10:11]).square().clamp(max=9.).mean(1)

    def _reward_tracking_lin_vel(self):
        velocity = self.heading_linear_velocity if self.omni else self.base_lin_vel[:, :2]
        return torch.exp(-(self.commands[:, :2]-velocity).square().sum(1)/self.cfg.rewards.tracking_sigma)

    def _reward_yaw_rate_error(self):
        actual = self.heading_yaw_rate if self.omni else self.base_ang_vel[:, 2]
        return (actual-self.commands[:, 2]).square()

    def _reward_tracking_ang_vel(self):
        return torch.exp(-self._reward_yaw_rate_error()/.25)

    def _reward_feet_air_time(self):
        if not self.omni:
            return super()._reward_feet_air_time()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        filtered = contact | self.last_contacts
        self.last_contacts.copy_(contact)
        first = (self.feet_air_time > 0.) & filtered
        self.feet_air_time += self.dt
        reward = ((self.feet_air_time-getattr(self.cfg.rewards, 'foot_air_time_target', .2))*first).sum(1)*self.clock.moving
        self.feet_air_time *= ~filtered
        return reward


def create_migration_env(cfg, noise=True):
    cfg.noise.add_noise = noise
    cfg.domain_rand.randomize_friction = noise
    params = gymapi.SimParams()
    params.dt = cfg.sim.dt
    params.substeps = cfg.sim.substeps
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    params.physx.num_threads = 4
    params.physx.solver_type = 1
    for key in ('num_position_iterations', 'num_velocity_iterations', 'contact_offset', 'rest_offset',
                'max_depenetration_velocity', 'bounce_threshold_velocity'):
        setattr(params.physx, key, getattr(cfg.sim.physx, key))
    params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
    params.physx.max_gpu_contact_pairs = 2**22
    params.physx.default_buffer_size_multiplier = 5.
    return MigrationRobot(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
