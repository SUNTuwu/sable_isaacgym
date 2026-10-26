"""Per-foot horizontal force/body-velocity contribution, not contact-point work.

Force is the force ON the foot. Both vectors are in world coordinates.
Integrate at every physics step; split positive/negative before averaging.
Motor power and frictional contact-point power are different quantities.
"""
import torch


def propulsion_terms(force_world, body_velocity_world, contact, tracking_gate,
                     eligible, power_scale_w=.05, braking_weight=2.):
    raw = (force_world[..., :2] * body_velocity_world[:, None, :2]).sum(-1)
    power = torch.where(contact, raw, torch.zeros_like(raw))
    positive = power.clamp(min=0.)
    braking = (-power).clamp(min=0.)
    # Cap only the positive incentive; large opposing forces remain costly.
    per_foot = (positive / power_scale_w).clamp(max=1.) * tracking_gate[:, None]
    per_foot -= braking_weight * braking / power_scale_w
    per_foot *= eligible[:, None]
    return power, positive, braking, per_foot


class LegPropulsion:
    def __init__(self, env):
        self.env = env
        self.count = 0
        self.power = torch.zeros(env.num_envs, 4, device=env.device)
        self.positive = torch.zeros_like(self.power)
        self.braking = torch.zeros_like(self.power)
        self.reward_per_foot = torch.zeros_like(self.power)
        self.force = torch.zeros(env.num_envs, 4, 3, device=env.device)
        self.body_velocity = torch.zeros(env.num_envs, 3, device=env.device)
        self.contact = torch.zeros_like(self.power)
        self.joint_power = torch.zeros(env.num_envs, env.num_dof, device=env.device)

    def begin(self):
        self.count = 0
        for value in [self.power, self.positive, self.braking, self.reward_per_foot,
                      self.force, self.body_velocity, self.contact, self.joint_power]:
            value.zero_()

    def sample(self):
        from isaacgym.torch_utils import quat_rotate_inverse
        e = self.env
        e.gym.refresh_actor_root_state_tensor(e.sim)
        e.gym.refresh_net_contact_force_tensor(e.sim)
        e.gym.refresh_force_sensor_tensor(e.sim)
        e.gym.refresh_rigid_body_state_tensor(e.sim)
        force = e.foot_sensor_forces
        state = e.rigid_states[:, e.feet_indices]
        endpoints = e.gait_signals.endpoints.unsqueeze(0).expand(e.num_envs, -1, -1, -1)
        q = state[:, :, None, 3:6].expand(-1, -1, 2, -1)
        w = state[:, :, None, 6:7]
        rotated = endpoints + 2 * torch.cross(q, torch.cross(q, endpoints, dim=-1) + w * endpoints, dim=-1)
        clearance = state[:, :, 2] + rotated[:, :, :, 2].min(2).values - e.gait_signals.radii
        contact = (force[..., 2] > e.cfg.propulsion.contact_force_n) & (clearance <= e.cfg.propulsion.contact_margin_m)
        velocity = e.root_states[:, 7:10]
        local = quat_rotate_inverse(e.base_quat, velocity)
        error = (local[:, :2] - e.commands[:, :2]).square().sum(1)
        tracking = torch.exp(-error / e.cfg.rewards.tracking_sigma)
        eligible = (e.commands[:, 0] > .02) & (e.episode_length_buf * e.dt >= 1.)
        power, positive, braking, reward = propulsion_terms(
            force, velocity, contact, tracking, eligible,
            e.cfg.propulsion.power_scale_w, e.cfg.propulsion.braking_weight)
        self.power += power
        self.positive += positive
        self.braking += braking
        self.reward_per_foot += reward
        self.force += force
        self.body_velocity += velocity
        self.contact += contact.float()
        self.joint_power += e.torques * e.dof_vel
        self.count += 1

    def finish(self):
        assert self.count == self.env.cfg.control.decimation
        for value in [self.power, self.positive, self.braking, self.reward_per_foot,
                      self.force, self.body_velocity, self.contact, self.joint_power]:
            value.div_(self.count)

    def reset(self, ids):
        for value in [self.power, self.positive, self.braking, self.reward_per_foot,
                      self.force, self.body_velocity, self.contact, self.joint_power]:
            value[ids] = 0.

    def snapshot(self):
        return {name: getattr(self, name).clone() for name in
                ['power', 'positive', 'braking', 'reward_per_foot', 'force',
                 'body_velocity', 'contact', 'joint_power']}
