"""Named 51D proprioception, single-update history, and 326D deployment layout."""
import torch
from action_schema import ActionSpec
from command_schema import CommandSpec

PROPRIO_SIZE = 51
HISTORY_LENGTH = 6
RAW_SIZE = 326
ACTOR_SIZE = 90
CRITIC_SIZE = 92
PROPRIO_SLICES = {'angular_velocity': (0, 3), 'gravity': (3, 6),
                 'joint_position': (6, 21), 'joint_velocity': (21, 36),
                 'executed_target': (36, 51)}


def proprioception(angular_velocity, gravity, q, qd, default, executed_target,
                   angular_scale=.25, velocity_scale=.05):
    return torch.cat((angular_velocity * angular_scale, gravity, q - default,
                      qd * velocity_scale, ActionSpec.offsets(executed_target, default)), -1)


def mirror_proprio(p):
    return torch.cat((p[..., :3] * p.new_tensor([-1., 1., -1.]),
                      p[..., 3:6] * p.new_tensor([1., -1., 1.]),
                      ActionSpec.mirror(p[..., 6:21]),
                      ActionSpec.mirror(p[..., 21:36]),
                      ActionSpec.mirror(p[..., 36:51])), -1)


def pack(history, commands, clocks):
    if history.shape[-2:] != (6, 51) or commands.shape[-1] != 16 or clocks.shape[-1] != 4:
        raise ValueError('Expected history [...,6,51], commands [...,16], clocks [...,4]')
    return torch.cat((history.flatten(-2), CommandSpec.normalize(commands), clocks), -1)


def mirror_raw(raw):
    history = raw[..., :306].reshape(*raw.shape[:-1], 6, 51)
    commands = CommandSpec.physical(raw[..., 306:322])
    return pack(mirror_proprio(history), CommandSpec.mirror(commands), raw[..., [323, 322, 325, 324]])


def mirror_critic(obs):
    commands = CommandSpec.mirror(CommandSpec.physical(obs[..., 51:67]))
    force = obs[..., 80:92].reshape(*obs.shape[:-1], 4, 3)
    force = force[..., [1, 0, 3, 2], :] * obs.new_tensor([1., -1., 1.])
    return torch.cat((mirror_proprio(obs[..., :51]), CommandSpec.normalize(commands),
                      obs[..., [68, 67, 70, 69]], obs[..., 71:74] * obs.new_tensor([1., -1., 1.]),
                      obs[..., 74:76], obs[..., [77, 76, 79, 78]], force.flatten(-2)), -1)


class HistoryBuffer:
    def __init__(self, num_envs, device):
        self.data = torch.zeros(num_envs, 6, 51, device=device)
        self.needs_fill = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.last_sample = None

    def reset(self, ids):
        self.needs_fill[ids] = True

    def sample(self, current, sample_id):
        if self.last_sample == sample_id:
            ids = self.needs_fill.nonzero(as_tuple=False).flatten()
            self.data[ids] = current[ids, None, :].expand(-1, 6, -1)
            self.needs_fill[ids] = False
            return self.data
        if current.shape != self.data[:, 0].shape:
            raise ValueError('Wrong proprioception shape')
        self.data[:, 1:] = self.data[:, :-1].clone()
        self.data[:, 0] = current
        ids = self.needs_fill.nonzero(as_tuple=False).flatten()
        self.data[ids] = current[ids, None, :].expand(-1, 6, -1)
        self.needs_fill.zero_()
        self.last_sample = sample_id
        return self.data


def metadata():
    return {'proprioception_size': 51, 'history_length': 6, 'raw_size': 326,
            'actor_encoded_size': 90, 'critic_size': 92, 'critic_schema': 'critic_v1_92',
            'proprioception_slices': PROPRIO_SLICES, 'history_order': 'current_to_past',
            'reset_fill': 'repeat_new_initial_proprioception_six_times',
            'raw_slices': {'history': [0, 306], 'commands': [306, 322], 'clocks': [322, 326]},
            'angular_velocity_scale': .25, 'joint_velocity_scale': .05,
            'velocity_estimator_units': 'ROOT m/s', 'actor_velocity_scale': 2.,
            'executed_target': 'last_substep_clamped_complete_target_offset',
            'critic_contact_force': 'net rigid-body contact resultant in ROOT coordinates, newtons'}
