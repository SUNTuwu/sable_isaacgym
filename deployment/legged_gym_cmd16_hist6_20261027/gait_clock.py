"""Continuous phase integration; changing frequency cannot reset phase."""
import math
import torch


def shortest_phase_delta(target, current):
    return torch.remainder(target - current + .5, 1.) - .5


def local_phases(phase, commands):
    offsets = torch.cat((torch.zeros_like(commands[..., :1]), commands[..., 6:9]), -1)
    return torch.remainder(phase[..., None] + offsets, 1.)


def clocks(phase, commands):
    local = local_phases(phase, commands)
    beta = commands[..., 9:10]
    u = torch.where(local < beta, local / (2 * beta), .5 + (local - beta) / (2 * (1 - beta)))
    return torch.sin(2 * math.pi * u)


class GaitClock:
    def __init__(self, num_envs, device, dt=.02):
        self.phase = torch.zeros(num_envs, device=device)
        self.moving = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.dt = dt

    def reset_from_teacher(self, ids, old_phase):
        self.phase[ids] = torch.remainder(old_phase + .75, 1.)
        self.moving[ids] = True

    def advance(self, commands, stop_curriculum=False):
        if stop_curriculum:
            # Small robot-specific initial thresholds; pure yaw is motion.
            intent = torch.maximum(commands[:, :2].norm(dim=1) / .01, commands[:, 2].abs() / .08)
            self.moving = torch.where(self.moving, intent > .6, intent > 1.)
        else:
            self.moving[:] = True
        self.phase = torch.remainder(self.phase + commands[:, 5] * self.dt * self.moving, 1.)

    def values(self, commands, stop_curriculum=False):
        values = clocks(self.phase, commands)
        return values * self.moving[:, None] if stop_curriculum else values

    def smooth_offsets(self, current, target, tau_s=.2):
        alpha = 1. - math.exp(-self.dt / tau_s)
        return torch.remainder(current + alpha * shortest_phase_delta(target, current), 1.)
