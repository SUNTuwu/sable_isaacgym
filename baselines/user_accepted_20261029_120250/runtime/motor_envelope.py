"""The existing motor envelope, scripted to reduce per-substep dispatch overhead."""
import math
import torch


@torch.jit.script
def motor_torques(target: torch.Tensor, q: torch.Tensor, velocity: torch.Tensor,
                  kp: torch.Tensor, kd: torch.Tensor, torque_limits: torch.Tensor,
                  max_speed: torch.Tensor, large: torch.Tensor, dependent: int):
    wanted = kp*(target-q)-kd*velocity
    rpm = velocity.abs()*60/(2*math.pi)
    large_limit = torch.where(rpm <= 245.1, 1.95-(1.95-.78)*rpm/245.1,
        .78*(421.3-rpm)/(421.3-245.1)).clamp(min=0)
    small_limit = torque_limits*(1-velocity.abs()/max_speed).clamp(min=0)
    driving_limit = torch.where(large, large_limit, small_limit)
    limit = torch.where(wanted*velocity < 0, torque_limits, driving_limit)
    torque = torch.maximum(torch.minimum(wanted, limit), -limit)
    torque[:, dependent] = 0.
    return torque
