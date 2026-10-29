"""Physical heading-rate tracking with an explicit cost normalization."""
import math
import torch


def yaw_error_cost(error_rad_s, scale_rad_s):
    if not math.isfinite(scale_rad_s) or scale_rad_s <= 0:
        raise ValueError('Yaw error scale must be finite and positive')
    return (error_rad_s/scale_rad_s).square()


def angular_tracking_reward(error_rad_s):
    # Preserve the existing Gaussian in physical units. Changing only the cost
    # scale must not accidentally narrow this independent positive reward.
    return torch.exp(-error_rad_s.square()/.25)
