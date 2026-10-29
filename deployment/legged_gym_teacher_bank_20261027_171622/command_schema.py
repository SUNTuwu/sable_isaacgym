"""Versioned physical command interface. Normalization never changes its semantics."""
import torch

SCHEMA_VERSION = 'sable_cmd16_hist6_v1'
COMMAND_NAMES = (
    'vx', 'vy', 'yaw_rate', 'shoulder_height', 'hip_height', 'gait_frequency',
    'phase_fr', 'phase_hl', 'phase_hr', 'stance_fraction', 'swing_height',
    'chest_roll', 'waist_roll', 'stance_width', 'waist_pitch', 'aux_reward_coeff',
)
COMMAND_UNITS = ('m/s', 'm/s', 'rad/s', 'm', 'm', 'Hz', 'cycle', 'cycle',
                 'cycle', '1', 'm', 'rad', 'rad', 'm', 'rad', '1')
COMMAND_SCALES = (.2, .1, 1., .1, .1, 1., 1., 1., 1., 1., .05, 1., 1., .1, 1., 1.)


class CommandSpec:
    names = COMMAND_NAMES
    units = COMMAND_UNITS
    size = 16

    @staticmethod
    def normalize(commands):
        return commands / commands.new_tensor(COMMAND_SCALES)

    @staticmethod
    def physical(normalized):
        return normalized * normalized.new_tensor(COMMAND_SCALES)

    @staticmethod
    def defaults(shoulder_height, hip_height, stance_width, waist_pitch):
        return [.105, 0., 0., shoulder_height, hip_height, 1., .5, .25, .75,
                .5, .023, 0., 0., stance_width, waist_pitch, 1.]

    @staticmethod
    def validate(commands):
        if commands.shape[-1] != 16 or not torch.isfinite(commands).all():
            raise ValueError('Commands must be finite [...,16] physical values')
        if not ((commands[..., 5] > 0) & (commands[..., 9] > 0) & (commands[..., 9] < 1)).all():
            raise ValueError('Frequency must be positive and stance fraction strictly inside (0,1)')
        if not (commands[..., 15] == 1).all():
            raise ValueError('aux_reward_coeff is reserved and must remain 1 in v1')

    @staticmethod
    def mirror(commands):
        """Reflect physical commands and rebase phases on the new left-front foot."""
        result = commands.clone()
        result[..., [1, 2, 11, 12]] *= -1
        fr, hl, hr = commands[..., 6], commands[..., 7], commands[..., 8]
        result[..., 6] = torch.remainder(-fr, 1.)
        result[..., 7] = torch.remainder(hr - fr, 1.)
        result[..., 8] = torch.remainder(hl - fr, 1.)
        return result

    @staticmethod
    def metadata():
        return {'names': list(COMMAND_NAMES), 'units': list(COMMAND_UNITS),
                'normalization_divisors': list(COMMAND_SCALES),
                'coordinate_frame': 'heading-aligned horizontal; S0-S2 teacher body-frame compatibility',
                'reserved_fixed': {'aux_reward_coeff': 1.}}
