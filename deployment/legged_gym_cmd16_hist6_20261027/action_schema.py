"""Independent motors only; the opposed right hip is never a policy action."""
import torch

ACTION_NAMES = (
    'spine_roll', 'hip_roll', 'L_shoulder_roll', 'L_shoulder_pitch',
    'L_elbow_pitch', 'L_hip_pitch', 'L_knee_pitch', 'R_shoulder_roll',
    'R_shoulder_pitch', 'R_elbow_pitch', 'R_hip_pitch', 'R_knee_pitch',
    'L_ankle_pitch', 'R_ankle_pitch', 'spine_pitch',
)
ACTION_INDEX = (0, 1, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 13, 12, 14)
ACTION_SIGN = (-1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1)
FOOT_NAMES = ('L_wrist_fixed', 'R_wrist_fixed', 'L_ankle_pitch', 'R_ankle_pitch')
FOOT_LABELS = ('FL', 'FR', 'HL', 'HR')


class ActionSpec:
    names = ACTION_NAMES
    size = 15
    scales = (.35,) * 15

    @staticmethod
    def indices(dof_names, device):
        if len(set(dof_names)) != len(dof_names):
            raise ValueError('Duplicate DOF names')
        return torch.tensor([dof_names.index(n) for n in ACTION_NAMES], device=device)

    @staticmethod
    def mirror(values):
        return values[..., list(ACTION_INDEX)] * values.new_tensor(ACTION_SIGN)

    @staticmethod
    def target(actions, default, limits, freeze_waist=False):
        if actions.shape[-1] != 15:
            raise ValueError('Expected 15 independent target offsets')
        if freeze_waist:
            actions = torch.cat((actions[..., :14], torch.zeros_like(actions[..., 14:])), -1)
        raw = default + actions * actions.new_tensor(ActionSpec.scales)
        return torch.maximum(torch.minimum(raw, limits[..., 1]), limits[..., 0])

    @staticmethod
    def offsets(target, default):
        return (target - default) / target.new_tensor(ActionSpec.scales)

    @staticmethod
    def metadata():
        return {'names': list(ACTION_NAMES), 'scale_rad': list(ActionSpec.scales),
                'definition': 'clamp(q_default + scale * action, physical_joint_limits)',
                'passive_joint': 'R_hip_roll_mimic', 'passive_motor_torque': 0.,
                'feet': list(FOOT_NAMES), 'foot_labels': list(FOOT_LABELS),
                'mirror_index': list(ACTION_INDEX), 'mirror_sign': list(ACTION_SIGN)}
