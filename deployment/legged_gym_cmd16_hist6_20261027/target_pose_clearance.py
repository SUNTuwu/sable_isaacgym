"""Soft reward for swing clearance of a hypothetical executed-target pose.

The root pose is measured, joint coordinates are the clipped executed targets,
and the dependent hip coordinate follows the same opposed coupling. This is a
reward-space geometric signal only; it never changes actuator targets or counts
as actual foot clearance, contact, or a completed step.
"""
import torch


def target_pose_clearance_cost(clearance, desired_clearance, swing_weight, height):
    shortfall = (desired_clearance - clearance).clamp(min=0.) / height
    return shortfall.square().clamp(max=9.) * swing_weight


def executed_target_coordinates(measured_q, active, executed_target, hip, dependent):
    q = measured_q.clone()
    q[:, active] = executed_target
    q[:, dependent] = -q[:, hip]
    return q
