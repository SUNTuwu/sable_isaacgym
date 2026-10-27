"""Four-terminal contact resultants and bounded task-space costs.

Runtime inputs contain measured normal forces and an independently reconstructed
whole-body resultant. Per-foot tangential forces are NOT available in this Gym
build. The full-vector branch is reserved for synthetic tests/a future verified
backend; runtime never infers zero tangential force from the normal-only tensor.
"""
import torch
from model_mapping import quat_apply,quat_inverse


def material_point_velocity(body_state, point_world, com_local=None):
    """Rigid point velocity; Gym's linear velocity is at COM (physically tested).

    Subtract rotated local COM without differentiating large world positions.
    com_local=None means COM coincides with pose origin (synthetic fixtures).
    """
    lever=point_world-body_state[...,:3]
    if com_local is not None:
        lever=lever-quat_apply(body_state[...,3:7],com_local.expand_as(lever))
    return body_state[...,7:10]+torch.cross(body_state[...,10:13],
        lever,dim=-1)


def material_point_velocity_from_poses(before, after, point_world, dt):
    """Secant velocity of the SAME rigid material point across one physics step.

    This does not differentiate the moving capsule-lowest locus, and does not
    assume whether a simulator's reported linear velocity is at COM or origin.
    """
    if dt<=0:raise ValueError('Positive physics step required')
    local=quat_apply(quat_inverse(after[...,3:7]),point_world-after[...,:3])
    previous=before[...,:3]+quat_apply(before[...,3:7],local)
    return (point_world-previous)/dt


def deadband_vector(value, threshold):
    length=value.norm(dim=-1,keepdim=True)
    return value*((length-threshold).clamp(min=0)/length.clamp(min=1e-8))


def analyze_contact_forces(force_world, point_world, body_state, root_world,
                           correction_xy_world, correction_yaw, clearance,
                           nominal_weight_n, velocity_deadband=.01,
                           yaw_deadband=.05, slip_deadband=.005, slip_scale=.04,
                           point_velocity=None, resultant_force=None,
                           resultant_valid=None, tangential_available=True):
    """Use pre-action velocity errors to interpret post-action sampled forces.

    Per-foot force dotted with the desired rigid-body correction field includes
    both translation and yaw work. Opposite forces forming a useful yaw couple
    therefore reinforce each other. Correct braking is useful when velocity is
    above its command. Vertical support does not enter directional costs.
    """
    if nominal_weight_n<=0 or slip_scale<=0:raise ValueError('Positive physical normalization required')
    contact=(force_world[...,2]>1.)&(clearance<=.006)
    force=force_world*contact[...,None]
    relative=point_world[...,:2]-root_world[:,None,:2]
    linear=deadband_vector(correction_xy_world,velocity_deadband)
    angular=correction_yaw.sign()*(correction_yaw.abs()-yaw_deadband).clamp(min=0)
    tangent=torch.stack((-relative[...,1],relative[...,0]),dim=-1)
    field=linear[:,None,:]+angular[:,None,None]*tangent
    scale=(linear.norm(dim=-1)+angular.abs()*relative.norm(dim=-1).mean(-1)).clamp(min=.01)
    effort=(force[...,:2]*field).sum(-1)
    positive=effort.clamp(min=0).sum(-1)
    negative=(-effort).clamp(min=0).sum(-1)
    cancelled=torch.minimum(positive,negative)
    wrong=(negative-positive).clamp(min=0)
    normalization=nominal_weight_n*scale
    velocity=material_point_velocity(body_state,point_world)if point_velocity is None else point_velocity
    slip_speed=velocity[...,:2].norm(dim=-1)
    slip=((slip_speed-slip_deadband).clamp(min=0)/slip_scale).square().clamp(max=4.)
    load=force[...,2].clamp(min=0)
    loaded_slip=(slip*load).sum(-1)/load.sum(-1).clamp(min=nominal_weight_n*.1)
    yaw_moment=relative[...,0]*force[...,1]-relative[...,1]*force[...,0]
    total=force.sum(1)
    wrong_cost=(wrong/normalization).clamp(max=2.)
    cancellation_cost=(cancelled/normalization).clamp(max=2.)
    if not tangential_available:
        # Absence is represented in schema/availability masks, never as evidence
        # that opposing per-foot forces or yaw moments equal zero.
        if resultant_force is None or resultant_valid is None:
            raise ValueError('Normal-only backend requires a verified resultant and validity mask')
        total=resultant_force
        linear_scale=linear.norm(dim=-1).clamp(min=.01)
        wrong=(-(total[...,:2]*linear).sum(-1)).clamp(min=0)*resultant_valid
        wrong_cost=(wrong/(nominal_weight_n*linear_scale)).clamp(max=2.)
        cancellation_cost=torch.zeros_like(wrong_cost)
        wrong_equivalent=wrong/linear_scale
    else:
        resultant_valid=torch.ones_like(wrong_cost,dtype=torch.bool)
        wrong_equivalent=wrong/scale
    return dict(contact=contact,force_world=force_world,contact_point_world=point_world,
        point_velocity_world=velocity,slip_speed_m_s=slip_speed,
        sum_force_world=total,resultant_valid=resultant_valid,
        tangential_available=torch.full_like(resultant_valid,tangential_available),yaw_moment_per_foot_nm=yaw_moment,
        sum_yaw_moment_nm=yaw_moment.sum(1),correction_field_world=field,
        directional_effort_per_foot_w=effort,positive_effort_w=positive,
        negative_effort_w=negative,cancelled_effort_w=cancelled,wrong_effort_w=wrong,
        cancelled_equivalent_force_n=cancelled/scale,wrong_equivalent_force_n=wrong_equivalent,
        sampled_friction_dissipation_w=(-(force[...,:2]*velocity[...,:2]).sum(-1)).clamp(min=0),
        wrong_cost=wrong_cost,cancellation_cost=cancellation_cost,
        loaded_slip_cost=loaded_slip)
