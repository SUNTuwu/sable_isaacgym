"""Soft task-space velocity shaping; never an actuator execution reference."""
import math
import torch


def command_velocity_at_feet(commands_xyz,forward_xy,root_xy,foot_xy):
    left=torch.stack((-forward_xy[:,1],forward_xy[:,0]),-1)
    translation=commands_xyz[:,0:1]*forward_xy+commands_xyz[:,1:2]*left
    offset=foot_xy-root_xy[:,None,:]
    tangent=torch.stack((-offset[:,:,1],offset[:,:,0]),-1)
    return translation[:,None,:]+commands_xyz[:,2,None,None]*tangent


def desired_foot_velocity(commands_xyz, forward_xy, root_xy, foot_xy, phase, duty, activation):
    """World-horizontal capsule-center velocity compatible with the command.

    A stationary stance and a smooth swing pulse have mean foot velocity equal
    to the commanded rigid-body velocity at that foot. For constant translation,
    integrating one cycle gives command_speed/frequency, without an 80mm floor.
    Yaw uses the instantaneous foot offset about ROOT, not a fixed forward axis.
    This is a soft reward target, not a prescribed touchdown point/joint path.
    """
    rigid_velocity=command_velocity_at_feet(commands_xyz,forward_xy,root_xy,foot_xy)
    progress=((phase-duty)/(1-duty)).clamp(0,1)
    pulse=(math.pi/2)*torch.sin(math.pi*progress)/(1-duty)
    pulse*=((phase>=duty)&(phase<1)).to(pulse.dtype)
    return rigid_velocity*pulse[:,:,None]*activation[:,None,None]


class FootCenterVelocity:
    """20ms center displacement, including capsule rotation; reset-safe."""
    def __init__(self,count,device,dt):
        self.dt=dt
        self.previous=torch.zeros(count,4,2,device=device)
        self.known=torch.zeros(count,dtype=torch.bool,device=device)
        self.velocity=torch.zeros_like(self.previous)
        self.valid=torch.zeros_like(self.known)

    def reset(self,ids):
        self.known[ids]=False
        self.valid[ids]=False
        self.velocity[ids]=0.

    def update(self,centers_xy):
        self.valid.copy_(self.known)
        self.velocity.copy_((centers_xy-self.previous)/self.dt)
        self.velocity*=self.valid[:,None,None]
        self.previous.copy_(centers_xy)
        self.known[:]=True
        return self.velocity


def foot_velocity_cost(actual,target,scale_m_s=.1):
    if isinstance(scale_m_s,torch.Tensor):scale_m_s=scale_m_s[:,:,None]
    return ((actual-target)/scale_m_s).square().sum(-1).clamp(max=9.)
