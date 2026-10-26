"""Left/right reflection for the current robot, including opposed rear roll.

The half-cycle phase shift swaps the four-beat crawl's left and right feet.
This module does not change or train a policy by itself.
"""
import numpy as np

NAMES=['spine_roll','hip_roll','L_shoulder_roll','L_shoulder_pitch','L_elbow_pitch','L_hip_pitch','L_knee_pitch',
       'R_shoulder_roll','R_shoulder_pitch','R_elbow_pitch','R_hip_pitch','R_knee_pitch','L_ankle_pitch','R_ankle_pitch']
ACTION_INDEX=[0,1,7,8,9,10,11,2,3,4,5,6,13,12]
ACTION_SIGN=[-1,1,-1,1,1,1,1,-1,1,1,1,1,1,1]

def indices(obs_dim):
    assert obs_dim in [54,56,58]
    index=list(range(obs_dim));sign=[1.]*obs_dim
    # Polar velocity, axial angular velocity, gravity, planar velocity/yaw commands.
    for i in [1,3,5,7,10,11]:sign[i]=-1.
    for start in [12,26,40]:
        index[start:start+14]=[start+i for i in ACTION_INDEX]
        sign[start:start+14]=ACTION_SIGN
    if obs_dim>=56:sign[54:56]=[-1.,-1.]
    return index,sign

class Mirror:
    def __init__(self,names,obs_dim,device):
        import torch
        assert names==NAMES,names
        oi,os=indices(obs_dim)
        self.oi=torch.tensor(oi,device=device);self.os=torch.tensor(os,device=device)
        self.ai=torch.tensor(ACTION_INDEX,device=device);self.a_s=torch.tensor(ACTION_SIGN,device=device)
    def observations(self,x):return x[...,self.oi]*self.os
    def actions(self,x):return x[...,self.ai]*self.a_s

def mirrored_coordinates(q):
    result=dict(q);result['spine_roll']=-q['spine_roll']
    # q_R=-q_L and both axes are +X: reflection preserves the single actuator q_L.
    result['hip_roll']=q['hip_roll'];result['R_hip_roll_mimic']=-q['hip_roll']
    for left in [n for n in q if n.startswith('L_')]:
        right='R_'+left[2:]
        sign=-1 if left=='L_shoulder_roll' else 1
        result[left]=sign*q[right];result[right]=sign*q[left]
    return result

def numpy_involution_error(obs_dim):
    i,s=indices(obs_dim);s=np.asarray(s);x=np.random.default_rng(947).normal(size=(17,obs_dim))
    return float(np.max(np.abs((x[:,i]*s)[:,i]*s-x)))
