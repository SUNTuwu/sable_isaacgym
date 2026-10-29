"""Separate mirrored XY placement costs at touchdown and last-contact liftoff."""
import math
import torch

METADATA={'scope':'Four forward/backward teachers only; front pair in chest frame, hind pair in pelvis frame.',
 'anchors':'Chest: midpoint of actual L/R shoulder-pitch hinges, heading of ROOT. Pelvis: midpoint of actual L/R hip-pitch hinges, heading of spine_roll. XY are horizontal projections.',
 'events':'Actual foot-contact point at landing; last-contact point and torso frame latched before departure at liftoff. Observed contact transitions with prior real stance; includes20ms contact/liftoff. No60ms flight gate; startup-air landing excluded.',
 'comparison':'X: x_left-x_right. Y: y_left+y_right, so ordinary left/right mirrored stance has zero error. Last matching events can be asynchronous.',
 'formula':'Separate landing/liftoff X/Y costs: -mean_pairs(expm1(min((error/.010m)^2,log(5)))*fresh_pair); continuously evaluate latest measured events, integrate dt once. Each term bounded4/s, total16/s; contact chatter cannot multiply the penalty.',
 'freshness':'Both sides initialized and each event within2reference periods; missing/stale side never zero-filled.',
 'scale_m':.01,'maximum_raw_cost_per_axis':4.}


def relative_xy(points,anchors,quaternions):
    x,y,z,w=quaternions.unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    cs=yaw.cos().repeat_interleave(2,1);sn=yaw.sin().repeat_interleave(2,1)
    delta=points[:,:,:2]-anchors[:,:,:2].repeat_interleave(2,1)
    return torch.stack((delta[:,:,0]*cs+delta[:,:,1]*sn,-delta[:,:,0]*sn+delta[:,:,1]*cs),-1)


class FootEventPairSymmetry:
    def __init__(self,n,device,dt):
        self.dt=dt
        self.stance=torch.zeros(n,4,2,device=device);self.stance_age=torch.zeros(n,4,device=device)
        self.stance_valid=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.record=torch.zeros(n,2,4,2,device=device);self.record_age=torch.zeros(n,2,4,device=device)
        self.valid=torch.zeros(n,2,4,device=device,dtype=torch.bool)
    def reset(self,ids):
        for v in (self.stance,self.stance_age,self.stance_valid,self.record,self.record_age,self.valid):v[ids]=0
    def update(self,xy,contact,takeoff,landing,age,period,enabled):
        takeoff=takeoff&self.stance_valid&enabled[:,None];landing=landing&enabled[:,None]
        events=torch.stack((landing,takeoff),1)
        values=torch.stack((xy,self.stance),1)
        timestamps=torch.stack((age[:,None].expand(-1,4),self.stance_age),1)
        self.record=torch.where(events[:,:,:,None],values,self.record)
        self.record_age=torch.where(events,timestamps,self.record_age);self.valid|=events
        fresh=(age[:,None,None]-self.record_age<=2*period[:,None,None]+1e-6)&self.valid
        paired=fresh.reshape(-1,2,2,2).all(-1)
        multiplicity=events.reshape(-1,2,2,2).sum(-1)*paired
        r=self.record.reshape(-1,2,2,2,2)
        delta=torch.stack((r[:,:,:,0,0]-r[:,:,:,1,0],r[:,:,:,0,1]+r[:,:,:,1,1]),-1)
        raw=-torch.expm1((delta/.01).square().clamp(max=math.log(5)))
        costs={}
        for typ,name in enumerate(('touchdown','liftoff')):
            for axis,label in enumerate(('x','y')):
                costs['paired_'+name+'_'+label+'_exp_cost']=(raw[:,typ,:,axis]*paired[:,typ]).mean(1)
        measured=contact&enabled[:,None]
        self.stance=torch.where(measured[:,:,None],xy,self.stance)
        self.stance_age=torch.where(measured,age[:,None],self.stance_age);self.stance_valid|=measured
        return costs,delta.clone(),multiplicity.clone()
