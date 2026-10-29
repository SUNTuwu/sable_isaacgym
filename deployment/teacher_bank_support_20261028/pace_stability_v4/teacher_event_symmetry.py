"""Separate mirrored XY placement costs at touchdown and last-contact liftoff."""
import math
import torch

METADATA={'scope':'Four forward/backward teachers only; front pair in chest frame, hind pair in pelvis frame.',
 'anchors':'Chest: midpoint of actual L/R shoulder-pitch hinges, heading of ROOT. Pelvis: midpoint of actual L/R hip-pitch hinges, heading of spine_roll. XY are horizontal projections.',
 'events':'Actual foot-contact point at landing; last-contact point and torso frame latched before departure at liftoff. Observed contact transitions with prior real stance; includes20ms contact/liftoff. No60ms flight gate; startup-air landing excluded.',
 'comparison':'X: x_left-x_right. Y: y_left+y_right, so ordinary left/right mirrored stance has zero error. Last matching events can be asynchronous.',
 'formula':'Epos=(mirrored XY error/.010m)^2. Per axis/type: reward=mean(exp(-Epos)*fresh), cost=-6*mean(Epos/(1+Epos)*fresh). Position costs increased from4 to6 per axis/type, total24/s, with no hard flat region.',
 'period_formula':'Same-foot same-event interval T measured from two actual events; compare Tleft-Tright, Etime=((Tleft-Tright)/(.10*reference_period))^2. Reward=2*mean(exp(-Etime)*fresh_period); cost=-3*mean(Etime/(1+Etime)*fresh_period). Left/right phase offsets are unrestricted.',
 'replaces':'Old airtime-integral and width-integral symmetry costs are replaced by signed position-square and measured-period-square scores; absolute hind30mm placement cost remains.',
 'freshness':'Both sides initialized and each event within2reference periods; missing/stale side never zero-filled.',
 'scale_m':.01,'maximum_raw_cost_per_axis':6.,'position_reward_per_axis':1.,'period_scale_fraction':.10}


def signed_position_scores(error):
    squared=(error/.01).square()
    return torch.exp(-squared),-6*(1-1/(1+squared))


def signed_period_scores(error,reference_period):
    squared=(error/(.10*reference_period)).square()
    return 2*torch.exp(-squared),-3*(1-1/(1+squared))


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
        self.last_period=torch.zeros(n,2,4,device=device)
        self.period_valid=torch.zeros_like(self.valid)
        self.period_delta=torch.zeros(n,2,2,device=device);self.period_weight=torch.zeros_like(self.period_delta)
        self.pair_periods=torch.zeros(n,2,2,2,device=device)
    def reset(self,ids):
        for v in (self.stance,self.stance_age,self.stance_valid,self.record,self.record_age,self.valid,
                  self.last_period,self.period_valid,self.period_delta,self.period_weight,self.pair_periods):v[ids]=0
    def update(self,xy,contact,takeoff,landing,age,period,enabled):
        takeoff=takeoff&self.stance_valid&enabled[:,None];landing=landing&enabled[:,None]
        events=torch.stack((landing,takeoff),1)
        values=torch.stack((xy,self.stance),1)
        timestamps=torch.stack((age[:,None].expand(-1,4),self.stance_age),1)
        interval=timestamps-self.record_age
        measured_period=events&self.valid&(interval>self.dt*.5)
        self.last_period=torch.where(measured_period,interval,self.last_period)
        self.period_valid|=measured_period
        self.record=torch.where(events[:,:,:,None],values,self.record)
        self.record_age=torch.where(events,timestamps,self.record_age);self.valid|=events
        fresh=(age[:,None,None]-self.record_age<=2*period[:,None,None]+1e-6)&self.valid
        paired=fresh.reshape(-1,2,2,2).all(-1)
        multiplicity=events.reshape(-1,2,2,2).sum(-1)*paired
        r=self.record.reshape(-1,2,2,2,2)
        delta=torch.stack((r[:,:,:,0,0]-r[:,:,:,1,0],r[:,:,:,0,1]+r[:,:,:,1,1]),-1)
        position_reward,position_cost=signed_position_scores(delta)
        costs={}
        for typ,name in enumerate(('touchdown','liftoff')):
            for axis,label in enumerate(('x','y')):
                costs['paired_'+name+'_'+label+'_squared_cost']=(position_cost[:,typ,:,axis]*paired[:,typ]).mean(1)
                costs['paired_'+name+'_'+label+'_symmetry_reward']=(position_reward[:,typ,:,axis]*paired[:,typ]).mean(1)
        periods=self.last_period.reshape(-1,2,2,2)
        period_ready=self.period_valid.reshape(-1,2,2,2).all(-1)&paired
        self.period_delta=periods[:,:,:,0]-periods[:,:,:,1]
        self.period_weight=multiplicity*period_ready
        self.pair_periods=periods.clone()
        time_reward,time_cost=signed_period_scores(self.period_delta,period[:,None,None])
        costs['paired_period_squared_cost']=(time_cost*period_ready).mean((1,2))
        costs['paired_period_symmetry_reward']=(time_reward*period_ready).mean((1,2))
        active_side=(fresh&self.period_valid).reshape(-1,2,2,2).any(-1)
        missing_cycle=active_side&~period_ready&(age[:,None,None]>2*period[:,None,None])
        costs['paired_period_missing_cost']=-missing_cycle.float().mean((1,2))
        measured=contact&enabled[:,None]
        self.stance=torch.where(measured[:,:,None],xy,self.stance)
        self.stance_age=torch.where(measured,age[:,None],self.stance_age);self.stance_valid|=measured
        return costs,delta.clone(),multiplicity.clone()
