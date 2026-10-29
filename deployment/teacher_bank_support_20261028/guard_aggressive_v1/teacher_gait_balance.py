"""Completed-cycle bilateral integrals and per-flight turning objective."""
import math
import torch

METADATA={
 'straight_teacher_ids':['01_FORWARD_POLICY','06_BACKWARD','09_BACKWARD_WALK','10_FORWARD_TROT'],
 'pairs':[['FL','FR'],['HL','HR']],
 'window':'One full reference period after1s warmup; reset discards partial windows. Pair costs settle once per completed window; no instantaneous left/right phase matching.',
 'airtime':{'scale_s':.1,'weight':1.,'maximum_cost_per_cycle':4.,
   'formula':'-mean_pairs(expm1(min(((integral_air_left-integral_air_right)/.100s)^2,log(5))))',
   'measurement':'Actual noncontact and capsule-bottom clearance>=2mm; integrate indicator*dt over the full cycle.'},
 'placement':{'scale_m':.010,'weight':1.,'maximum_cost_per_cycle':4.,
   'formula':'-mean_pairs(expm1(min(((integral_width_left-integral_width_right)/(.010m*period))^2,log(5))))',
   'measurement':'Absolute lateral ground distance from each actual contact point to heading/body midline. Refresh during contact, hold latest measured placement during flight. Both pair members must have actual contact data for the entire scored cycle; no missing-foot zero fill.'},
 'turn':{'teacher_ids':['04_TURN_LEFT','05_TURN_RIGHT'],'feet':['FL','FR','HL','HR'],
   'inverse_duration_coefficient_s':.2,'distance_reward_start_m':.2,'distance_reward_cap_m':.4,'distance_reward_max':3.,
   'short_stride_max_cost':20.,'short_stride_formula':'-20*expm1(3*clamp((.200m-signed_stride)/.200m,0,1))/expm1(3), on every real-air landing',
   'frequency':{'window_s':2.,'free_per_foot_hz':.5,'weight_per_second':4.,'maximum_cost_per_second':16.,
      'formula':'f=confirmed_real_liftoffs_per_foot_in_last2s/2s; cost=-4*mean(clamp(relu((f-.5Hz)/.5Hz)^2,max4))',
      'scope':'Count each observed-stance departure once when>=2mm clearance for60ms is confirmed; excludes startup air and contact chatter. No1s grace period for turn events.'},
   'formula':'One completed true flight: cost=-.200s/max(actual_air_time,dt); reward=3*expm1(3*clamp((signed_stride-.200m)/.200m,0,1))/expm1(3)',
   'events':'Observed contact departure, >=2mm clearance continuously>=60ms. Charge once at landing when full duration is known; unfinished true flights are settled on termination. Startup air/contact chatter excluded.',
   'positive_gate':'Genuine v44 landing, correct signed turn response, no fall, signed tangent stride>200mm. No old200ms hard positive gate.',
   'replacement':'Replaces all generic turn foot-height/stride rewards and low-height/short-stride costs, old100/300mm front rule, and old fixed below200ms swing penalty. Straight and pace generic step rules remain.',
   'event_normalization':'Per-foot score * period/(4*dt), integrated once by dt.'}}


def turn_scores(duration,distance,event,valid,dt):
    cost=-.2/duration.clamp(min=dt)*event
    reward=3*torch.expm1(3*((distance-.2)/.2).clamp(0,1))/math.expm1(3)*valid*(distance>.2+1e-7)
    return reward,cost


def turn_short_stride_cost(distance,event):
    return -20*torch.expm1(3*((.2-distance)/.2).clamp(0,1))/math.expm1(3)*event


class TurnStepFrequency:
    def __init__(self,n,device,dt):
        self.steps=int(round(2./dt));self.window=self.steps*dt;self.index=0
        self.history=torch.zeros(self.steps,n,4,device=device,dtype=torch.bool)
        self.count=torch.zeros(n,4,device=device)
    def reset(self,ids):
        self.history[:,ids]=False;self.count[ids]=0
    def update(self,liftoff,enabled):
        event=liftoff&enabled[:,None]
        self.count+=event.float()-self.history[self.index].float()
        self.history[self.index]=event;self.index=(self.index+1)%self.steps
        hz=self.count/self.window
        cost=-4*((hz-.5).clamp(min=0)/.5).square().clamp(max=4).mean(1)*enabled
        return hz,cost


class BilateralCycleIntegrals:
    def __init__(self,n,device,dt):
        self.dt=dt
        self.elapsed=torch.zeros(n,device=device)
        self.air=torch.zeros(n,4,device=device)
        self.width=torch.zeros_like(self.air)
        self.last_width=torch.zeros_like(self.air)
        self.width_valid=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.coverage=torch.zeros(n,2,device=device)

    def reset(self,ids):
        for v in (self.elapsed,self.air,self.width,self.last_width,self.width_valid,self.coverage):v[ids]=0

    def update(self,contact,clearance,outward_distance,age,period,enabled):
        active=enabled&(age>=1.-1e-7)
        measured=contact&active[:,None]
        self.last_width=torch.where(measured,outward_distance.abs(),self.last_width)
        self.width_valid|=measured
        pair_valid=self.width_valid.reshape(-1,2,2).all(-1)&active[:,None]
        air=(~contact)&(clearance>=.002-1e-7)&active[:,None]
        self.elapsed+=active*self.dt
        self.air+=air*self.dt
        self.width+=self.last_width*pair_valid.repeat_interleave(2,1)*self.dt
        self.coverage+=pair_valid*self.dt
        complete=active&(self.elapsed>=period-1e-6)
        air_delta=self.air[:,[0,2]]-self.air[:,[1,3]]
        width_delta=(self.width[:,[0,2]]-self.width[:,[1,3]])/period[:,None]
        scored_width=(self.coverage>=period[:,None]-1e-6)&complete[:,None]
        air_cost=-torch.expm1((air_delta/.1).square().clamp(max=math.log(5))).mean(1)*complete
        width_cost=-(torch.expm1((width_delta/.01).square().clamp(max=math.log(5)))*scored_width).mean(1)
        result=dict(complete=complete.clone(),air_delta_s=air_delta.clone(),width_delta_m=width_delta.clone(),
                    width_scored=scored_width.clone(),air_cost=air_cost,width_cost=width_cost)
        # The present references have1s periods at20ms control ticks.
        for v in (self.elapsed,self.air,self.width,self.coverage):v[complete]=0
        return result
