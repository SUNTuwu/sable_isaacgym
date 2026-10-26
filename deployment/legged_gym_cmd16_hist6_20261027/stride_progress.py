"""Capped stride reward at real air/landing events;20mm lift remains an independent acceptance metric."""
import torch,math
from landing_stride_rewards import signed_hind_stride
class StrideProgress:
    def __init__(self,n,device,period,dt,cap_m=.10,exponent=4.,hind_zero_m=.08,front_zero_m=.08,variance_normalization_m=.08,variance_max_age_periods=2.):
        self.period=period;self.dt=dt;self.cap_m=cap_m;self.exponent=exponent;self.hind_zero_m=hind_zero_m
        self.zero_m=torch.tensor([front_zero_m,front_zero_m,hind_zero_m,hind_zero_m],device=device)
        self.variance_normalization_m=variance_normalization_m;self.variance_max_age_s=variance_max_age_periods*period
        self.prior=torch.ones(n,4,device=device,dtype=torch.bool)
        self.armed=torch.zeros_like(self.prior);self.start=torch.zeros(n,4,2,device=device);self.axis=torch.zeros_like(self.start)
        self.takeoff_age=torch.zeros(n,4,device=device);self.target=torch.zeros(n,4,device=device);self.clear=torch.zeros_like(self.target);self.maxclear=torch.zeros_like(self.target);self.peak=torch.zeros_like(self.target)
        self.qualified=torch.zeros_like(self.prior);self.last_signed=torch.zeros_like(self.target);self.reward=torch.zeros(n,device=device)
        self.hind_landing=torch.zeros(n,2,device=device,dtype=torch.bool);self.shortfall=torch.zeros(n,device=device)
        self.landing=torch.zeros_like(self.prior)
        self.completed_stride=torch.zeros_like(self.target);self.completed_age=torch.zeros_like(self.target);self.completed_valid=torch.zeros_like(self.prior)
        self.variance_m2=torch.zeros(n,device=device);self.variance_event=torch.zeros(n,device=device,dtype=torch.bool);self.variance_cost=torch.zeros(n,device=device)
    def reset(self,ids):
        self.prior[ids]=True;self.armed[ids]=False
        for v in [self.start,self.axis,self.takeoff_age,self.target,self.clear,self.maxclear,self.peak,self.last_signed,self.reward]:v[ids]=0
        for v in [self.landing,self.completed_stride,self.completed_age,self.completed_valid,self.variance_m2,self.variance_event,self.variance_cost]:v[ids]=0
        self.qualified[ids]=False;self.hind_landing[ids]=False;self.shortfall[ids]=0
    def update(self,xy,contact,clearance,forward,command,age,body_forward_velocity):
        takeoff=self.prior&~contact;landing=~self.prior&contact
        self.takeoff_age[takeoff]=age[:,None].expand_as(self.takeoff_age)[takeoff]
        self.start[takeoff]=xy[takeoff];self.axis[takeoff]=forward[:,None,:].expand_as(xy)[takeoff]
        target=torch.full_like(self.target,self.cap_m)
        self.target[takeoff]=target[takeoff];self.armed[takeoff]=((age>=1.)&(command>.02))[:,None].expand_as(self.armed)[takeoff]
        self.clear[takeoff]=0;self.maxclear[takeoff]=0;self.peak[takeoff]=0
        self.clear=torch.where(~contact&(clearance>=.002),self.clear+self.dt,torch.zeros_like(self.clear))
        self.maxclear=torch.maximum(self.maxclear,self.clear)
        self.peak=torch.where(~contact,torch.maximum(self.peak,clearance),self.peak)
        signed=((xy-self.start)*self.axis).sum(-1);self.last_signed=signed
        self.qualified=landing&self.armed&(self.maxclear>=.060-1e-7)&(signed>=.005)&(body_forward_velocity[:,None]>.005)
        # Same signed exponential for all four feet. Real-air short/backward
        # landings stay eligible for cost, independent of forward-motion gate.
        self.landing=landing&self.armed&(self.maxclear>=.060-1e-7)
        self.hind_landing=self.landing[:,2:]
        score=signed_hind_stride(signed,self.zero_m,self.cap_m,self.exponent)
        self.reward=(score.clamp(min=0.)*self.qualified).sum(1)*self.period/(4*self.dt)
        self.shortfall=((-score).clamp(min=0.)*self.landing).sum(1)*self.period/(4*self.dt)
        # Compare the most recent real completed step of EACH foot, not swing
        # positions or zero-filled missing feet. Actual measurements are uncapped.
        self.completed_stride=torch.where(self.landing,signed,self.completed_stride)
        self.completed_age=torch.where(self.landing,age[:,None],self.completed_age)
        self.completed_valid|=self.landing
        fresh=(self.completed_valid&((age[:,None]-self.completed_age)<=self.variance_max_age_s+1e-7)).all(1)
        mean=self.completed_stride.mean(1,keepdim=True)
        self.variance_m2=(self.completed_stride-mean).square().mean(1)
        self.variance_event=fresh&self.landing.any(1)&(command>.02)
        self.variance_cost=(self.variance_m2/self.variance_normalization_m**2)*self.variance_event*self.landing.sum(1)*self.period/(4*self.dt)
        self.armed[landing]=False;self.prior=contact.clone()
        return self.reward
