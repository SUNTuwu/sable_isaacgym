"""Capped forward stride reward at valid landings, with real clearance gating."""
import torch
class StrideProgress:
    def __init__(self,n,device,period,dt,cap_m=.05):
        self.period=period;self.dt=dt;self.cap_m=cap_m
        self.prior=torch.ones(n,4,device=device,dtype=torch.bool)
        self.armed=torch.zeros_like(self.prior);self.start=torch.zeros(n,4,2,device=device);self.axis=torch.zeros_like(self.start)
        self.takeoff_age=torch.zeros(n,4,device=device);self.target=torch.zeros(n,4,device=device);self.clear=torch.zeros_like(self.target);self.maxclear=torch.zeros_like(self.target);self.peak=torch.zeros_like(self.target)
        self.qualified=torch.zeros_like(self.prior);self.last_signed=torch.zeros_like(self.target);self.reward=torch.zeros(n,device=device)
    def reset(self,ids):
        self.prior[ids]=True;self.armed[ids]=False
        for v in [self.start,self.axis,self.takeoff_age,self.target,self.clear,self.maxclear,self.peak,self.last_signed,self.reward]:v[ids]=0
        self.qualified[ids]=False
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
        self.qualified=landing&self.armed&(self.peak>.020)&(self.maxclear>=.060-1e-7)&(signed>=.005)&(body_forward_velocity[:,None]>.005)
        fraction=(signed/self.target.clamp(min=.005)).clamp(0,1)
        self.reward=(fraction*self.qualified).sum(1)*self.period/(4*self.dt)
        self.armed[landing]=False;self.prior=contact.clone()
        return self.reward
