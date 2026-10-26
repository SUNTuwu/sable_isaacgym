"""Capped stride reward at real air/landing events;20mm lift remains an independent acceptance metric."""
import torch,math
from landing_stride_rewards import signed_hind_stride
class StrideProgress:
    def __init__(self,n,device,period,dt,cap_m=.10,exponent=4.,hind_zero_m=.04):
        self.period=period;self.dt=dt;self.cap_m=cap_m;self.exponent=exponent;self.hind_zero_m=hind_zero_m
        self.prior=torch.ones(n,4,device=device,dtype=torch.bool)
        self.armed=torch.zeros_like(self.prior);self.start=torch.zeros(n,4,2,device=device);self.axis=torch.zeros_like(self.start)
        self.takeoff_age=torch.zeros(n,4,device=device);self.target=torch.zeros(n,4,device=device);self.clear=torch.zeros_like(self.target);self.maxclear=torch.zeros_like(self.target);self.peak=torch.zeros_like(self.target)
        self.qualified=torch.zeros_like(self.prior);self.last_signed=torch.zeros_like(self.target);self.reward=torch.zeros(n,device=device)
        self.hind_landing=torch.zeros(n,2,device=device,dtype=torch.bool);self.shortfall=torch.zeros(n,device=device)
    def reset(self,ids):
        self.prior[ids]=True;self.armed[ids]=False
        for v in [self.start,self.axis,self.takeoff_age,self.target,self.clear,self.maxclear,self.peak,self.last_signed,self.reward]:v[ids]=0
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
        fraction=(signed/self.target.clamp(min=.005)).clamp(0,1)
        score=torch.expm1(self.exponent*fraction)/math.expm1(self.exponent)
        # Negative hind placement must include tiny/backward/low steps, otherwise the policy can dodge it.
        # Require a real air phase and landing, but do not require >20mm peak or forward body speed for the cost.
        self.hind_landing=(landing&self.armed&(self.maxclear>=.060-1e-7))[:,2:]
        hind_score=signed_hind_stride(signed[:,2:],self.hind_zero_m,self.cap_m,self.exponent)
        score[:,2:]=hind_score.clamp(min=0.)
        self.reward=(score*self.qualified).sum(1)*self.period/(4*self.dt)
        self.shortfall=((-hind_score).clamp(min=0.)*self.hind_landing).sum(1)*self.period/(4*self.dt)
        self.armed[landing]=False;self.prior=contact.clone()
        return self.reward
