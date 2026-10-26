"""Actual stance excursion diagnostics only; no legacy ankle reward."""
import torch
class StanceAnkleMetrics:
    def __init__(self,n,device,dt):
        self.dt=dt
        self.lo=torch.zeros(n,2,device=device);self.hi=torch.zeros_like(self.lo)
        self.age=torch.zeros_like(self.lo);self.air=torch.zeros_like(self.lo)
        self.active=torch.zeros(n,2,device=device,dtype=torch.bool)
        self.armed=torch.zeros_like(self.active);self.previous=torch.zeros_like(self.active);self.landed=torch.zeros_like(self.active)
    def reset(self,ids):
        for value in [self.lo,self.hi,self.age,self.air,self.active,self.armed,self.previous,self.landed]:value[ids]=0
    def update(self,q,contact,clearance,*unused):
        self.air=torch.where((~contact)&(clearance>=.002),self.air+self.dt,torch.zeros_like(self.air))
        self.armed|=self.air>=.06-1e-7
        self.landed=contact&~self.previous&self.armed
        self.active=(self.active&contact)|self.landed;self.armed&=~self.landed
        self.lo=torch.where(self.landed,q,self.lo);self.hi=torch.where(self.landed,q,self.hi)
        self.age=torch.where(self.active,torch.where(self.landed,torch.zeros_like(self.age),self.age)+self.dt,torch.zeros_like(self.age))
        self.lo=torch.where(self.active,torch.minimum(self.lo,q),self.lo)
        self.hi=torch.where(self.active,torch.maximum(self.hi,q),self.hi)
        self.previous.copy_(contact)
