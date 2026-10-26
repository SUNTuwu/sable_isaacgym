"""Bounded range reward for real continuous hind-foot stance, not angular-speed farming."""
import torch,math
class StanceAnkleMotion:
    def __init__(self,n,device,dt,period=.8,target_deg=15.):
        self.dt=dt;self.factor=period/(2*dt);self.target=math.radians(target_deg)
        self.lo=torch.zeros(n,2,device=device);self.hi=torch.zeros_like(self.lo);self.last=torch.zeros_like(self.lo)
        self.paid=torch.zeros_like(self.lo);self.age=torch.zeros_like(self.lo);self.air=torch.zeros_like(self.lo)
        self.active=torch.zeros(n,2,device=device,dtype=torch.bool);self.armed=torch.zeros_like(self.active);self.previous=torch.zeros_like(self.active);self.landed=torch.zeros_like(self.active)
        self.reward=torch.zeros(n,device=device);self.reversal_cost=torch.zeros_like(self.reward)
    def reset(self,ids):
        for x in [self.lo,self.hi,self.last,self.paid,self.age,self.air,self.active,self.armed,self.previous,self.landed,self.reward,self.reversal_cost]:x[ids]=0
    def update(self,q,contact,clearance,foot_angle_deg,commands,vx,episode_age):
        self.air=torch.where((~contact)&(clearance>=.002),self.air+self.dt,torch.zeros_like(self.air))
        self.armed|=self.air>=.06-1e-7
        self.landed=contact&~self.previous&self.armed
        self.active=(self.active&contact)|self.landed
        self.armed&=~self.landed
        for x in [self.lo,self.hi,self.last]:x.copy_(torch.where(self.landed,q,x))
        self.paid=torch.where(self.landed,torch.zeros_like(self.paid),self.paid)
        self.age=torch.where(self.active,torch.where(self.landed,torch.zeros_like(self.age),self.age)+self.dt,torch.zeros_like(self.age))
        old_range=self.hi-self.lo
        self.lo=torch.where(self.active,torch.minimum(self.lo,q),self.lo);self.hi=torch.where(self.active,torch.maximum(self.hi,q),self.hi)
        excursion=self.hi-self.lo;increase=(excursion-old_range).clamp(min=0.)
        travel=torch.where(self.active,(q-self.last).abs(),torch.zeros_like(q))
        moving=(commands>.02)&(vx>.005)&(episode_age>=1.)
        mature=self.age>=.08-1e-7
        eligible=self.active&mature&moving[:,None]&(foot_angle_deg<90.)
        capped=excursion.clamp(max=self.target)
        credit=(capped-self.paid).clamp(min=0.)*eligible
        self.reward=(credit/self.target).sum(1)*self.factor
        self.reversal_cost=(((travel-increase).clamp(min=0.)/self.target)*self.active* moving[:,None]).sum(1)*self.factor
        # No delayed credit for parking or wrong-facing feet. Each distinct range is paid once.
        self.paid=torch.where(mature|~moving[:,None],capped,self.paid)
        self.previous.copy_(contact);self.last.copy_(q)
