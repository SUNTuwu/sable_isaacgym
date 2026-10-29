"""Body-lateral separation of the last real hind-foot landing points."""
import torch

class HindLandingWidth:
    def __init__(self,n,device,minimum=.06):
        self.minimum=minimum
        self.points=torch.zeros(n,2,2,device=device)
        self.known=torch.zeros(n,2,device=device,dtype=torch.bool)
        self.width=torch.zeros(n,device=device)
        self.valid=torch.zeros(n,device=device,dtype=torch.bool)
        self.cost=torch.zeros(n,device=device)

    def reset(self,ids):
        for value in [self.points,self.known,self.width,self.valid,self.cost]:value[ids]=0

    def update(self,xy,contact,landed,forward,command,age):
        fresh=landed|(~self.known&contact)
        self.points=torch.where(fresh[:,:,None],xy,self.points)
        self.known|=fresh
        left=torch.stack([-forward[:,1],forward[:,0]],1)
        left=left/left.norm(dim=1,keepdim=True).clamp(min=1e-6)
        self.width=((self.points[:,0]-self.points[:,1])*left).sum(1)
        self.valid=self.known.all(1)&(age>=1.)
        # Keep signed left-minus-right width so crossing cannot satisfy the target.
        self.cost=((self.minimum-self.width).clamp(min=0.)/self.minimum).square().clamp(max=9.)*self.valid*(command>.02)
