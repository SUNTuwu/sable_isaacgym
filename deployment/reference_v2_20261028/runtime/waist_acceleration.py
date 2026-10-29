"""World-Z acceleration of the waist pitch anchor, sampled at control rate."""
import torch
class WaistHeightAcceleration:
    def __init__(self,n,device,dt):
        self.dt=dt
        self.previous=torch.zeros(n,device=device);self.previous2=torch.zeros_like(self.previous)
        self.samples=torch.zeros(n,device=device,dtype=torch.long)
        self.acceleration=torch.zeros_like(self.previous)
    def reset(self,ids):
        for value in vars(self).values():
            if torch.is_tensor(value):value[ids]=0
    def update(self,height):
        self.acceleration=torch.where(self.samples>=2,(height-2*self.previous+self.previous2)/(self.dt*self.dt),torch.zeros_like(height))
        self.previous2.copy_(self.previous);self.previous.copy_(height);self.samples+=1
        return self.acceleration
    def cost(self,normalization):return (self.acceleration/normalization).square()
