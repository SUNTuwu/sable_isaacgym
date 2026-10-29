"""Episode signed velocity integrals; fixed user boundaries, no tightening."""
import math
import torch

class TurnBoundary:
    def __init__(self,ids,device):
        self.turn=torch.tensor(['TURN' in x for x in ids],device=device)
        self.limit=torch.full((len(ids),),.20,device=device)
        self.heading_limit=torch.full((len(ids),),30.,device=device)
        self.windows=[[] for _ in ids];self.pending=set();self.angle=None
    def reset(self,ids):
        if self.angle is not None:
            self.angle[ids]=0;self.displacement[ids]=0;self.last_step[ids]=-1
    def evaluate(self,e):
        if self.angle is None:
            self.angle=torch.zeros(e.num_envs,device=e.device)
            self.displacement=torch.zeros(e.num_envs,2,device=e.device)
            self.last_step=torch.full((e.num_envs,),-1,device=e.device,dtype=torch.long)
        # Explicit reset hook and step guard prevent duplicate integration during observations.
        fresh=e.episode_length_buf<self.last_step
        self.reset(torch.nonzero(fresh,as_tuple=False).flatten())
        active=e.episode_length_buf!=self.last_step
        self.angle+=e.heading_yaw_rate*e.dt*active
        self.displacement+=e.root_states[:,7:9]*e.dt*active[:,None]
        self.last_step.copy_(e.episode_length_buf)
        return torch.where(self.turn[e.teacher_ids],self.displacement.norm(dim=1)>.20+1e-7,self.angle.abs()>math.radians(30)+1e-7)
    def state(self,i):
        return dict(mode='signed_velocity_integral_fixed',limit_m=.20,limit_deg=30.,recent_complete_episodes=self.windows[i],pending=False,
                    definition='integral(yaw_dot dt); norm(integral(world_xy_velocity dt)); reset per episode')
    def load(self,i,s):
        # Intentional migration: old adaptive thresholds are superseded by user.
        if s:self.windows[i]=list(s.get('recent_complete_episodes',[]))[-20:]
    def observe(self,i,alive):self.windows[i]=(self.windows[i]+[bool(x) for x in alive])[-20:]
    def apply(self,e):return []

def cpu_check():
    from types import SimpleNamespace as N
    e=N(num_envs=2,device='cpu',teacher_ids=torch.arange(2),root_states=torch.zeros(2,13),heading_yaw_rate=torch.tensor([math.radians(10),0.]),dt=.1,episode_length_buf=torch.zeros(2,dtype=torch.long))
    c=TurnBoundary(['FORWARD','TURN_LEFT'],'cpu');e.root_states[1,7]=.1
    for _ in range(20):e.episode_length_buf+=1;c.evaluate(e)
    assert not c.evaluate(e).any(),'20cm boundary is strict >, duplicate calls do not integrate'
    e.episode_length_buf+=1;assert c.evaluate(e).tolist()==[False,True]
    for _ in range(10):e.episode_length_buf+=1;c.evaluate(e)
    assert c.evaluate(e).tolist()==[True,True]
    c.reset(torch.arange(2));assert not c.angle.any() and not c.displacement.any()
    e.heading_yaw_rate[0]*=-1;e.root_states[1,7]*=-1
    e.episode_length_buf+=1;assert not c.evaluate(e).any()
    assert c.angle[0]<0 and c.displacement[1,0]<0
    print('integral boundaries: threshold, sign, reset, duplicate-step checks passed')
if __name__=='__main__':cpu_check()
