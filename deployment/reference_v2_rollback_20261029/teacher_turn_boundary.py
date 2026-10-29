"""Turn-only 30cm -> 20cm boundary, gated by 19/20 complete eval episodes."""
import torch
import math

class TurnBoundary:
    def __init__(self,ids,device):
        self.turn=torch.tensor(['TURN' in x for x in ids],device=device)
        self.limit=torch.full((len(ids),),.30,device=device)
        self.heading_limit=torch.full((len(ids),),30.,device=device)
        self.windows=[[] for _ in ids];self.pending=set()
    def evaluate(self,e):
        radius=(e.root_states[:,:2]-e.translation_guard.start_xy).norm(dim=1)
        from teacher_translation_guard import root_heading
        delta=root_heading(e.root_states[:,3:7])-e.translation_guard.start_yaw
        angle=torch.atan2(delta.sin(),delta.cos()).abs()
        turn=self.turn[e.teacher_ids]
        return e.translation_guard.ready&torch.where(turn,radius>self.limit[e.teacher_ids]+1e-7,angle>self.heading_limit[e.teacher_ids]*math.pi/180+1e-7)
    def state(self,i):
        if not bool(self.turn[i]):return dict(limit_deg=float(self.heading_limit[i]),recent_complete_episodes=self.windows[i],pending=i in self.pending,threshold=.95,window_size=20,final_limit_deg=10.)
        return dict(limit_m=float(self.limit[i]),recent_complete_episodes=self.windows[i],pending=i in self.pending,threshold=.95,window_size=20,final_limit_m=.20)
    def load(self,i,s):
        if s is None:return
        if bool(self.turn[i]):
            assert abs(s['limit_m']-.30)<1e-6 or abs(s['limit_m']-.20)<1e-6
            self.limit[i]=s['limit_m']
        else:
            assert s['limit_deg'] in (30.,20.,10.)
            self.heading_limit[i]=s['limit_deg']
        self.windows[i]=list(s['recent_complete_episodes'])[-20:]
        if s.get('pending'):self.pending.add(i)
    def observe(self,i,alive):
        self.windows[i]=(self.windows[i]+[bool(x) for x in alive])[-20:]
        if len(self.windows[i])==20 and sum(self.windows[i])>=19 and (float(self.limit[i])>.25 if bool(self.turn[i]) else float(self.heading_limit[i])>10.1):self.pending.add(i)
    def apply(self,e):
        changed=sorted(self.pending);self.pending.clear()
        for i in changed:
            if bool(self.turn[i]):self.limit[i]=.20
            else:self.heading_limit[i]=20. if float(self.heading_limit[i])>20.1 else 10.
            self.windows[i]=[]
            e.reset_idx(torch.nonzero(e.teacher_ids==i,as_tuple=False).flatten())
        return changed

def cpu_check():
    from types import SimpleNamespace as N
    c=TurnBoundary(['04_TURN_LEFT','05_TURN_RIGHT','01_FORWARD_POLICY'],'cpu')
    root=torch.zeros(3,13);root[:,0]=.301
    e=N(root_states=root,teacher_ids=torch.arange(3),translation_guard=N(start_xy=torch.zeros(3,2),start_yaw=torch.zeros(3),ready=torch.ones(3,dtype=torch.bool)),reset_idx=lambda _:None)
    assert c.evaluate(e).tolist()==[True,True,False]
    c.observe(0,[True]*18+[False]);assert not c.pending
    c.observe(0,[True]);assert c.pending=={0}
    c.observe(1,[True]*18+[False]*2);assert c.pending=={0}
    copy=TurnBoundary(['04_TURN_LEFT','05_TURN_RIGHT','01_FORWARD_POLICY'],'cpu');copy.load(0,c.state(0));assert copy.pending=={0}
    assert c.apply(e)==[0];root[:,0]=.25
    assert c.evaluate(e).tolist()==[True,False,False]
    for _ in range(10):c.observe(0,[True]*4)
    assert not c.pending and abs(float(c.limit[0])-.2)<1e-6
    root[2,5]=math.sin(math.radians(31)/2);root[2,6]=math.cos(math.radians(31)/2)
    assert c.evaluate(e)[2]
    c.observe(2,[True]*20);c.apply(e);assert float(c.heading_limit[2])==20.
    assert not c.windows[2]
    c.observe(2,[True]*19);assert 2 not in c.pending
    c.observe(2,[True]);c.apply(e);assert float(c.heading_limit[2])==10.
    c.observe(2,[True]*20);assert 2 not in c.pending
    print('turn/translation boundary: threshold, 95% episode window, persistence, turn-only scope and final 20cm checks passed')

if __name__=='__main__':cpu_check()
