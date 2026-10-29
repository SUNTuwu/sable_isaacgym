"""Per-teacher, evaluation-gated turning commands; reference timing is unchanged."""
import math

class TurnCurriculum:
    stages=(10,20,30,45,60,90)
    def __init__(self,ids,periods):
        self.ids=ids;self.periods=periods
        self.states=[dict(stage=0,streak=0,pending=False) if 'TURN' in ident else None for ident in ids]
    def state(self,i):
        s=self.states[i]
        return None if s is None else dict(s,degree_per_cycle=self.stages[s['stage']],period_s=self.periods[i],stages=list(self.stages))
    def load(self,i,s):
        if self.states[i] is not None and s is not None:
            self.states[i]={k:s[k] for k in ('stage','streak','pending')}
            assert 0<=self.states[i]['stage']<len(self.stages)
    def command(self,i):
        return math.radians(self.stages[self.states[i]['stage']])/self.periods[i]*(1 if 'LEFT' in self.ids[i] else -1)
    def observe(self,i,r):
        s=self.states[i]
        if s is None:return
        target=self.command(i)
        passed=(r['survival_fraction']==1. and r['mean_actual'][2]*target>0
                and r['yaw_rmse_rad_s']<=max(.02,.3*abs(target)) and r['xy_rmse_m_s']<=.03
                and max(r.get('foot_rmse_m_per_foot',[1.]))<=.03)
        s['streak']=s['streak']+1 if passed else 0
        s['pending']=s['streak']>=3 and s['stage']<len(self.stages)-1
    def apply(self,e):
        changed=[]
        for i,s in enumerate(self.states):
            if s is not None and s['pending']:
                s.update(stage=s['stage']+1,streak=0,pending=False);changed.append(i)
        for i,s in enumerate(self.states):
            if s is not None:e.teacher_commands[i,2]=self.command(i)
        if changed:
            import torch
            for i in changed:
                ids=torch.nonzero(e.teacher_ids==i,as_tuple=False).flatten();e.reset_idx(ids)
        return changed
