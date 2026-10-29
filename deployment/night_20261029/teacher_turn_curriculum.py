"""Explicit fixed +/-30 deg/s, superseding prior per-cycle curriculum."""
import math
class TurnCurriculum:
    stages=(30,)
    def __init__(self,ids,periods):self.ids=ids;self.periods=periods
    def state(self,i):
        return None if 'TURN' not in self.ids[i] else dict(stage=0,streak=0,pending=False,mode='fixed_rate',target_deg_s=30.,degree_per_cycle=30.*self.periods[i],period_s=self.periods[i],stages=[30.*self.periods[i]])
    def load(self,i,s):pass
    def command(self,i):return math.radians(30)*(1 if 'LEFT' in self.ids[i] else -1)
    def observe(self,i,r):pass
    def apply(self,e):
        for i,ident in enumerate(self.ids):
            if 'TURN' in ident:e.teacher_commands[i,:2]=0.;e.teacher_commands[i,2]=self.command(i)
        return []
