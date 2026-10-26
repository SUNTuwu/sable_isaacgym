from isaacgym import gymapi
import torch,math,json,hashlib,types
from pathlib import Path
from ankle_phase import PhaseAnkleMotion
from sable_config import SableCfg
from sable_env import SableRobot
D=Path(__file__).parent

def cycle(mode='correct',dt=.02):
 m=PhaseAnkleMotion(1,'cpu',dt);total=reverse=events=0.
 def step(contact,angle,age=2.):
  nonlocal total,reverse,events
  q=120-angle if mode not in ['wrong','body_only'] else angle if mode=='wrong' else 50
  a=torch.full((1,2),angle);c=torch.full((1,2),contact,dtype=torch.bool)
  m.update(torch.full((1,2),q*math.pi/180),c,torch.full((1,2),0. if contact or mode=='drag' else .023),a,torch.tensor([0. if mode=='park' else .1]),torch.tensor([-.1 if mode=='backward' else .1]),torch.tensor([age]))
  total+=m.reward.item()*dt;reverse+=m.reverse_cost.item()*dt;events+=m.event_cost.item()*dt
 for _ in range(round(.6/dt)):step(True,90.,.5)
 for k in range(round(.2/dt)):step(False,90-45*(k+1)/round(.2/dt))
 step(True,45.)
 for k in range(round(.6/dt)):step(True,45+45*(k+1)/round(.6/dt))
 step(False,90.)
 return m,(total,reverse,events)
_,good=cycle();_,wrong=cycle('wrong');_,body=cycle('body_only');_,park=cycle('park');_,back=cycle('backward');_,drag=cycle('drag');_,fine=cycle(dt=.01)
assert good[0]>.7 and good[1]<1e-7 and good[2]<1e-7,(good,wrong)
assert wrong[0]==0 and wrong[1]>0 and body[0]==park[0]==back[0]==drag[0]==0
assert abs(good[0]-fine[0])<.02,(good,fine)
m,_=cycle();m.reset(torch.tensor([0]));assert all(not x.any() for x in vars(m).values() if torch.is_tensor(x))
# The reward needs ankle travel: translating/rotating the entire leg without ankle rotation earns zero.
obj=types.SimpleNamespace(cfg=SableCfg(),rew_buf=torch.zeros(1),reward_names=['ankle_phase_angle','ankle_phase_event','ankle_phase_reverse'],reward_functions=[lambda:torch.ones(1)]*3,reward_scales={'ankle_phase_angle':-4,'ankle_phase_event':-2,'ankle_phase_reverse':-.5},episode_sums={k:torch.zeros(1) for k in ['ankle_phase_angle','ankle_phase_event','ankle_phase_reverse']})
SableRobot.compute_reward(obj);assert obj.rew_buf.item()==-6.5
report={'passed':True,'correct_cycle_reward_reversal_event_cost':good,'wrong_direction':wrong,'body_only_reward':body[0],'park_reward':park[0],'drag_reward':drag[0],'backward_reward':back[0],'dt01':fine,'reset_cleared':True,'negative_cost_survives_clipping':True,'source_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['ankle_phase.py','sable_env.py','sable_config.py','evaluate.py']}}
(D/'verification/ankle_phase_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
