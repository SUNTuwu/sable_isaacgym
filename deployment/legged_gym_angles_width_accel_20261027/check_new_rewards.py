from isaacgym import gymapi
import torch,math,json,hashlib,types
from pathlib import Path
from ankle_phase import PhaseAnkleMotion
from sable_config import SableCfg
from sable_env import SableRobot
D=Path(__file__).parent

def cycle(mode='correct',dt=.02):
 m=PhaseAnkleMotion(1,'cpu',dt,period=1.,landing_deg=30.,takeoff_deg=110.);total=reverse=events=0.
 def step(contact,angle,age=2.):
  nonlocal total,reverse,events
  q=150-angle if mode not in ['wrong','body_only'] else angle if mode=='wrong' else 50
  a=torch.full((1,2),angle);c=torch.full((1,2),contact,dtype=torch.bool)
  m.update(torch.full((1,2),q*math.pi/180),c,torch.full((1,2),0. if contact or mode=='drag' else .023),a,torch.tensor([0. if mode=='park' else .1]),torch.tensor([-.1 if mode=='backward' else .1]),torch.tensor([age]))
  total+=m.reward.item()*dt;reverse+=m.reverse_cost.item()*dt;events+=m.event_cost.item()*dt
 for _ in range(round(.75/dt)):step(True,110.,.5)
 for k in range(round(.25/dt)):step(False,110-80*(k+1)/round(.25/dt))
 step(True,30.)
 for k in range(round(.75/dt)):step(True,30+80*(k+1)/round(.75/dt))
 step(False,110.)
 return m,(total,reverse,events)
_,good=cycle();_,wrong=cycle('wrong');_,body=cycle('body_only');_,park=cycle('park');_,back=cycle('backward');_,drag=cycle('drag');_,fine=cycle(dt=.01)
assert good[0]>.7 and good[1]<1e-7 and good[2]<1e-7,(good,wrong)
assert wrong[0]==0 and wrong[1]>0 and body[0]==park[0]==back[0]==drag[0]==0
assert abs(good[0]-fine[0])<.02,(good,fine)
m,_=cycle();m.reset(torch.tensor([0]));assert all(not x.any() for x in vars(m).values() if torch.is_tensor(x))
# The reward needs ankle travel: translating/rotating the entire leg without ankle rotation earns zero.
obj=types.SimpleNamespace(cfg=SableCfg(),rew_buf=torch.zeros(1),reward_names=['ankle_phase_angle','ankle_phase_event','ankle_phase_reverse'],reward_functions=[lambda:torch.ones(1)]*3,reward_scales={'ankle_phase_angle':-4,'ankle_phase_event':-2,'ankle_phase_reverse':-.5},episode_sums={k:torch.zeros(1) for k in ['ankle_phase_angle','ankle_phase_event','ankle_phase_reverse']})
SableRobot.compute_reward(obj);assert obj.rew_buf.item()==-6.5

from hind_landing_width import HindLandingWidth
from peak_acceleration import peak_cost
from landing_stride_rewards import landing_shortfall
from legged_gym.utils.helpers import class_to_dict
widths=torch.tensor([0.,.03,.059,.06,.08,-.06]);n=len(widths);w=HindLandingWidth(n,'cpu')
xy=torch.zeros(n,2,2);xy[:,0,0]=.1;xy[:,1,0]=-.1;xy[:,0,1]=widths/2;xy[:,1,1]=-widths/2
contact=torch.ones(n,2,dtype=torch.bool);forward=torch.tensor([[1.,0.]]).repeat(n,1)
w.update(xy,contact,contact,forward,torch.ones(n)*.1,torch.ones(n)*2)
expected=((.06-widths).clamp(min=0)/.06).square();assert torch.allclose(w.width,widths,atol=1e-6) and torch.allclose(w.cost,expected,atol=1e-6)
rotation=torch.tensor([[0.,-1.],[1.,0.]])
w.reset(torch.arange(n));w.update(xy@rotation.T,contact,contact,forward@rotation.T,torch.ones(n)*.1,torch.ones(n)*2);assert torch.allclose(w.cost,expected,atol=1e-6)
w.reset(torch.arange(n));assert not w.valid.any() and not w.cost.any()
ac=torch.tensor([[0.,100.,150.],[150.,150.,150.],[300.,-300.,300.]])
assert torch.allclose(peak_cost(ac,150.),torch.tensor([0.,0.,1.]))
assert landing_shortfall(torch.tensor([30.]),minimum=30.).item()==0 and landing_shortfall(torch.tensor([29.]),minimum=30.).item()>0
names=['hind_landing_width','proximal_acceleration_peak'];obj=types.SimpleNamespace(cfg=SableCfg(),rew_buf=torch.zeros(1),reward_names=names,reward_functions=[lambda:torch.ones(1)]*2,reward_scales={k:-4. for k in names},episode_sums={k:torch.zeros(1) for k in names});SableRobot.compute_reward(obj);assert obj.rew_buf.item()==-8.
sc=class_to_dict(SableCfg.rewards.scales);assert all(v!=0 for v in sc.values())
removed=['ankle_stance_motion','ankle_stance_reversal','hind_foot_angle','foot_spacing','stand_still','dof_vel','feet_stumble'];assert not any(k in sc for k in removed)
assert SableCfg.ankle_phase.landing_angle_deg==SableCfg.ankle_phase.landing_min_deg==30 and SableCfg.ankle_phase.takeoff_angle_deg==110
report={'passed':True,'correct_30_to110_cycle_reward_reversal_event_cost':good,'wrong_direction':wrong,'body_only_reward':body[0],'park_reward':park[0],'drag_reward':drag[0],'backward_reward':back[0],'width60_boundary_yaw_and_foreaft_invariance':True,'acceleration150_threshold':True,'negative_cost_survives_clipping':True,'active_reward_count':len(sc),'removed_keys':removed,'source_sha256':{name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ['ankle_phase.py','ankle_stance.py','hind_landing_width.py','peak_acceleration.py','sable_env.py','sable_config.py','evaluate.py','train.py']}}
(D/'verification/new_rewards_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
