from isaacgym import gymapi
import torch,math,json,hashlib,types
from pathlib import Path
from ankle_phase import PhaseAnkleMotion
from sable_config import SableCfg
from sable_env import SableRobot
D=Path(__file__).parent

def cycle(mode='correct',dt=.02):
 m=PhaseAnkleMotion(1,'cpu',dt,period=1.,landing_deg=30.,takeoff_deg=110.);total=reverse=0.
 def step(contact,ideal,age=2.):
  nonlocal total,reverse
  angle=70. if mode in ['stalled','backward','park'] else ideal+(-20 if contact else 20) if mode=='lagging' else ideal
  q=50 if mode=='body_only' else angle if mode=='wrong' else 150-angle
  m.update(torch.full((1,2),q*math.pi/180),torch.full((1,2),contact,dtype=torch.bool),torch.full((1,2),0. if contact else .023),torch.full((1,2),angle),torch.tensor([0. if mode=='park' else .1]),torch.tensor([-.1 if mode=='backward' else 0. if mode=='stalled' else .1]),torch.tensor([age]))
  total+=m.lag_cost.item()*dt;reverse+=m.reverse_cost.item()*dt
 for _ in range(round(.75/dt)):step(True,110.,.5)
 for k in range(round(.25/dt)):step(False,110-80*(k+1)/round(.25/dt))
 step(True,30.)
 for k in range(round(.75/dt)):step(True,30+80*(k+1)/round(.75/dt))
 step(False,110.)
 return m,(total,reverse)
results={mode:cycle(mode)[1] for mode in ['correct','lagging','body_only','wrong','stalled','backward','park']}
assert results['correct'][0]<1e-7 and results['park'][0]==0,results
assert all(results[k][0]>.01 for k in ['lagging','body_only','wrong','stalled','backward']),results
assert results['wrong'][1]>0 and abs(cycle(dt=.01)[1][0]-results['correct'][0])<1e-7
m,_=cycle();m.reset(torch.tensor([0]));assert all(not x.any() for x in vars(m).values() if torch.is_tensor(x))
assert not hasattr(m,'reward') and not hasattr(SableRobot,'_reward_ankle_phase_motion')
names=['ankle_phase_lag'];obj=types.SimpleNamespace(cfg=SableCfg(),rew_buf=torch.zeros(1),reward_names=names,reward_functions=[lambda:torch.ones(1)],reward_scales={'ankle_phase_lag':-8.},episode_sums={'ankle_phase_lag':torch.zeros(1)})
SableRobot.compute_reward(obj);assert obj.rew_buf.item()==-8.
lag=torch.tensor([0.,5.,5.001,35.,200.]);penalty=-8*((lag-5).clamp(min=0)/30).square().clamp(max=16)
assert penalty[0]==penalty[1]==0 and (penalty[2:]<0).all() and penalty[3]==-8 and penalty[4]==-128
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
removed=['ankle_phase_motion','hip_stand_exp','ankle_stance_motion','ankle_stance_reversal','hind_foot_angle','foot_spacing','stand_still','dof_vel','feet_stumble'];assert not any(k in sc for k in removed)
assert SableCfg.ankle_phase.landing_angle_deg==SableCfg.ankle_phase.landing_min_deg==30 and SableCfg.ankle_phase.takeoff_angle_deg==110
from landing_stride_rewards import signed_hind_stride
points=torch.tensor([0.,.05,.079,.08,.081,.10,.12])
values=signed_hind_stride(points,zero=SableCfg.stride.hind_zero_m,cap=SableCfg.stride.reward_cap_m,exponent=SableCfg.stride.reward_exponent)
assert (values[:3]<0).all() and abs(values[3].item())<1e-7 and values[4]>0 and torch.allclose(values[5:],torch.ones(2))
assert (values[1:]-values[:-1]>=0).all() and len(sc)==29 and not hasattr(SableRobot,'_reward_hip_stand_exp')
names=['hind_stride_shortfall'];obj=types.SimpleNamespace(cfg=SableCfg(),rew_buf=torch.zeros(1),reward_names=names,reward_functions=[lambda:torch.ones(1)],reward_scales={'hind_stride_shortfall':-2.},episode_sums={'hind_stride_shortfall':torch.zeros(1)});SableRobot.compute_reward(obj);assert obj.rew_buf.item()==-2.
report={'passed':True,'tracking_lag_cycles':results,'negative_cost_survives_positive_clip':True,'lag_0_5_5point001_35_200_deg_weighted_cost':penalty.tolist(),'old_positive_motion_reward_removed':True,'hip_posture_removed':True,'signed_stride80_scores':values.tolist(),'active_reward_count':len(sc),'removed_keys':removed,'source_sha256':{name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ['ankle_phase.py','ankle_stance.py','hind_landing_width.py','peak_acceleration.py','sable_env.py','sable_config.py','evaluate.py','train.py','supervise.py','progress_report.py','landing_stride_rewards.py']}}
(D/'verification/new_rewards_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
