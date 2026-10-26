from isaacgym import gymapi
import torch,math,json,hashlib,types
from pathlib import Path
from stride_progress import StrideProgress
from landing_stride_rewards import signed_hind_stride
from sable_env import SableRobot
from sable_config import SableCfg
D=Path(__file__).parent

def cycle(dist,feet=None,dt=.02,height=.003,vx=.1,age=2.,cmd=.1):
 m=StrideProgress(1,'cpu',1.,dt,.1,4.,.08,.08,.08,2.)
 xy=torch.zeros(1,4,2);contact=torch.ones(1,4,dtype=torch.bool);feet=list(range(4)) if feet is None else feet
 def step(c,h=0.,t=age):
  return m.update(xy,c,torch.full((1,4),h),torch.tensor([[1.,0.]]),torch.tensor([cmd]),torch.tensor([t]),torch.tensor([vx]))
 step(contact);air=contact.clone();air[:,feet]=False;step(air,height)
 xy[0,:,0]=torch.tensor(dist)
 for k in range(round(.08/dt)):step(air,height,age+(k+1)*dt)
 step(contact,t=age+.1)
 result=(float(m.reward)*dt,float(m.shortfall)*dt,float(m.variance_cost)*dt)
 return m,result,step,contact

distances=[-.01,0.,.002,.04,.079,.08,.081,.09,.1,.12]
for distance in distances:
 results=[]
 for foot in range(4):
  m,v,step,c=cycle([distance]*4,[foot]);results.append(v[:2]);assert not m.variance_event.any()
  step(c);assert not m.reward.any() and not m.shortfall.any() and not m.variance_cost.any()
 assert all(abs(a-b)<1e-7 for result in results[1:] for a,b in zip(results[0],result)),(distance,results)
 expected=float(signed_hind_stride(torch.tensor(distance),.08,.1,4.))
 assert abs(results[0][0]-max(expected,0)/4)<1e-6
 assert abs(results[0][1]-max(-expected,0)/4)<1e-6
# Physically brief/drag-only/no-command/startup do not earn or trigger a false landing cost.
for options in [dict(height=.001),dict(age=.5),dict(cmd=0.)]:
 m,v,_,_=cycle([.04]*4,**options);assert v==(0.,0.,0.)
assert cycle([.04]*4,vx=-.1)[1][1]>0
assert cycle([.1]*4,vx=-.1)[1][0]==0
for vals,expected_var in [([.08]*4,0.),([.1,.1,.06,.06],.0004),([.12,.12,.04,.04],.0016),([.2,.2,.06,.06],.0049)]:
 m,v,_,_=cycle(vals);assert m.variance_event.all() and abs(m.variance_m2.item()-expected_var)<1e-7
 assert abs(v[2]-expected_var/.08**2)<1e-6
 assert abs(v[2]-cycle(vals,dt=.01)[1][2])<1e-6
m,_,step,c=cycle([.1,.1,.06,.06]);assert m.variance_m2.item()>0
m.reset(torch.tensor([0]));assert not m.completed_valid.any() and not m.variance_cost.any() and not m.variance_event.any()
# Missing feet are excluded, not interpreted as zero-length steps; stale foot records cannot persist.
m,_,step,c=cycle([.1]*4,[0,1,2]);assert not m.variance_event.any()
m.completed_age[:]=0.;m.completed_valid[:]=True
step(c,t=5.);air=c.clone();air[:,3]=False;step(air,.003,5.02)
for k in range(4):step(air,.003,5.04+k*.02)
step(c,t=5.14);assert m.landing[0,3] and not m.variance_event.any() and not m.variance_cost.any()
for name in ['stride_shortfall','stride_variance']:
 obj=types.SimpleNamespace(cfg=SableCfg(),rew_buf=torch.zeros(1),reward_names=[name],reward_functions=[lambda:torch.ones(1)],reward_scales={name:-2.},episode_sums={name:torch.zeros(1)})
 SableRobot.compute_reward(obj);assert obj.rew_buf.item()==-2
assert not hasattr(SableRobot,'_reward_hind_stride_shortfall')
report={'passed':True,'front_hind_same_reward_and_shortfall_at_all_test_distances':True,'distance_cases_m':distances,'zero80_saturation100_verified':True,'no_forward_gate_for_negative_short_steps':True,'variance_population_definition':True,'variance_grows_quadratically_with_spread':True,'missing_and_stale_feet_excluded':True,'reset_clears_records':True,'unclamped_actual_stride_variance':True,'event_normalization_dt_invariant':True,'negative_costs_survive_clip':True,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')}}
(D/'verification/stride_all_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
