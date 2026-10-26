from isaacgym import gymapi
import torch, math, json, hashlib, types
from pathlib import Path
from landing_stride_rewards import signed_hind_stride, landing_shortfall
from stride_progress import StrideProgress
from sable_env import SableRobot
from sable_config import SableCfg

D=Path(__file__).parent
dist=torch.tensor([-.01,0.,.025,.049,.05,.051,.06,.075,.1,.12])
assert SableCfg.stride.hind_zero_m==.05
score=signed_hind_stride(dist,zero=SableCfg.stride.hind_zero_m)
assert torch.all(score[1:]>=score[:-1]) and score[4]==0
assert torch.all(score[:4]<0) and torch.all(score[5:]>0)
assert score[-1]==score[-2]==1 and score[0]==score[1]==-1
angles=torch.tensor([0.,15.,25.,35.,44.,45.,46.,90.])
cost=landing_shortfall(angles)
assert torch.all(cost[:-1]>=cost[1:]) and torch.all(cost[5:]==0) and torch.all(cost[:5]>0)
assert torch.isfinite(cost).all()

def event(distance,peak=.023,vx=.1,dt=.02):
    m=StrideProgress(1,'cpu',.8,dt,hind_zero_m=SableCfg.stride.hind_zero_m)
    contact=torch.ones(1,4,dtype=torch.bool);xy=torch.zeros(1,4,2)
    def update(c,z):
        return m.update(xy,c,torch.full((1,4),z),torch.tensor([[1.,0.]]),torch.tensor([.1]),torch.tensor([2.]),torch.tensor([vx]))
    update(contact,0.);update(~contact,peak)
    xy[:,:,0]=distance
    for _ in range(round(.1/dt)):update(~contact,peak)
    update(contact,0.)
    value=(float(m.reward)*dt,float(m.shortfall)*dt)
    update(contact,0.);assert not m.reward.any() and not m.shortfall.any()
    m.reset(torch.tensor([0]));assert not m.hind_landing.any() and not m.shortfall.any()
    return value

low_long=event(.07,peak=.0187);high_long=event(.07,peak=.023)
assert low_long[0]>0 and abs(low_long[0]-high_long[0])<1e-7
short=event(.02);low=event(.002,peak=.004,vx=-.01);back=event(-.01);long=event(.07);threshold=event(.05);drag=event(.002,peak=.001)
assert short[1]>0 and low[1]>0 and back[1]>0 and long[1]==threshold[1]==drag[1]==0
assert abs(short[1]-event(.02,dt=.01)[1])<1e-6
# Costs are applied after the positive-only clamp; a cost-only step stays negative.
names=['ankle_landing_low','hind_stride_shortfall']
obj=types.SimpleNamespace(cfg=SableCfg(),rew_buf=torch.zeros(1),reward_names=names,reward_functions=[lambda:torch.ones(1)]*2,reward_scales={k:-2. for k in names},episode_sums={k:torch.zeros(1) for k in names})
SableRobot.compute_reward(obj);assert obj.rew_buf.item()==-4.
reference=json.loads((D/'assets/gait_reference.json').read_text());assert reference['period_s']==1. and reference['stance_fraction']==.75 and reference['minimum_limit_margin_deg']>0
train=(D/'train.py').read_text();assert 'cfg.gait.period_s=1.;' in train
from gait_support import GaitSignals
env=types.SimpleNamespace(episode_length_buf=torch.tensor([0,12,25,37,50,100]),dt=.02,cfg=types.SimpleNamespace(gait=types.SimpleNamespace(period_s=1.)),gait_phase_offset=torch.zeros(6))
phase=GaitSignals.phase(None,env);assert torch.allclose(phase,torch.tensor([0.,.24,.5,.74,0.,0.]),atol=1e-6)
report={'passed':True,'period_s':1.,'frequency_hz':1.,'swing_duration_s':.25,'long_stride18_7mm_height_still_rewarded':low_long[0],'reference_min_limit_margin_deg':reference['minimum_limit_margin_deg'],'reference_max_ik_error_m':reference['max_ik_error_m'],'distance_mm':(dist*1000).tolist(),'signed_score':score.tolist(),'landing_angles_deg':angles.tolist(),'landing_cost':cost.tolist(),'short_step':short,'low_backward_velocity_step':low,'backward_step':back,'long_step':long,'zero_threshold':threshold,'drag':drag,'negative_cost_survives_clipping':True,'source_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['landing_stride_rewards.py','stride_progress.py','ankle_phase.py','sable_env.py','sable_config.py','evaluate.py','train.py','guided_reference.py','assets/gait_reference.json']}}
(D/'verification/cadence_stride_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
