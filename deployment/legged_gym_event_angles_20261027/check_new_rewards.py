from isaacgym import gymapi
import torch,json,hashlib,types,math,xml.etree.ElementTree as ET
from pathlib import Path
from ankle_events import AnkleStepEvents
from waist_acceleration import WaistHeightAcceleration
from sable_config import SableCfg
from sable_env import SableRobot
from legged_gym.utils.helpers import class_to_dict
D=Path(__file__).parent
m=AnkleStepEvents(1,'cpu',.02);age=2.;x=0.
def step(c,h=0.,angle=70.,pos=None,cmd=.1):
 global age,x
 age+=.02
 if pos is not None:x=pos
 m.update(torch.full((1,2),c,dtype=torch.bool),torch.full((1,2),h),torch.full((1,2),float(angle)),torch.tensor([[[x,0.],[x,0.]]]),torch.tensor([cmd]),torch.tensor([age]))
 return bool(m.landed[0,0]),bool(m.lifted[0,0])
assert step(True)==(False,False)
# Chatter and too-short flights never count.
for _ in range(4):
 assert step(False,.003,pos=x+.006)==(False,False)
 assert step(True,angle=30)==(False,False)
# Startup qualifying flight can land, but has no preceding landing for takeoff.
for i in range(3):assert step(False,.003,pos=x+.006)==(False,False)
assert step(True,angle=30)==(True,False) and m.event_cost.item()==0
first=int(m.landing_id[0,0])
for _ in range(4):
 assert step(False,.0001,angle=55)==(False,False)
 assert step(True,angle=110)==(False,False)
# Real takeoff stores110 at last contact, not60 at delayed confirmation.
for i in range(3):
 flags=step(False,.003,angle=60,pos=x-.006)
 assert flags==(False,i==2),flags
assert m.takeoff_angle_deg[0,0]==110 and int(m.takeoff_pair_id[0,0])==first and m.event_cost.item()==0
assert m.confirmation_time_s[0,0]>m.takeoff_time_s[0,0]
for _ in range(8):assert step(False,.02,angle=20,pos=x-.006)==(False,False) and m.event_cost.item()==0
assert step(True,angle=0)==(True,False)
assert abs(m.event_cost.item()-2*math.expm1(1)*m.factor)<1e-4
for _ in range(5):assert step(True,angle=110)==(False,False) and m.event_cost.item()==0
#60ms with <5mm is not enough; displacement reaching5mm at landing confirms exactly once.
base=x
for _ in range(3):assert step(False,.003,pos=base+.004)==(False,False)
assert step(True,angle=30,pos=base+.005)==(True,True)
assert step(True)==(False,False)
m.reset(torch.tensor([0]));assert all(not v.any() for v in vars(m).values() if torch.is_tensor(v))
# Parking/startup suppression.
for _ in range(4):step(True,cmd=0);assert step(False,.01,pos=x+.01,cmd=0)==(False,False)
angles=torch.tensor([30.,29.,31.,0.,60.,180.]);cost=m.angle_cost(angles,30)
assert cost[0]==0 and cost[1]==cost[2] and cost[3]==cost[4] and cost[1]>0 and torch.isfinite(cost).all()
#World-height second difference detects both signs; translation/constant speed costs0; isolated resets do not spike.
w=WaistHeightAcceleration(3,'cpu',.02)
for k in range(6):
 t=k*.02;w.update(torch.tensor([100.+t, .1+.5*4*t*t,.1-.5*4*t*t],dtype=torch.float64))
 # float32 storage at large offset has roundoff; exact test below uses small coordinate.
 if k>=2:assert torch.allclose(w.acceleration[1:],torch.tensor([4.,-4.],dtype=torch.float64),atol=.001)
w.reset(torch.tensor([1]));w.update(torch.tensor([.1,50.,.1]));assert w.acceleration[1]==0
w.update(torch.tensor([.1,70.,.1]));assert w.acceleration[1]==0
w2=WaistHeightAcceleration(1,'cpu',.02)
for k in range(8):
 w2.update(torch.tensor([.1+k*.002]));assert w2.acceleration.abs().max()<.0001
anchor=ET.parse(D/'assets/sable.xml').find(".//body[@name='spine_pitch']");assert anchor is not None and anchor.find("joint[@name='spine_pitch']").get('pos','0 0 0')=='0 0 0'
sc=class_to_dict(SableCfg.rewards.scales)
removed=['ankle_landing_low','ankle_phase_angle','ankle_phase_event','ankle_phase_lag','ankle_phase_reverse','knee_stand_exp','hip_stand_exp']
assert all(k not in sc for k in removed) and len(sc)==26 and sc['ankle_event_angle']==-2 and sc['waist_vertical_acceleration']==-1
assert not hasattr(SableRobot,'_reward_knee_stand_exp')
assert SableCfg.material_friction.front_scale==1 and SableCfg.gait.stance_fraction==.5 and SableCfg.gait.period_s==1
assert sc['yaw_rate_error']==-2 and SableCfg.stride.hind_zero_m==.08 and SableCfg.termination.knee_contact_enabled
for name in ['waist_vertical_acceleration','ankle_event_angle']:
 obj=types.SimpleNamespace(cfg=SableCfg(),rew_buf=torch.zeros(1),reward_names=[name],reward_functions=[lambda:torch.ones(1)],reward_scales={name:sc[name]},episode_sums={name:torch.zeros(1)})
 SableRobot.compute_reward(obj);assert obj.rew_buf.item()==sc[name]
report={'passed':True,'active_reward_count':len(sc),'removed_keys':removed,'contact_chatter_no_duplicate_events':True,'startup_takeoff_excluded':True,'confirmed_takeoff_uses_last_contact_angle':True,'backward_real_flight_counted':True,'landing_threshold_crossing_tested':True,'waist_anchor_confirmed_from_mjcf':True,'waist_reset_samples_excluded':True,'waist_acceleration_sign_symmetric':True,'negative_costs_survive_positive_clip':True,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')}}
(D/'verification/new_rewards_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
