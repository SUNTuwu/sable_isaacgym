"""Reward boundary, false-positive and reset checks; no PPO or simulation."""
import hashlib,json,math,sys,types
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');RT=ROOT/'deployment/legged_gym_stride_hip9_20261026';sys.path.insert(0,str(RT))
from isaacgym import gymapi
import torch
from stride_progress import StrideProgress
from sable_env import SableRobot
from sable_config import SableCfg
cfg=SableCfg();cfg.gait.period_s=.8
assert cfg.rewards.scales.hip_stand_exp==-2.5 and cfg.posture.hip_preferred_max_deg==-9 and cfg.rewards.scales.stride_progress==2
values=[]
for angle in [-85,-20,-9,-8,0,24]:
 fake=types.SimpleNamespace(cfg=cfg,dof_pos=torch.full((1,2),math.radians(angle)),hip_pitch_indices=torch.tensor([0,1]))
 values.append(float(SableRobot._reward_hip_stand_exp(fake)))
assert values[:3]==[0.,0.,0.] and 0<values[3]<values[4]<values[5]
def exercise(distance=.048,height=.023,vx=.06,cmd=.06,age=2.,dt=.02,clear_ticks=100,reset=False):
 tracker=StrideProgress(1,'cpu',.8,dt);axis=torch.tensor([[1.,0.]])
 kwargs=[axis,torch.tensor([cmd]),torch.tensor([age]),torch.tensor([vx])]
 ticks=round(.2/dt)
 for i in range(ticks):
  xy=torch.zeros(1,4,2);xy[:,:,0]=distance*i/ticks
  tracker.update(xy,torch.zeros(1,4,dtype=torch.bool),torch.full((1,4),height if i<clear_ticks else 0.),*kwargs)
 if reset:tracker.reset(torch.tensor([0]))
 xy=torch.zeros(1,4,2);xy[:,:,0]=distance
 result=float(tracker.update(xy,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4),*kwargs))*dt/.8
 second=float(tracker.update(xy,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4),*kwargs))
 assert second==0
 return result
checks={'target_stride':exercise(),'half_stride':exercise(distance=.024),'too_low':exercise(height=.019),
 'backward':exercise(distance=-.048),'no_body_progress':exercise(vx=0.),'stop_command':exercise(cmd=0.),
 'startup':exercise(age=.1),'too_short_air':exercise(clear_ticks=2),'reset':exercise(reset=True),'dt_fine':exercise(dt=.01)}
assert abs(checks['target_stride']-1)<1e-6 and abs(checks['half_stride']-.5)<1e-6 and abs(checks['dt_fine']-1)<1e-6
assert all(checks[k]==0 for k in ['too_low','backward','no_body_progress','stop_command','startup','too_short_air','reset'])
report={'passed':True,'created_unix':__import__('time').time(),'checks':checks,'hip_angles_deg':[-85,-20,-9,-8,0,24],'hip_penalty_unweighted':values,
 'source_sha256':{n:hashlib.sha256((RT/n).read_bytes()).hexdigest() for n in ['stride_progress.py','sable_env.py','sable_config.py']},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out=ROOT/'results/legged_gym/stride_hip9_reward_checks_20261026.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
