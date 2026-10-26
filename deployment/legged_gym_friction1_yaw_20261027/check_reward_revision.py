from isaacgym import gymapi
import torch,math,json,types,hashlib
from pathlib import Path
from sable_config import SableCfg
from sable_env import SableRobot
from stride_progress import StrideProgress
D=Path(__file__).parent;cfg=SableCfg();standard=torch.tensor([[cfg.init_state.default_joint_angles[n] for n in ['L_knee_pitch','R_knee_pitch']]])
center=standard+math.radians(-10);off=torch.tensor([-11.,-10.,0.,10.,11.,20.]);q=center+off[:,None]*math.pi/180
o=types.SimpleNamespace(cfg=cfg,dof_pos=q,default_dof_pos=standard,knee_pitch_indices=torch.tensor([0,1]));cost=SableRobot._reward_knee_stand_exp(o);assert torch.allclose(cost[1:4],torch.zeros(3),atol=1e-7) and cost[0]>0 and cost[4]>0 and cost[5]>cost[4]
n=7;m=StrideProgress(n,'cpu',.8,.02,.10,4.);xy=torch.zeros(n,4,2);c=torch.zeros(n,4,dtype=torch.bool);clear=torch.full((n,4),.023);axis=torch.tensor([[1.,0.]]).repeat(n,1);cmd=torch.full((n,),.1);age=torch.full((n,),2.);v=torch.full((n,),.1)
for _ in range(4):m.update(xy,c,clear,axis,cmd,age,v)
dist=torch.tensor([.01,.025,.05,.075,.1,.15,-.1]);xy[:,:,0]=dist[:,None];reward=m.update(xy,~c,clear*0,axis,cmd,age,v)/40
expected=torch.expm1(4*(dist/.1).clamp(0,1))/math.expm1(4);assert torch.allclose(reward,expected,atol=1e-6) and reward[2]<.13 and reward[4]==reward[5] and reward[-1]==0
assert torch.all(m.update(xy,~c,clear*0,axis,cmd,age,v)==0)
result={'passed':True,'knee_free_lower_deg':(center[0]*180/math.pi-10).tolist(),'knee_free_upper_deg':(center[0]*180/math.pi+10).tolist(),'stride_distances_m':dist.tolist(),'stride_reward_fraction':reward.tolist(),'stride_exponent':4,'stride_cap_m':.1,'physical_assets_changed':False,'source_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['sable_config.py','sable_env.py','stride_progress.py','evaluate.py']}}
(D/'verification/reward_revision_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
