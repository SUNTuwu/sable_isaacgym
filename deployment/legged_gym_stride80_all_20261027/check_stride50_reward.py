from isaacgym import gymapi
import torch,json,hashlib
from pathlib import Path
from stride_progress import StrideProgress
from sable_config import SableCfg
D=Path(__file__).parent
# One true landing per row, forward placements20/50/80mm, backward/lateral, low/no flight and parked controls.
n=8;z=torch.zeros(n,4);xy=torch.zeros(n,4,2);forward=torch.tensor([[1.,0.]]).repeat(n,1);cmd=torch.full((n,),.1);cmd[7]=0.;age=torch.full((n,),2.);v=torch.full((n,),.1)
m=StrideProgress(n,'cpu',.8,.02,cap_m=.05);contact=torch.zeros(n,4,dtype=torch.bool);clear=torch.full((n,4),.023);clear[5]=.001
for _ in range(4):m.update(xy,contact,clear,forward,cmd,age,v)
xy[:,:,0]=torch.tensor([.02,.05,.08,-.05,0.,.08,0.,.08])[:,None];xy[4,:,1]=.08
reward=m.update(xy,torch.ones_like(contact),z,forward,cmd,age,v).clone();assert torch.allclose(reward[:3],torch.tensor([16.,40.,40.])) and torch.all(reward[3:]==0)
assert torch.all(m.update(xy,torch.ones_like(contact),z,forward,cmd,age,v)==0),'No repeated landing reward'
m.reset(torch.arange(n));assert torch.all(m.update(xy,torch.ones_like(contact),z,forward,cmd,age,v)==0)
cfg=SableCfg();assert cfg.rewards.scales.ankle_stance_motion==4 and cfg.stride.reward_cap_m==.05
r={'passed':True,'forward_20_50_80_mm_rewards':reward[:3].tolist(),'50mm_saturation':True,'no_backward_lateral_park_contact_chatter_reward':True,'no_repeat_reward':True,'ankle_weight':4,'stride_weight':cfg.rewards.scales.stride_progress,'source_sha256':{k:hashlib.sha256((D/k).read_bytes()).hexdigest() for k in ['stride_progress.py','sable_env.py','gait_support.py','sable_config.py','train.py','evaluate.py']}}
(D/'verification/stride50_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
