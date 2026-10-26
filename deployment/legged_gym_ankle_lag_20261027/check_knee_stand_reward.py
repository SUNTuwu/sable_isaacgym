from isaacgym import gymapi
import torch,types,math,json,hashlib
from pathlib import Path
from sable_config import SableCfg
from sable_env import SableRobot
D=Path(__file__).parent;cfg=SableCfg()
standard=torch.tensor([[cfg.init_state.default_joint_angles[n] for n in ['L_knee_pitch','R_knee_pitch']]])
offset=torch.tensor([-40.,-20.,-11.,-10.,0.,10.,11.,20.,40.]);q=standard+offset[:,None]*math.pi/180
obj=types.SimpleNamespace(cfg=cfg,dof_pos=q,default_dof_pos=standard,knee_pitch_indices=torch.tensor([0,1]))
cost=SableRobot._reward_knee_stand_exp(obj)
assert torch.allclose(cost[3:6],torch.zeros(3),atol=1e-7)
assert (cost[0:3]>0).all() and (cost[6:]>0).all() and torch.allclose(cost,cost.flip(0),atol=1e-6)
assert cost[0]>cost[1]>cost[2] and cost[8]>cost[7]>cost[6]
assert torch.isfinite(cost).all()
obj.rew_buf=torch.zeros(9);obj.reward_names=['knee_stand_exp'];obj.reward_functions=[lambda:cost];obj.reward_scales={'knee_stand_exp':cfg.rewards.scales.knee_stand_exp};obj.episode_sums={'knee_stand_exp':torch.zeros(9)}
SableRobot.compute_reward(obj);assert (obj.rew_buf[:3]<0).all() and torch.allclose(obj.rew_buf,cost*(-2.5))
result={'passed':True,'training':False,'stand_deg':(standard[0]*180/math.pi).tolist(),'free_lower_deg':(standard[0]*180/math.pi-10).tolist(),'free_upper_deg':(standard[0]*180/math.pi+10).tolist(),'offsets_deg':offset.tolist(),'raw_cost':cost.tolist(),'negative_cost_survives_clipping':True,'scope':'Actual reward function unit checks only; no PPO or gait validation','source_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['sable_env.py','sable_config.py']}}
(D/'verification/knee_reward_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
