from isaacgym import gymapi
import torch,json,math,types
from pathlib import Path
from sable_config import SableCfg,MODEL
from sable_env import SableRobot
from run_io import atomic_json,sha256
D=Path(__file__).parent;R=D.parents[1];T=R/'runs/ankle_inline_transfer_smoke_20261026';P=R/'runs/stand5_20261026_v27_powered_ankles/checkpoints/model_006401.pt'
p=torch.load(str(P),map_location='cpu');a=torch.load(str(T/'checkpoints/model_006401.pt'),map_location='cpu');b=torch.load(str(T/'checkpoints/model_006402.pt'),map_location='cpu')
assert all(torch.equal(v,a['model_state_dict'][k]) for k,v in p['model_state_dict'].items() if k!='std')
assert not a['optimizer_state_dict']['state'] and b['optimizer_state_dict']['state']
for prefix in ['actor.','critic.']:assert any(not torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith(prefix))
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values()) and (b['model_state_dict']['std']>=.04).all()
cfg=SableCfg();config=json.load(open(T/'config.json'));assert config['env']['posture']['hip_preferred_max_deg']==-9 and config['env']['rewards']['scales']['hip_stand_exp']==-2.5
angles=torch.tensor([-85.,-24.,-9.,-8.,0.,24.]);q=angles[:,None].repeat(1,2)*math.pi/180
obj=types.SimpleNamespace(cfg=cfg,dof_pos=q,hip_pitch_indices=torch.tensor([0,1]))
cost=SableRobot._reward_hip_stand_exp(obj);assert torch.allclose(cost[:3],torch.zeros(3),atol=1e-7) and (cost[3:]>0).all() and (cost[4:]>cost[3:-1]).all()
obj.rew_buf=torch.zeros(6);obj.reward_names=['hip_stand_exp'];obj.reward_functions=[lambda:cost];obj.reward_scales={'hip_stand_exp':-2.5};obj.episode_sums={'hip_stand_exp':torch.zeros(6)}
SableRobot.compute_reward(obj);assert torch.allclose(obj.rew_buf,-2.5*cost);assert (obj.rew_buf[3:]<0).all(),'posture must survive positive-only clipping'
physical=json.load(open(D/'assets/physical_manifest.json'));auth=json.load(open(D/'assets/authored_model.json'))
for side in ['L','R']:
 j=next(r for r in auth['joints'] if r['name']==side+'_hip_pitch');assert max(abs(x-y) for x,y in zip(j['limits_deg'],[-85,24]))<1e-4
report={'passed':True,'exact_network_transfer':True,'fresh_optimizer':True,'actual_ppo_actor_critic_update':True,'hip_penalty_enabled':True,'hip_preferred_max_deg':-9,'hip_weight':-2.5,'hip_exponential_scale_deg':15,'only_angles_le_minus9_exempt':True,'negative_cost_survives_positive_clipping':True,'angles_deg':angles.tolist(),'weighted_cost_per_s':(-2.5*cost).tolist(),'physical_hip_limits_deg':[-85,24],'parent_sha256':sha256(P),'env_sha256':sha256(D/'sable_env.py'),'config_sha256':sha256(D/'sable_config.py')}
atomic_json(D/'verification/inline_transfer_checks.json',report);print(json.dumps(report,indent=2))
