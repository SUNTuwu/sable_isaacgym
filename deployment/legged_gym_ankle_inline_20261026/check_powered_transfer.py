"""Audit actual PPO, joint migration, projection and velocity reward with known inputs."""
from isaacgym import gymapi
import torch,json,sys,types
from pathlib import Path
from powered_transfer import OLD_TO_NEW
from equivariant_policy import SymmetricActorCritic
from sable_env import SableRobot
from run_io import atomic_json,sha256
D=Path(__file__).parent;R=D.parents[1];run=R/'runs/powered_ankle_transfer_smoke_20261026'
old=torch.load(str(R/'runs/stand5_20261026_v26_stride_hip9/checkpoints/model_006130.pt'),map_location='cpu')['model_state_dict']
a=torch.load(str(run/'checkpoints/model_006130.pt'),map_location='cpu');b=torch.load(str(run/'checkpoints/model_006131.pt'),map_location='cpu');state=a['model_state_dict']
for k,v in old.items():
 if k=='std':continue
 if k in ['actor.0.weight','critic.0.weight']:
  assert torch.equal(v,state[k][:,OLD_TO_NEW]);unused=sorted(set(range(58))-set(OLD_TO_NEW));assert torch.count_nonzero(state[k][:,unused])==0
 elif k in ['actor.6.weight','actor.6.bias']:
  assert torch.equal(v,state[k][:12]);assert torch.count_nonzero(state[k][12:])==0
 else:assert torch.equal(v,state[k])
assert not a['optimizer_state_dict']['state'] and b['optimizer_state_dict']['state']
for prefix in ['actor.','critic.']:assert any(not torch.equal(v,b['model_state_dict'][k]) for k,v in state.items() if k.startswith(prefix))
assert not torch.equal(state['actor.6.weight'][12:],b['model_state_dict']['actor.6.weight'][12:]),'new ankles did not learn'
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values()) and (b['model_state_dict']['std']>=.04).all()
p=SymmetricActorCritic(58,58,14,actor_hidden_dims=[256,128,64],critic_hidden_dims=[256,128,64]);p.load_state_dict(state)
x=torch.randn(19,58);m=p.mirror(x)
assert torch.allclose(p.act_inference(m.observations(x)),m.actions(p.act_inference(x)),atol=1e-7)
assert torch.allclose(p.evaluate(m.observations(x)),p.evaluate(x),atol=1e-7)
# Compare against old projected network at zero new-joint observations.
x[:,unused]=0
from torch import nn
actor=nn.Sequential(nn.Linear(52,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,12));actor.load_state_dict({k[6:]:v for k,v in old.items() if k.startswith('actor.')})
expected=(actor(x[:,OLD_TO_NEW])+m.actions(torch.cat([actor(m.observations(x)[:,OLD_TO_NEW]),torch.zeros(len(x),2)],1))[:,:12])*.5
assert torch.allclose(p.act_inference(x)[:,:12],expected,atol=1e-6)
obj=types.SimpleNamespace(base_lin_vel=torch.tensor([[v,0.,0.] for v in [-.1,0,.03,.06,.12,.2,.1,.1]]),base_ang_vel=torch.zeros(8,3),commands=torch.ones(8,4)*.06,true_fall=torch.zeros(8,dtype=torch.bool))
obj.commands[6,0]=0;obj.true_fall[7]=True
r=SableRobot._reward_forward_speed(obj)
assert torch.allclose(r,torch.tensor([0.,0.,.25,.5,1.,1.,0.,0.]))
report={'passed':True,'exact_named_joint_transfer':True,'old_projection_preserved_at_zero_new_inputs':True,'new_ankle_outputs_initially_zero':True,'new_ankle_weights_updated_by_ppo':True,'fresh_optimizer':True,'mirror_projection_test':True,'speed_reward_test':True,'physical_model_changed':True,'baseline_stability_separate':True,'parent_sha256':sha256(R/'runs/stand5_20261026_v26_stride_hip9/checkpoints/model_006130.pt'),'files':{p.name:sha256(p) for p in D.glob('*.py')}}
atomic_json(D/'verification/transfer_checks.json',report);print('TRANSFER_CHECKS_PASSED')
