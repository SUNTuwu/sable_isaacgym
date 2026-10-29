"""Signed yaw integration and checkpoint/moment migration CPU contracts."""
import ast,json,math,runpy
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from teacher_yaw_integral import YawIntegral,expand_checkpoint,SCALE_RAD

q=YawIntegral(4,'cpu',.02)
actual=torch.tensor([.1,-.1,1.57,-1.57]);command=torch.tensor([0.,0.,1.57,-1.57])
for i in range(50):cost=q.update(actual,command)
assert torch.allclose(q.value,torch.tensor([.1,-.1,0.,0.]),atol=1e-6)
assert torch.allclose(cost[:2],torch.full((2,),-2*math.expm1(.1/SCALE_RAD)),atol=1e-5)
assert (cost[2:]==0).all()
for i in range(50):q.update(-actual[:2].repeat(2),torch.zeros(4))
assert q.value[:2].abs().max()<1e-6,'Signed opposite rates must cancel cumulative drift'
q.reset(torch.arange(4));assert not q.value.any()
for i in range(4000):cost=q.update(torch.ones(4)*1.57,torch.zeros(4))
assert q.value.min()>2*math.pi,'Integral must not wrap after a complete revolution'
assert torch.isfinite(cost).all() and torch.allclose(cost,torch.full((4,),-8.))
assert torch.isfinite(q.observation()).all() and q.observation().max()<=10

source=Path(__file__).with_name('teacher_bank_core.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Policy')
text='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
scope=dict(torch=torch,nn=nn,Normal=Normal,math=math)
exec(text,scope);New=scope['Policy']
for source_dim in (86,87,89):
 exec(text.replace('nn.Linear(94,256)','nn.Linear(%d,256)'%source_dim),scope);Old=scope['Policy']
 torch.manual_seed(27);old=Old();opt=torch.optim.Adam(old.parameters(),lr=3e-4)
 obs=torch.randn(16,source_dim)
 for i in range(2):
  loss=(old.actor(obs)-.1).square().mean()+(old.value(obs)-.2).square().mean()
  opt.zero_grad();loss.backward();opt.step()
 new=New();adapted,receipt=expand_checkpoint(dict(model_state_dict=old.state_dict(),optimizer_state_dict=opt.state_dict()),[n for n,_ in new.named_parameters()])
 new.load_state_dict(adapted['model_state_dict']);newopt=torch.optim.Adam(new.parameters());newopt.load_state_dict(adapted['optimizer_state_dict'])
 aug=torch.cat((obs,torch.randn(16,94-source_dim)*5),1)
 assert torch.allclose(old.actor(obs),new.actor(aug),atol=1e-6,rtol=1e-5)
 assert torch.allclose(old.value(obs),new.value(aug),atol=1e-6,rtol=1e-5)
 for (name,op),(_,np) in zip(old.named_parameters(),new.named_parameters()):
  for key,ov in opt.state[op].items():
   nv=newopt.state[np][key]
   if ov.ndim==2 and ov.shape[1]==source_dim:
    assert torch.equal(ov,nv[:,:source_dim]) and not nv[:,source_dim:].any()
   else:assert torch.equal(ov,nv)
 assert receipt['original_columns_equal'] and receipt['adam_moments_zero_padded']
runpy.run_path(str(Path(__file__).with_name('check_teacher_step_quality.py')))
runpy.run_path(str(Path(__file__).with_name('check_teacher_translation_guard.py')),run_name='__main__')
runpy.run_path(str(Path(__file__).with_name('check_teacher_lift_symmetry.py')),run_name='__main__')
print(json.dumps(dict(passed=True,checks=['signed_integral','task_relative_turn_integral','signed_cancellation',
 'reset','no_two_pi_wrap','finite_exponential_bound','86_87_89_to94_initial_actor_critic_equivalence','Adam_original_moments_preserved','new_input_moments_zero'])))
