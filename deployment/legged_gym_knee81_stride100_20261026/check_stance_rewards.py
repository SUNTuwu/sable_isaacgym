from isaacgym import gymapi
import torch,math,json,types
from pathlib import Path
from ankle_stance import StanceAnkleMotion
from sable_env import SableRobot
from sable_config import SableCfg
from run_io import atomic_json,sha256
D=Path(__file__).parent;tests={}
def run_sequence(dt=.02,oscillate=False,park=False,bad_angle=False,chatter=False):
 m=StanceAnkleMotion(1,'cpu',dt);q=torch.ones(1,2)*math.radians(70);cmd=torch.tensor([0. if park else .06]);vx=torch.tensor([.04]);age=torch.tensor([2.]);angles=torch.ones(1,2)*(110 if bad_angle else 60)
 total=0.;reversal=0.
 for i in range(round(.08/dt)):m.update(q,torch.zeros(1,2,dtype=torch.bool),torch.ones(1,2)*.005,angles,cmd,vx,age)
 for i in range(round(.8/dt)):
  angle=70+min(30,i*dt/.4*30) if not oscillate else 70+(15 if i%2 else 0)
  q[:]=math.radians(angle)
  contact=torch.ones(1,2,dtype=torch.bool)
  if chatter and i%3==2:contact[:]=False
  m.update(q,contact,torch.zeros(1,2),angles,cmd,vx,age)
  total+=m.reward.item()*dt;reversal+=m.reversal_cost.item()*dt
 return total,reversal
normal=run_sequence();osc=run_sequence(oscillate=True);park=run_sequence(park=True);bad=run_sequence(bad_angle=True);chatter=run_sequence(chatter=True);fine=run_sequence(dt=.01)
assert abs(normal[0]-.8)<1e-5 and normal[1]<1e-5
assert osc[0]<=.8+1e-5 and osc[1]>normal[1]
assert park[0]==0 and bad[0]==0 and chatter[0]==0
assert abs(fine[0]-normal[0])<1e-5
m=StanceAnkleMotion(1,'cpu',.02);m.reset(torch.tensor([0]));m.update(torch.ones(1,2),torch.ones(1,2,dtype=torch.bool),torch.zeros(1,2),torch.ones(1,2)*60,torch.ones(1)*.06,torch.ones(1)*.04,torch.ones(1)*2);assert m.reward.item()==0 and not m.active.any()
obj=types.SimpleNamespace(cfg=SableCfg(),hip_pitch_indices=torch.tensor([0,1]),dof_pos=torch.tensor([[-30.,-30.],[-19.,-19.],[-18.,-18.],[0.,0.]])*math.pi/180)
cost=SableRobot._reward_hip_stand_exp(obj);assert torch.allclose(cost[:2],torch.zeros(2),atol=1e-7) and (cost[2:]>0).all()
obj.hind_foot_angle_deg=torch.tensor([[0.,0.],[60.,60.],[89.,89.],[90.,90.],[120.,120.]])
fc=SableRobot._reward_hind_foot_angle(obj);assert torch.allclose(fc[:3],torch.zeros(3)) and fc[3]>0 and fc[4]>fc[3]
# Directed angle discriminates backwards feet (the unoriented line-plane angle cannot).
axis=torch.tensor([[1.,0.,0.],[0.,0.,-1.],[-.5,0.,-math.sqrt(.75)]])
angles=torch.acos(axis[:,0])*180/math.pi;assert torch.allclose(angles,torch.tensor([0.,90.,120.]),atol=1e-5)
obj.rew_buf=torch.zeros(5);obj.reward_names=['hind_foot_angle'];obj.reward_functions=[lambda:fc];obj.reward_scales={'hind_foot_angle':-4.};obj.episode_sums={'hind_foot_angle':torch.zeros(5)};SableRobot.compute_reward(obj);assert torch.allclose(obj.rew_buf,-4*fc)
report={'passed':True,'normal_integrated_range_reward':normal,'oscillatory_reward_and_reversal_cost':osc,'park':park,'backward_foot':bad,'contact_chatter':chatter,'dt01':fine,'reset_has_no_reward':True,'hip_free_max_deg':-19,'hip_boundary_checked':True,'directed_foot_angle_deg':angles.tolist(),'foot_negative_cost_survives_clipping':True,'scope':'Synthetic reward/geometry convention checks, not learned gait evidence','source_sha256':{n:sha256(D/n) for n in ['ankle_stance.py','sable_env.py','sable_config.py','gait_support.py']}}
atomic_json(D/'verification/reward_checks.json',report);print(json.dumps(report,indent=2))
