"""CPU-only semantic and shape checks; no simulator or policy update."""
import sys,math,json
from pathlib import Path
from types import SimpleNamespace as S
import torch
sys.path[:0]=[sys.argv[1],sys.argv[2]]
from teacher_motion_rewards import TeacherMotionRewards,direction_terms,directed_ground_angle_deg
from ankle_events import AnkleStepEvents

n=4;dev='cpu';cmd=torch.tensor([[.1,0,0],[-.1,0,0],[0,0,.175],[0,0,-.175]])
turn=torch.tensor([False,False,True,True])
right=direction_terms(cmd[:,:2],cmd[:,2],cmd,turn)
assert all(torch.equal(v,torch.zeros_like(v)) for v in right.values())
wrong=direction_terms(torch.zeros(4,2),torch.tensor([.2,.2,-.175,.175]),cmd,turn)
assert torch.all(wrong['task_yaw_cost']<0) and torch.all(wrong['task_wrong_turn_cost'][2:]<0)
angles=torch.tensor([30.,110.])*math.pi/180
axis=torch.stack([torch.cos(angles),torch.zeros(2),torch.sin(angles)],-1)[None]
computed=directed_ground_angle_deg(axis,torch.tensor([[1.,0.]]))
assert torch.allclose(computed,torch.tensor([[30.,110.]]),atol=1e-4)

# Preserve v44 landing/takeoff semantics, swapping only targets for backwalk.
event=AnkleStepEvents(2,dev,.02);event.landing_deg=torch.tensor([[30.],[110.]])
event.takeoff_deg=torch.tensor([[110.],[30.]])
touch=torch.ones(2,2,dtype=torch.bool);air=~touch;low=torch.zeros(2,2);high=torch.ones(2,2)*.003
xy=torch.zeros(2,2,2);command=torch.ones(2)*.1;age=torch.ones(2)
angle_landing=torch.tensor([[30.,30.],[110.,110.]])
angle_takeoff=torch.tensor([[110.,110.],[30.,30.]])
event.update(touch,low,angle_takeoff,xy,command,age)
for i in range(4):event.update(air,high,angle_takeoff,xy+torch.tensor([.006,0.]),command,age+(i+1)*.02)
assert not event.lifted.any(),'Startup takeoff must not be penalized'
event.update(touch,low,angle_landing,xy+torch.tensor([.008,0.]),command,age+.12)
assert event.landed.all() and torch.equal(event.event_cost,torch.zeros(2))
event.update(touch,low,angle_takeoff,xy+torch.tensor([.008,0.]),command,age+.14)
for i in range(3):event.update(air,high,angle_takeoff,xy+torch.tensor([.016,0.]),command,age+.16+i*.02)
assert event.lifted.all() and torch.equal(event.event_cost,torch.zeros(2))
assert torch.equal(event.takeoff_angle_deg,angle_takeoff),'Takeoff must latch last contact angle'

# Fake only tensor state; exercise the full apply/reset interface on CPU.
e=S(num_envs=n,device=dev,dt=.02,teacher_ids=torch.arange(n),teacher_commands=cmd,
    cfg=S(teacher_bank={'clips':[{'id':x} for x in ['01_FORWARD_POLICY','09_BACKWARD_WALK','04_TURN_LEFT','05_TURN_RIGHT']]},
          gait=S(force_sigma_n=15.,stance_speed_sigma_m_s=.05)))
e.base_quat=torch.tensor([[0.,0.,0.,1.]]).repeat(n,1);e.gravity_vec=torch.tensor([[0.,0.,-1.]]).repeat(n,1)
e.projected_gravity=e.gravity_vec.clone();e.base_ang_vel=torch.zeros(n,3);e.heading_linear_velocity=torch.zeros(n,2);e.heading_yaw_rate=torch.zeros(n)
e.episode_length_buf=torch.ones(n,dtype=torch.long)*60;e.common_step_counter=60
e.feet_indices=torch.tensor([1,2,3,4]);e.waist_body_index=0;e.penalised_contact_indices=torch.tensor([5]);e.proximal_indices=torch.arange(7)
e.rigid_states=torch.zeros(n,6,13);e.rigid_states[:,:,6]=1.;e.rigid_states[:,:,2]=.13
e.contact_forces=torch.zeros(n,6,3);e.contact_forces[:,e.feet_indices,2]=3.
e.foot_clearance=torch.ones(n,4)*.0005;e.foot_long_axis_world=torch.tensor([1.,0.,0.])[None,None].repeat(n,4,1)
e.foot_center_world=torch.zeros(n,4,3);e.foot_contact_point=torch.zeros(n,4,3);e.contact_foot_com=torch.zeros(4,3)
e.dof_pos=torch.zeros(n,16);e.dof_vel=torch.zeros(n,16);e.last_dof_vel=torch.zeros(n,16)
e.active=torch.arange(15);e.hard_limits=torch.tensor([[-2.,2.]]).repeat(16,1);e.executed_target=torch.zeros(n,15)
e.current_reference_target=torch.zeros(n,15);e.current_offset_rad=torch.zeros(n,15);e.torques=torch.zeros(n,16)
e.root_states=torch.zeros(n,13);e.root_states[:,2]=.13;e.env_origins=torch.zeros(n,3);e.true_fall=torch.zeros(n,dtype=torch.bool)
surface=torch.linspace(.0005,.02,51)[None,:,None].repeat(n,1,4);e.ref={'surface':surface}
ref={'q':torch.zeros(n,15),'quat':e.base_quat,'root':torch.eye(4)[None].repeat(n,1,1),'surface':e.foot_clearance}
ref['root'][:,2,3]=.13
bank=TeacherMotionRewards(e)
base={x:torch.ones(n) for x in ('velocity','yaw','orientation','height','foot_clearance','slip','torque','action_rate')}
out=bank.apply(base,ref)
assert all(v.shape==(n,) and torch.isfinite(v).all() for v in out.values())
for key,v in out.items():
    if key.startswith('walk_v44_'):assert torch.equal(v[2:],torch.zeros(2)),'Walk cost leaked into turns: '+key
assert torch.equal(bank.ankle.landing_deg[:,0],torch.tensor([30.,110.,30.,30.]))
assert torch.equal(bank.ankle.takeoff_deg[:,0],torch.tensor([110.,30.,110.,110.]))
bank.reset(torch.tensor([0,1]));assert not bank.ankle.initialized[:2].any()
assert torch.equal(bank.ankle.landing_deg[:,0],torch.tensor([30.,110.,30.,30.]))
assert torch.equal(bank.ankle.takeoff_deg[:,0],torch.tensor([110.,30.,110.,110.]))
print(json.dumps({'passed':True,'direction_signs':True,'angles30_110_distinct':True,
                 'backwalk_landing110_takeoff30':True,'startup_takeoff_excluded':True,'last_contact_takeoff_latched':True,
                 'full_apply_shape_and_finite':True,'walk_reward_does_not_leak_to_turns':True,'reset_clears_events':True,
                 'angle_targets_preserved_after_reset':True,
                 'gpu_work':False,'scope':'CPU semantic/tensor checks; not dynamic locomotion validation'}))
