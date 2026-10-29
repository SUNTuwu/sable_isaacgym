"""CPU boundaries and task-aware v44 event contract; no local RL workload."""
from types import SimpleNamespace as S
import json,math
import torch
from teacher_step_quality import *

h=torch.tensor([[.009,.01,.02,.03,.05]]);d=torch.ones_like(h)*.10
ones=torch.ones_like(h,dtype=torch.bool)
r=event_scores(h,d,ones,ones)
assert r['step_low_height_cost'].tolist()==[[-2.,0.,0.,0.,0.]]
assert r['step_height_reward'][0,0]==0 and r['step_height_reward'][0,1]==0
assert 0<float(r['step_height_reward'][0,2])<2
assert torch.allclose(r['step_height_reward'][0,3:],torch.tensor([2.,2.]))
d=torch.tensor([[0.,.019,.02,.049,.05,.075,.10,.20]])
r=event_scores(torch.ones_like(d)*.03,d,torch.ones_like(d,dtype=torch.bool),d>=.02)
assert r['step_invalid_distance_cost'].tolist()==[[-2.,-2.,0.,0.,0.,0.,0.,0.]]
assert (r['step_distance_reward'][0,:5]==0).all() and 0<float(r['step_distance_reward'][0,5])<3
assert torch.allclose(r['step_distance_reward'][0,6:],torch.tensor([3.,3.]))
width,hard=placement_scores(torch.tensor([.02,.03,.04,.05,.06,.0602,-.04]))
assert (width[1:4].abs()<1e-5).all() and width[0]<0 and width[4]<width[0]+1e-5
assert hard[4]==0 and hard[5]==-2 and width[6]<0
cmd=torch.tensor([[.1,0,0],[-.1,0,0],[0,.025,0],[0,-.025,0],[0,0,1.57],[0,0,-1.57]])
cost=strict_direction_terms(cmd[:,:2],cmd[:,2],cmd)
assert all((v==0).all() for v in cost.values())
badxy=cmd[:,:2].clone();badxy[:2,1]=.02;badxy[2:4,0]=.02
cost=strict_direction_terms(badxy,torch.ones(6)*.15,cmd)
assert (cost['task_cross_axis_cost'][:4]==-2).all() and (cost['translation_zero_yaw_cost'][:4]==-2).all()
assert (cost['translation_zero_yaw_cost'][4:]==0).all()
huge=strict_direction_terms(torch.ones(6,2)*100,torch.ones(6)*100,cmd)
assert all((v>=-4).all() for k,v in huge.items() if k!='yaw_error_squared_cost')
assert torch.allclose(cost['yaw_error_squared_cost'],-8*(torch.ones(6)*.15-cmd[:,2]).square())

def fixture(command=(.1,0,0)):
    n=1;xy=torch.tensor([[[.1,.04],[.1,-.04],[-.06,.04],[-.06,-.04]]])
    center=torch.cat((xy,torch.zeros(n,4,1)),2)
    e=S(num_envs=n,device='cpu',dt=.02,teacher_ids=torch.zeros(n,dtype=torch.long),ref_duration=torch.ones(1),
        foot_center_world=center.clone(),foot_contact_point=center.clone(),foot_clearance=torch.zeros(n,4),
        root_states=torch.zeros(n,13),heading_linear_velocity=torch.tensor([[command[0],command[1],0.]]),
        heading_yaw_rate=torch.tensor([command[2]]),true_fall=torch.zeros(n,dtype=torch.bool),
        dof_vel=torch.zeros(n,15),active=torch.arange(15),episode_length_buf=torch.ones(n,dtype=torch.long)*60)
    q=TeacherStepQuality(e,torch.tensor([command]));forward=torch.tensor([[1.,0.]])
    def tick(contact,age):
        return q.apply({'task_yaw_cost':torch.zeros(n)},forward,torch.full((n,4),contact,dtype=torch.bool),torch.tensor([age]))
    return e,q,tick,center

def step_test(command,travel,height=.02,frames=10):
    e,q,tick,start=fixture(command);tick(True,1.2)
    direction=torch.tensor(command[:2]);turn=abs(command[2])>.001
    if turn:
        radial=start[:,:,:2];axis=torch.stack((-radial[:,:,1],radial[:,:,0]),-1)*math.copysign(1,command[2]);axis/=axis.norm(dim=2,keepdim=True)
    else:axis=(direction/direction.norm())[None,None].expand(1,4,2)
    for j in range(frames):
        e.foot_clearance[:]=height;e.foot_center_world[:,:,:2]=start[:,:,:2]+axis*travel*(j+1)/frames
        tick(False,1.22+j*.02)
    e.foot_clearance[:]=0;e.foot_contact_point.copy_(e.foot_center_world)
    out=tick(True,1.22+frames*.02)
    return e,q,tick,out

for command in [(0.1,0,0),(-.1,0,0),(0,.025,0),(0,-.025,0),(0,0,1.57),(0,0,-1.57)]:
    e,q,tick,out=step_test(command,.08)
    assert q.valid.all(),('valid directed step',command,q.diagnostics)
    assert out['step_distance_reward'][0]>0 and out['step_height_reward'][0]>0
    assert q.statistics(1,1)[0,0]==4
    tick(True,1.5);assert not q.valid.any(),'One landing must not be counted twice'
for distance,genuine,valid in [(.001,False,False),(.010,True,False),(-.08,True,False),(.025,True,True)]:
    e,q,tick,out=step_test((.1,0,0),distance)
    assert bool(q.diagnostics['quality_genuine_landing'].all())==genuine
    assert bool(q.valid.all())==valid
    assert out['step_invalid_distance_cost'][0]<0 if distance<.02 else out['step_invalid_distance_cost'][0]==0
    if not valid:assert out['step_height_reward'][0]==0 and out['step_distance_reward'][0]==0
e,q,tick,out=step_test((.1,0,0),.08,frames=1)
assert not q.diagnostics['quality_attempted_landing'].any(),'Contact chatter cannot create an event'
e,q,tick,start=fixture();tick(False,1.2);e.foot_clearance[:]=.02
for j in range(4):e.foot_center_world[:,:,0]+=.02;tick(False,1.22+j*.02)
tick(True,1.4);assert not q.valid.any(),'Startup flight has no real takeoff'
q.reset(torch.tensor([0]));e.dof_vel[:]=100;e.episode_length_buf[:]=0
out=tick(True,0);assert out['joint_acceleration_excess_cost'][0]==0
e.episode_length_buf[:]=2;e.dof_vel[:]+=3
out=tick(True,.04);assert abs(float(out['joint_acceleration_excess_cost'][0]))<1e-5
e.dof_vel[:]+=6;out=tick(True,.06);assert out['joint_acceleration_excess_cost'][0]<0
from teacher_step_quality import SwingDurationTracker
for count in [1,2,3,8,9,10,11,15]:
 tracker=SwingDurationTracker(1,'cpu',.02);touch=torch.ones(1,4,dtype=torch.bool);clear=torch.zeros(1,4)
 tracker.update(touch,clear)
 for j in range(count):tracker.update(~touch,clear+.003)
 landed,duration,short=tracker.update(touch,clear)
 assert bool(landed.all())==(count>=3) and torch.allclose(duration,torch.ones(1,4)*count*.02)
 assert bool(short.all())==(3<=count<10)
 tracker.update(touch,clear);assert not tracker.update(touch,clear)[0].any()
tracker=SwingDurationTracker(1,'cpu',.02)
tracker.update(torch.zeros(1,4,dtype=torch.bool),torch.ones(1,4)*.02)
assert not tracker.update(torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4))[0].any()
e,q,tick,out=step_test((.1,0,0),.08,frames=9)
assert not q.valid.any() and out['step_short_duration_cost'][0]<0
assert out['step_height_reward'][0]==0 and out['step_distance_reward'][0]==0
e,q,tick,out=step_test((.1,0,0),.08,frames=10)
assert q.valid.all() and out['step_short_duration_cost'][0]==0
print(json.dumps(dict(passed=True,checks=['height10_30_and_fixed_low_cost','stride20_50_100','width30_50_60_and_crossing',
 'strict_cross_axis_and_translation_yaw','all_six_task_directions','no_chatter_or_startup_or_double_landings',
 'sub5mm_air_attempt_penalized_without_valid_event','reverse_and_short_no_positive_rewards','reset_and_acceleration_threshold','duration20_to300ms_boundary','sub60ms_chatter_excluded','short_swing_no_positive_reward','yaw_error_true_square_and_turn_target'])))
