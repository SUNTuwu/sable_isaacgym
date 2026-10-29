"""CPU contract checks: task sign, period, reward ordering, and resume migration."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import teacher_task_tracking as task

ids=['01_FORWARD_POLICY','04_TURN_LEFT','05_TURN_RIGHT','06_BACKWARD',
     '07_LATERAL_TROT_LEFT','08_LATERAL_TROT_RIGHT','09_BACKWARD_WALK','10_FORWARD_TROT']
raw=torch.tensor([[.16,0.,0.],[0.,0.,math.pi/18],[0.,0.,-math.pi/18],[-.15,0.,0.],
                  [0.,.025,0.],[0.,-.025,0.],[-.16,0.,0.],[.15,0.,0.]])
periods=torch.ones(8)
command,vw,yw=task.configure(ids,raw,periods,task.PROFILE,True)
assert torch.allclose(command[1:3,2],torch.tensor([math.pi/2,-math.pi/2]))
assert torch.equal(command[[0,3,4,5,6,7]],raw[[0,3,4,5,6,7]])
slow=task.configure(ids,raw,periods*2,task.PROFILE,True)[0]
assert torch.allclose(slow[1:3,2]*2,command[1:3,2])
assert vw.tolist()==[1.5,4.,4.,1.5,8.,8.,1.5,1.5]
assert yw.tolist()==[1.,8.,8.,1.,4.,4.,1.,1.]
sv=torch.ones(8)*.02;sy=torch.ones(8)*.15;sy[1:3]=1.1781
correct=task.tracking_terms(command[:,:2],command[:,2],command,sv,sy,vw,yw)
zero=task.tracking_terms(command[:,:2]*0,command[:,2]*0,command,sv,sy,vw,yw)
wrong=task.tracking_terms(-command[:,:2],-command[:,2],command,sv,sy,vw,yw)
assert torch.equal(correct['velocity'],vw) and torch.equal(correct['yaw'],yw)
assert (correct['yaw'][1:3]>zero['yaw'][1:3]).all() and (zero['yaw'][1:3]>wrong['yaw'][1:3]).all()
assert (correct['velocity'][4:6]>zero['velocity'][4:6]).all() and (zero['velocity'][4:6]>wrong['velocity'][4:6]).all()
drift=command[:,:2].clone();drift[4:6,0]=.04
bad=task.tracking_terms(drift,torch.ones(8)*.2,command,sv,sy,vw,yw)
assert (bad['velocity'][4:6]<correct['velocity'][4:6]).all()
assert (bad['yaw'][4:6]<correct['yaw'][4:6]).all()
# Use the real curriculum class without importing Isaac Gym or running a sim.
tree=ast.parse(Path(__file__).with_name('teacher_bank_core.py').read_text())
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AdaptiveTeacherCurriculum')
scope={'torch':torch};exec(compile(ast.Module(body=[klass],type_ignores=[]),'<curriculum>','exec'),scope)
env=SimpleNamespace(device='cpu',teacher_count=8,teacher_commands=command,num_envs=8)
c=scope['AdaptiveTeacherCurriculum'](env);c.sigma[1,2]=.35;c.error_ema[1,2]=.2
c.failures[1,3]=10;c.phase_prob[1,3]=.2;c.progress_ema[1]=.7;c.success_cycle_ema[1]=.6
before=c.teacher_state(1)
change=task.migrate_curriculum_command(c,1,raw[1],command[1])
assert change and abs(float(c.sigma[1,2])-1.178097)<1e-5
assert not c.progress_ema[1] and not c.success_cycle_ema[1]
for name in ['failures','phase_prob','genuine_events','survival_ema']:
 assert torch.equal(c.teacher_state(1)['values'][name],before['values'][name])
unchanged=c.teacher_state(4)
assert task.migrate_curriculum_command(c,4,raw[4],command[4]) is None
assert all(torch.equal(v,c.teacher_state(4)['values'][k]) for k,v in unchanged['values'].items())
print(json.dumps({'passed':True,'checks':['turn_sign_and_90deg_period','lateral_reference_speed','selective_weights',
 'correct_beats_stationary_beats_wrong','lateral_drift_and_yaw','changed_goal_bandwidth_recalibration',
 'failure_history_retained','unchanged_teacher_state_exact'],'commands':command.tolist(),'weights':{'velocity':vw.tolist(),'yaw':yw.tolist()}}))
