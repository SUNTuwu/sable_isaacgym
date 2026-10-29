import os,sys,math
os.environ.setdefault('SABLE_GYM_ROOT','.')
sys.path.insert(0,sys.argv[1])
import torch
from types import SimpleNamespace as N
from night_tuning import apply
n=3
root=torch.zeros(n,13);angles=torch.tensor([0.,math.pi/2,math.pi]);root[:,5]=torch.sin(angles/2);root[:,6]=torch.cos(angles/2)
root[:,7]=.075*angles.cos();root[:,8]=.075*angles.sin()
e=N(cfg=N(teacher_bank={'clips':[{'id':'FORWARD'}]}),teacher_ids=torch.zeros(n,dtype=torch.long),root_states=root,translation_guard=N(start_yaw=torch.zeros(n)),
    teacher_commands=torch.tensor([[.075,0,0.]]),heading_yaw_rate=torch.tensor([0.,.5,1.]),dt=.02,night_control={'teachers':{'FORWARD':{'straight_frame':True,'yaw_sigma':.4}}})
terms={key:torch.zeros(n) for key in ['v2_velocity','v2_heading_integral','v2_translation_yaw_rate','v2_fall_event']}
out=apply(e,terms,torch.zeros(n,dtype=torch.bool))
assert out['v2_velocity'][0]>5.9 and out['v2_velocity'][1]<.01 and out['v2_velocity'][2]<-1.9
assert out['v2_heading_integral'][0]==0 and out['v2_heading_integral'][1]<-3.9
assert out['v2_translation_yaw_rate'][2]<out['v2_translation_yaw_rate'][1]<0
print('Fixed heading: forward rewarded, 90deg circle loses progress, backwards negative, yaw cost not flat at0.5/1rad/s')
