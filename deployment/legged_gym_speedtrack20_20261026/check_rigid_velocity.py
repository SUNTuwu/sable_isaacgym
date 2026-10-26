"""Bounded remote fixture to identify the frame of Gym rigid-body linear velocity."""
import sys,json,math
from pathlib import Path
runtime=Path(sys.argv[1]).resolve();sys.path.insert(0,str(runtime))
from isaacgym import gymapi
import torch,numpy as np
from preflight import create_env
from sable_config import SableCfg,MODEL
from run_io import atomic_json
cfg=SableCfg();cfg.init_state.pos=list(cfg.init_state.pos);cfg.init_state.pos[2]+=.1
env=create_env(1,fixed=True,noise=False,cfg=cfg);env.reset_noise=0.;env.reset()
props=env.gym.get_actor_rigid_body_properties(env.envs[0],env.actor_handles[0]);com=[]
for i in env.feet_indices.cpu().tolist():
    c=props[i].com
    if hasattr(c,'p'):c=c.p
    com.append([c.x,c.y,c.z])
com=np.array(com);states=[]
for k in range(100):
    action=torch.zeros(1,12,device=env.device)
    for name in ['L_elbow_pitch','R_elbow_pitch','L_knee_pitch','R_knee_pitch']:
        action[:,MODEL['policy_joint_names'].index(name)]=.35*math.sin(2*math.pi*.5*k*env.dt)
    env.step(action);states.append(env.rigid_states[0,env.feet_indices].cpu().numpy().copy())
s=np.array(states);q=s[:,:,3:6];w=s[:,:,6:7];world_com=com+2*np.cross(q,np.cross(q,com)+w*com)
fd=(s[2:,:,:3]-s[:-2,:,:3])/(2*env.dt);reported=s[1:-1,:,7:10];offset_v=np.cross(s[1:-1,:,10:13],world_com[1:-1])
sel=slice(20,None)
err={name:float(np.sqrt(np.mean((vel[sel]-fd[sel])**2))) for name,vel in [('reported_direct',reported),('reported_minus_omega_cross_com',reported-offset_v),('reported_plus_omega_cross_com',reported+offset_v)]}
result={'rms_velocity_error_m_s':err,'best':min(err,key=err.get),'local_com_m':com.tolist(),'scope':'fixed root raised 0.1m, slow joint motion, finite-difference body origins'}
atomic_json(Path(sys.argv[2]),result);print('VELOCITY_FRAME',json.dumps(result),flush=True);env.gym.destroy_sim(env.sim)
