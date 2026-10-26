from pathlib import Path
root=Path('/root/gpufree-data/projects/sable-isaacgym');d=root/'deployment/legged_gym_powered_ankles_20261026'
def edit(name,changes):
 p=d/name;s=p.read_text()
 for a,b in changes:
  assert a in s,(name,a);s=s.replace(a,b)
 p.write_text(s)
# Keep all prior body masses; new motor housings +40g, longer capsules +1.2375g.
edit('build_asset.py',[("if key in ['hand','foot']:mass=.00165","if key in ['hand','foot']:mass=.00165*(1.375 if key=='foot' else 1.)"),('remaining=1.5-fixed_mass','remaining=1.5412375-fixed_mass'),("'L_ankle_fixed','R_ankle_fixed'","'L_ankle_pitch','R_ankle_pitch'"),("'physical_independent_actuators':13","'physical_independent_actuators':15"),("'policy_joint_names':[r['name'] for r in MODEL['joints'] if r['actuated'] and r['name']!='spine_pitch']","'policy_joint_names':[r['name'] for r in MODEL['joints'] if r['actuated'] and r['name']!='spine_pitch' and 'ankle' not in r['name']]+['L_ankle_pitch','R_ankle_pitch']"),('1.5 kg provisional total mass; fresh box-based inertia approximation','1.5412375 kg provisional total: prior body mass unchanged, two 10g ankle blocks replaced by 30g motors and rear pad volume scaled 1.375x'),('four unchanged capsules','four capsules (new rear feet 40mm)')])
edit('sable_config.py',[('num_observations=48;num_actions=12','num_observations=54;num_actions=14'),("'knee_pitch':5.","'knee_pitch':5.,'ankle_pitch':5."),("'knee_pitch':.025","'knee_pitch':.025,'ankle_pitch':.025"),('target_limits=-2.;','target_limits=-2.;forward_speed=2.;')])
edit('sable_env.py',[('assert self.num_dof==14','assert self.num_dof==16'),('    def _reward_stride_progress(self):', '''    def _reward_forward_speed(self):
        # Positive body-forward velocity, capped at the declared course top speed.
        # Suppress parking, lateral slides, yaw spins and terminal falls.
        forward=self.base_lin_vel[:,0].clamp(0.,.12)/.12
        straight=torch.exp(-self.base_lin_vel[:,1].square()/.0025-self.base_ang_vel[:,2].square()/.25)
        return forward*straight*(self.commands[:,0]>.02)*(~self.true_fall)

    def _reward_stride_progress(self):''')])
edit('preflight.py', [('abs(env.import_mass-1.5)<.02',"abs(env.import_mass-MODEL['total_mass_kg'])<.002")])
for name in ['guided_reference.py','gait_support.py']:
 edit(name,[('L_ankle_fixed','L_ankle_pitch'),('R_ankle_fixed','R_ankle_pitch')])
# New IK table uses new geometry with ankle held at initial70deg; PPO controls all14 residuals.
# Start with known command range; add bounded positive speed incentive instead of unlimited acceleration.
edit('policy_symmetry.py',[("'R_shoulder_roll','R_shoulder_pitch','R_elbow_pitch','R_hip_pitch','R_knee_pitch']","'R_shoulder_roll','R_shoulder_pitch','R_elbow_pitch','R_hip_pitch','R_knee_pitch','L_ankle_pitch','R_ankle_pitch']"),('ACTION_INDEX=[0,1,7,8,9,10,11,2,3,4,5,6]','ACTION_INDEX=[0,1,7,8,9,10,11,2,3,4,5,6,13,12]'),('ACTION_SIGN=[-1,1,-1,1,1,1,1,-1,1,1,1,1]','ACTION_SIGN=[-1,1,-1,1,1,1,1,-1,1,1,1,1,1,1]'),('assert obs_dim in [48,50,52]','assert obs_dim in [54,56,58]'),('for start in [12,24,36]:','for start in [12,26,40]:'),('start:start+12','start:start+14'),('if obs_dim>=50:sign[48:50]','if obs_dim>=56:sign[54:56]')])
edit('equivariant_policy.py',[('num_actor_obs==num_critic_obs==52 and num_actions==12','num_actor_obs==num_critic_obs==58 and num_actions==14')])
edit('train.py', [('verified 52D guided','verified 58D guided'),('cfg.env.num_observations=50','cfg.env.num_observations=56'),('cfg.env.num_observations=52','cfg.env.num_observations=58'),("'actuators':13,'policy_actions':12","'actuators':15,'policy_actions':14"),('cfg.rewards.scales.tracking_lin_vel=6.','cfg.rewards.scales.tracking_lin_vel=8.'),('''            for key,value in previous['model_state_dict'].items():
                if state[key].shape==value.shape:state[key]=value
                elif key in ['actor.0.weight','critic.0.weight'] and (value.shape[1],state[key].shape[1]) in [(48,50),(50,52)]:
                    state[key].zero_();state[key][:,:value.shape[1]]=value
                else:raise ValueError('Unsupported policy transfer shape: '+key)''','''            from powered_transfer import transfer_state
            state=transfer_state(previous['model_state_dict'],state)''')])
edit('evaluate.py',[('env.num_obs,env.num_obs,12','env.num_obs,env.num_obs,env.num_actions'),('torch.zeros((a.num_envs,12)','torch.zeros((a.num_envs,env.num_actions)'),('env.commands[:,0]=sequence[index];env.commands[:,1:]=0.;','env.commands[:,0]=sequence[index];env.commands[:,1]=0.;env.commands[:,3]=0.;')])
# Visual inference must use the new policy dimensions as well.
for name in ['record_side_video.py','diagnose_policy.py']:
 p=d/name;s=p.read_text().replace('env.num_obs,env.num_obs,12','env.num_obs,env.num_obs,env.num_actions').replace('L_ankle_fixed','L_ankle_pitch').replace('R_ankle_fixed','R_ankle_pitch');p.write_text(s)
(d/'powered_transfer.py').write_text('''"""Explicit joint-name-preserving 52/12 -> 58/14 transfer, new inputs/outputs zero."""
import torch
OLD_TO_NEW=list(range(12))+list(range(12,24))+list(range(26,38))+list(range(40,52))+list(range(54,58))
def transfer_state(old,new):
    for key,value in old.items():
        if new[key].shape==value.shape:new[key]=value.clone()
        elif key in ['actor.0.weight','critic.0.weight'] and value.shape[1]==52 and new[key].shape[1]==58:
            new[key].zero_();new[key][:,OLD_TO_NEW]=value
        elif key in ['actor.6.weight','actor.6.bias','std'] and value.shape[0]==12 and new[key].shape[0]==14:
            new[key].zero_();new[key][:12]=value
        else:raise ValueError('Unsupported transfer: '+key)
    return new
''')
print('PREPARED',d)
