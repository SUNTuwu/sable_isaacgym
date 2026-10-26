"""Validate the requested exponential posture cost and actual reward aggregation."""
from pathlib import Path
import math,json
from isaacgym import gymapi
import torch
from preflight import create_env
from run_io import atomic_json,sha256
here=Path(__file__).resolve().parent
env=create_env(5,fixed=True,noise=False)
env.reset_noise=0.;env.fixed_command=0.;env.reset()
indices=env.hip_pitch_indices
angles=torch.tensor([0.,5.,10.,20.,38.],device=env.device)
env.dof_pos[:]=env.default_dof_pos
env.dof_pos[:,indices]=env.default_dof_pos[:,indices]+angles[:,None]*math.pi/180
cost=env._reward_hip_stand_exp().clone()
slopes=(cost[1:]-cost[:-1])/(angles[1:]-angles[:-1])
checks={'zero_at_standard_stand':bool(cost[0]==0),'zero_through_five_degree_tolerance':bool((cost[:2].abs()<1e-8).all()),'strictly_increasing_outside_tolerance':bool((cost[2:]>cost[1:-1]).all()),
        'increasing_slope':bool((slopes[1:]>slopes[:-1]).all()),'finite_in_range':bool(torch.isfinite(cost).all())}
env.dof_pos[:,indices]=env.default_dof_pos[:,indices]-angles[:,None]*math.pi/180
checks['opposite_deviation_same_cost']=bool(torch.allclose(cost,env._reward_hip_stand_exp(),atol=1e-6))
env.dof_pos[:]=env.default_dof_pos
env.dof_pos[:,indices[0]]+=angles*math.pi/180
left=env._reward_hip_stand_exp().clone()
env.dof_pos[:]=env.default_dof_pos
env.dof_pos[:,indices[1]]+=angles*math.pi/180
checks['left_right_symmetric']=bool(torch.allclose(left,env._reward_hip_stand_exp()))
env.dof_pos[:]=env.default_dof_pos
for name in ['L_elbow_pitch','R_elbow_pitch','L_knee_pitch','R_knee_pitch']:
    env.dof_pos[:,env.dof_names.index(name)]+=.2
checks['elbows_knees_not_locked_by_this_term']=bool((env._reward_hip_stand_exp()==0).all())
env.dof_pos[:,indices]=env.default_dof_pos[:,indices]+angles[:,None]*math.pi/180
# A deliberately negative ordinary reward should clip to zero, while the
# posture penalty and death penalty remain, each included exactly once.
env.reward_names=['negative_base','hip_stand_exp']
env.reward_functions=[lambda:torch.full((5,),-1.,device=env.device),env._reward_hip_stand_exp]
scale=env.reward_scales['hip_stand_exp'];death=env.reward_scales['termination']
env.reward_scales={'negative_base':1.,'hip_stand_exp':scale,'termination':death}
env.episode_sums={k:torch.zeros(5,device=env.device) for k in env.reward_scales}
env.true_fall[:]=False;env.true_fall[-1]=True
env.compute_reward();expected=cost*scale+env.true_fall.float()*death
checks['survives_positive_reward_clip']=bool(torch.allclose(env.rew_buf,expected,atol=1e-6)) and bool((env.rew_buf[2:]<0).all()) and bool((env.rew_buf[:2].abs()<1e-8).all())
checks['episode_posture_sum_once']=bool(torch.allclose(env.episode_sums['hip_stand_exp'],cost*scale))
checks['death_cost_preserved']=bool(torch.allclose(env.episode_sums['termination'],env.true_fall.float()*death))
env.reward_names.append('swing_peak_balance');env.reward_functions.append(env._reward_swing_peak_balance)
env.reward_scales['swing_peak_balance']=-.5*env.dt
env.episode_sums['swing_peak_balance']=torch.zeros(5,device=env.device)
env.swing_peak_balance_cost[:]=torch.tensor([0.,.25,1.,2.,3.],device=env.device)
env.compute_reward()
expected_with_balance=expected+env.swing_peak_balance_cost*(-.5*env.dt)
checks['peak_balance_survives_clip_and_preserves_death']=bool(torch.allclose(env.rew_buf,expected_with_balance,atol=1e-6))
checks['peak_balance_sum_once']=bool(torch.allclose(env.episode_sums['swing_peak_balance'],env.swing_peak_balance_cost*(-.5*env.dt)))
report={'passed':all(checks.values()),'checks':checks,'reference_deg':(env.default_dof_pos[:,indices]*180/math.pi).cpu().tolist(),
        'deviation_deg':angles.cpu().tolist(),'cost_per_second':(cost*(-scale)/env.dt).cpu().tolist(),
        'aggregated_control_step_rewards':env.rew_buf.cpu().tolist(),
        'source_sha256':{n:sha256(here/n) for n in ['sable_env.py','sable_config.py','swing_peak_balance.py','check_hip_stand_reward.py','assets/physical_manifest.json']}}
atomic_json(here/'verification/hip_stand_reward.json',report)
print(json.dumps(report));env.gym.destroy_sim(env.sim)
if not report['passed']:raise SystemExit(2)
