"""Verify reward timing/reset against real pre-reset simulator snapshots."""
import argparse
from pathlib import Path
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from history_policy import HistoryActorCritic
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(419)
    cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',8)
    cfg.reference.file='/intentionally_absent_foot_motion_check.json';cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();command=env.commands16.clone();command[:,:3]=command.new_tensor([[.105,0,0],[-.04,0,0],[0,.04,0],[0,0,.35]]*2)
    env.set_commands(command);previous=torch.zeros(8,4,2,device=env.device)
    known=torch.zeros(8,dtype=torch.bool,device=env.device);worst_velocity=worst_reward=0.
    reset_samples=0
    with torch.no_grad():
        for step in range(64):
            if step==32:
                ids=torch.tensor([0,3],device=env.device);env.reset_idx(ids);env.compute_observations();known[ids]=False
            _,_,_,done,_=env.step(policy.act_inference(env.get_observations()));s=env.eval_snapshot
            assert torch.equal(s['command_foot_velocity_valid'],known)
            expected=(s['foot_centers'][:,:,:2]-previous)/env.dt*known[:,None,None]
            worst_velocity=max(worst_velocity,float((expected-s['command_foot_velocity_actual_m_s']).abs().max()))
            expected_reward=-2*env.dt*s['command_foot_velocity_cost_per_foot'].mean(1)
            worst_reward=max(worst_reward,float((expected_reward-s['weighted_rewards']['command_foot_velocity']).abs().max()))
            assert (s['command_foot_velocity_cost_per_foot'][~known]==0).all()
            reset_samples+=int((~known).sum());previous=s['foot_centers'][:,:,:2].clone();known=~done
    assert worst_velocity<1e-6 and worst_reward<1e-7
    assert env.reference_execution_calls==0
    atomic_json(a.output,dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),
        real_control_steps=64,num_envs=8,manual_reset_environments=[0,3],reset_samples_excluded=reset_samples,
        maximum_velocity_error_m_s=worst_velocity,maximum_weighted_reward_error=worst_reward,
        reference_execution_calls=0,scope='Physical snapshot timing and reset/aggregation check; not task acceptance'))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
