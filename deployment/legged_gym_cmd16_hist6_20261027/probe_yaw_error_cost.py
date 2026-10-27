"""Verify physical yaw cost/positive tracking and aggregation in real GPU steps."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from sable_env import SableRobot
from history_policy import HistoryActorCritic
from teacher_adapter import verify_teacher_manifest
from model_mapping import heading_velocity,quat_apply
from run_io import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(443);teacher=verify_teacher_manifest(a.teacher)
    cfg=student_config(teacher['run'],'S3',8)
    cfg.migration.yaw_error_scale_rad_s=.35  # Explicit rejected-global experiment fixture.
    assert cfg.migration.yaw_error_scale_rad_s==.35
    assert student_config(teacher['run'],'S2',8).migration.yaw_error_scale_rad_s==1.
    cfg.reference.file='/intentionally_absent_yaw_cost_probe.json';cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();command=env.commands16.clone()
    command[:,:3]=command.new_tensor([[.105,0,0],[0,0,0],[.065625,0,.13125],[.065625,0,-.13125],
        [0,0,.35],[0,0,-.35],[0,.04,0],[-.04,0,0]])
    env.set_commands(command);cost_error=positive_error=accounting_error=0.
    with torch.no_grad():
        for tick in range(64):
            if tick==32:
                env.reset_idx(torch.tensor([2,5],device=env.device));env.compute_observations()
            _,_,reward,_,_=env.step(policy.act_inference(env.get_observations()));s=env.eval_snapshot
            _,yaw=heading_velocity(s['base_quat'],quat_apply(s['base_quat'],s['velocity_local']),
                quat_apply(s['base_quat'],s['angular_velocity_local']))
            error=yaw-command[:,2]
            expected_cost=-2*env.dt*(error/.35).square()
            expected_positive=3*env.dt*torch.exp(-error.square()/.25)
            cost_error=max(cost_error,float((s['weighted_rewards']['yaw_rate_error']-expected_cost).abs().max()))
            positive_error=max(positive_error,float((s['weighted_rewards']['tracking_ang_vel']-expected_positive).abs().max()))
            accounting_error=max(accounting_error,float((torch.stack([v for k,v in s['weighted_rewards'].items()if k!='total']).sum(0)-reward).abs().max()))
            assert torch.isfinite(reward).all() and torch.equal(env.executed_target,env.student_target)
    # Force a negative ordinary reward aggregate through the actual aggregator:
    # the yaw cost must remain after positive clipping, not disappear with it.
    probe=SimpleNamespace(device=env.device,rew_buf=torch.zeros(8,device=env.device),
        reward_names=['orientation','yaw_rate_error'],
        reward_functions=[lambda:torch.ones(8,device=env.device),lambda:torch.ones(8,device=env.device)],
        reward_scales={'orientation':-10*env.dt,'yaw_rate_error':-2*env.dt},
        episode_sums={n:torch.zeros(8,device=env.device)for n in ('orientation','yaw_rate_error')},cfg=cfg)
    SableRobot.compute_reward(probe)
    torch.testing.assert_close(probe.rew_buf,torch.full((8,),-2*env.dt,device=env.device))
    assert max(cost_error,positive_error,accounting_error)<2e-6
    assert env.reference_execution_calls==0
    atomic_json(a.output,dict(passed=True,checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),
        num_envs=8,control_steps=64,forced_reset_at_step32=[2,5],yaw_error_scale_rad_s=.35,
        maximum_cost_error=cost_error,maximum_positive_tracking_error=positive_error,
        maximum_reward_accounting_error=accounting_error,negative_cost_survives_positive_clip=True,
        S2_cost_scale=1.,positive_tracking_gaussian_unchanged=True,reference_execution_calls=0,
        executed_target_equal_to_student=True,task_acceptance=False))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
