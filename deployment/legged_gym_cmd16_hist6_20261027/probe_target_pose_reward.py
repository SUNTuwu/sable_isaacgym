"""Check target-pose reward timing and scalar accounting in real GPU physics."""
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
    a=p.parse_args();torch.manual_seed(423)
    cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',8)
    cfg.reference.file='/intentionally_absent_target_pose_reward.json';cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();commands=env.commands16.clone()
    commands[:,:3]=commands.new_tensor([[.105,0,0],[-.04,0,0],[0,.04,0],[0,0,.35],
        [0,0,0],[-.04,0,0],[0,-.04,0],[0,0,-.35]])
    env.set_commands(commands);fk_error=cost_error=accounting_error=0.;rows=[]
    with torch.no_grad():
        for tick in range(96):
            if tick==48:
                env.reset_idx(torch.tensor([1,6],device=env.device));env.compute_observations()
            _,_,reward,_,_=env.step(policy.act_inference(env.get_observations()));s=env.eval_snapshot
            before=s['q'].clone();q=s['q'].clone();q[:,env.active]=s['executed_target'];q[:,env.dependent]=-q[:,env.hip]
            expected=env.mapping.task_measurements(s['xyz'],s['base_quat'],q,env.dof_names)[3]
            fk_error=max(fk_error,float((s['target_pose_clearance_m']-expected).abs().max()))
            manual=((env.gait_target_clearance-expected).clamp(min=0)/commands[:,10:11]).square().clamp(max=9.)*env.gait_swing_weight
            cost_error=max(cost_error,float((manual-s['target_pose_clearance_cost_per_foot']).abs().max()))
            weighted=manual.mean(1)*(-8.)*env.dt
            accounting_error=max(accounting_error,float((weighted-s['weighted_rewards']['target_pose_clearance']).abs().max()))
            components=[v for k,v in s['weighted_rewards'].items() if k!='total']
            accounting_error=max(accounting_error,float((torch.stack(components).sum(0)-reward).abs().max()))
            assert torch.equal(before,s['q'])
            assert torch.equal(env.executed_target,env.student_target)
            assert torch.isfinite(reward).all()
            assert s['target_pose_clearance_cost_per_foot'][4].abs().max()==0
            rows.append(weighted.cpu().tolist())
    assert fk_error<2e-6 and cost_error<2e-6 and accounting_error<2e-6
    assert env.reference_execution_calls==0
    s2=student_config(verify_teacher_manifest(a.teacher)['run'],'S2',8)
    assert s2.rewards.scales.target_pose_clearance==0.
    atomic_json(a.output,dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),seed=423,
        num_envs=8,control_steps=96,forced_reset_at_step48=[1,6],FK_error_m=fk_error,
        cost_error=cost_error,reward_accounting_error=accounting_error,reference_execution_calls=0,
        executed_target_equal_to_student=True,S2_weight=0.,weighted_mean_per_environment_per_second=
        (torch.tensor(rows).mean(0)/env.dt).tolist(),task_acceptance=False))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
