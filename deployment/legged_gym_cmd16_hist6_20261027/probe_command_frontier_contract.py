"""Check that the frontier changes actual commands consistently, never actions."""
import argparse
from pathlib import Path
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from history_policy import HistoryActorCritic
from command_schema import CommandSpec
from curriculum import command_frontier
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(443);teacher=verify_teacher_manifest(a.teacher)
    cfg=student_config(teacher['run'],'S3',32)
    assert cfg.migration.command_frontier_alpha==1. and cfg.migration.command_ranges['frequency']==[1.,1.]
    cfg.migration.command_frontier_alpha=.25;cfg.migration.moving_start_training=True
    cfg.commands.resampling_time=1000.;cfg.reference.file='/intentionally_absent_frontier_contract.json'
    env=create_migration_env(cfg,noise=False);env.capture_eval=True
    policy=HistoryActorCritic().to(env.device).eval();policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();maximum=0.;mapped=0
    with torch.no_grad():
        for tick in range(64):
            if tick==32:env.reset_idx(torch.tensor([1,6],device=env.device));env.compute_observations()
            expected=command_frontier(env.training_command_goals,.25)
            maximum=max(maximum,float((env.moving_start.target-expected).abs().max()))
            mapped+=int((env.training_command_goals-expected).abs().sum(1).gt(0).sum())
            env.prepare_training_commands();actual=env.commands16.clone()
            maximum=max(maximum,float((env.get_observations()[:,306:322]-CommandSpec.normalize(actual)).abs().max()))
            maximum=max(maximum,float((env.get_privileged_observations()[:,51:67]-CommandSpec.normalize(actual)).abs().max()))
            _,_,_,_,_=env.step(policy.act_inference(env.get_observations()))
            maximum=max(maximum,float((env.eval_snapshot['commands'][:,:3]-actual[:,:3]).abs().max()))
        # Explicit override uses the full requested velocity, even when this
        # diagnostic env has training alpha.25. Production evaluation defaults1.
        full=env.commands16.clone();full[:,:3]=full.new_tensor([0.,.04,.35]);env.set_commands(full)
        env.prepare_training_commands();maximum=max(maximum,float((env.commands16-full).abs().max()))
        env.reset_idx(torch.tensor([0],device=env.device));env.compute_observations()
        maximum=max(maximum,float((env.commands16-full).abs().max()))
    assert maximum<1e-6 and mapped>0 and env.reference_execution_calls==0
    atomic_json(a.output,dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),num_envs=32,
        control_steps=64,training_alpha=.25,production_evaluation_alpha=1.,maximum_command_contract_error=maximum,
        mapped_sample_count=mapped,forced_reset_ids=[1,6],override_survives_reset=True,reference_execution_calls=0,
        actual_S3_frequency_hz=1.,task_acceptance=False))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
