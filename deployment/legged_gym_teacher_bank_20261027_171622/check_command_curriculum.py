"""Real-simulator command timing and fresh-reset regression probe."""
import argparse
import json
from pathlib import Path
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',8)
    cfg.migration.moving_start_training=True
    cfg.reference.file='/intentionally_absent_command_probe.json'
    env=create_migration_env(cfg,noise=False)
    env.reset()
    ids=torch.arange(8,device=env.device)
    target=torch.tensor([[0.,.04,0.]],device=env.device).repeat(8,1)
    env.moving_start.resample(ids,target,torch.zeros(8,device=env.device),torch.zeros(8,device=env.device))
    errors=[]
    for t in (0.,2.,3.,4.,6.):
        env.episode_length_buf[:]=round(t/env.dt)
        history=env.history.data.clone()
        env.prepare_training_commands()
        raw,critic=env.obs_buf.clone(),env.privileged_obs_buf.clone()
        env.compute_observations()
        torch.testing.assert_close(history,env.history.data,rtol=0,atol=0)
        torch.testing.assert_close(raw,env.obs_buf,rtol=0,atol=0)
        torch.testing.assert_close(critic,env.privileged_obs_buf,rtol=0,atol=0)
        errors.append(float((critic-env.privileged_obs_buf).abs().max()))
    torch.testing.assert_close(env.commands16[:,:3],target)
    env.episode_length_buf[:]=900
    env.reset_idx(ids)
    assert (env.moving_start.sample_time==0.).all()
    env.prepare_training_commands()
    assert (env.episode_length_buf==0).all()
    torch.testing.assert_close(env.commands16[:,:3],env.moving_start.start)
    # Explicit evaluation commands bypass all training warm-start changes.
    requested=env.commands16.clone();requested[:,:3]=target
    env.set_commands(requested)
    env.prepare_training_commands()
    torch.testing.assert_close(env.commands16,requested)
    atomic_json(a.output,{'passed':True,'command_timing_errors':errors,
        'fresh_reset_time_zero':True,'evaluation_override_exact':True,
        'reference_execution_calls':env.reference_execution_calls})
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
