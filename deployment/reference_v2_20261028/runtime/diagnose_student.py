"""Measure student target error on live teacher trajectories before blaming learning."""
import argparse,json
from pathlib import Path
from isaacgym import gymapi
import torch,numpy as np
from migration_env import create_migration_env
from migration_config import student_config
from history_policy import HistoryActorCritic
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    teacher=verify_teacher_manifest(a.teacher);torch.manual_seed(947);np.random.seed(947)
    cfg=student_config(teacher['run'],num_envs=32);cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.reset_noise=.015;env.fixed_command=.105
    policy=HistoryActorCritic().to(env.device).eval();cp=torch.load(str(a.checkpoint),map_location=env.device);policy.load_state_dict(cp['model_state_dict'],strict=True)
    env.attach_teacher(teacher['checkpoint'],teacher['sha256'],0.);obs,_=env.reset();rows=[]
    with torch.no_grad():
        for k in range(200):
            action=policy.act_inference(obs);previous=obs[:,36:51].clone()
            obs,_,_,done,_=env.step(action);target=env.teacher_target_label
            rows.append({'t':(k+1)*env.dt,'student_target_rmse_rad':float(((action[:,:14]-target[:,:14])*.35).square().mean().sqrt()),
                         'copy_previous_rmse_rad':float(((previous[:,:14]-target[:,:14])*.35).square().mean().sqrt()),
                         'max_error_rad':float(((action-target)*.35).abs().max()),'resets':int(done.sum())})
    atomic_json(a.output,{'checkpoint':str(a.checkpoint),'sha256':sha256(a.checkpoint),'iteration':cp['iter'],
                         'mean_live_teacher_target_rmse_rad':float(np.mean([r['student_target_rmse_rad'] for r in rows])),
                         'rows':rows,'scope':'Student predictions on32 live teacher-driven trajectories, not independent student survival'})
    print(json.dumps({'mean_error':np.mean([r['student_target_rmse_rad'] for r in rows]),'first':rows[:5],'last':rows[-5:]}),flush=True);env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
