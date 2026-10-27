"""Compare actual clearance with FK of executed joint targets during swing."""
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
    groups=[('forward',[.105,0.,0.]),('reverse',[-.04,0.,0.]),('left',[0.,.04,0.]),('turn_left',[0.,0.,.35])]
    per=4;n=len(groups)*per;cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.reference.file='/intentionally_absent_target_clearance_probe.json';cfg.commands.resampling_time=1000.
    cfg.env.episode_length_s=7.;env=create_migration_env(cfg,noise=False);env.capture_eval=True;env.reset_noise=.015
    policy=HistoryActorCritic().to(env.device).eval();policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();commands=env.commands16.clone()
    for i,(_,command) in enumerate(groups):commands[i*per:(i+1)*per,:3]=commands.new_tensor(command)
    env.set_commands(commands);alive=torch.ones(n,dtype=torch.bool,device=env.device)
    sums=torch.zeros(n,4,8,device=env.device);worst=0.
    with torch.no_grad():
        for step in range(300):
            _,_,_,done,_=env.step(policy.act_inference(env.get_observations()));s=env.eval_snapshot
            actual_fk=env.mapping.task_measurements(s['xyz'],s['base_quat'],s['q'],env.dof_names)[3]
            q=s['q'].clone();q[:,env.active]=s['executed_target'];q[:,env.dependent]=-q[:,env.hip]
            target_fk=env.mapping.task_measurements(s['xyz'],s['base_quat'],q,env.dof_names)[3]
            valid=alive&~done&((step+1)*env.dt>=1.);mask=valid[:,None]&(env.gait_swing_weight>.5)
            error=(actual_fk-s['clearance']).abs()
            if valid.any():worst=max(worst,float(error[valid].max()))
            values=torch.stack((torch.ones_like(target_fk),s['clearance'],target_fk,env.gait_target_clearance,
                (target_fk<0).float(),(target_fk<.002).float(),(s['clearance']<.002).float(),
                (env.gait_target_clearance-target_fk).clamp(min=0)),dim=-1)
            sums+=values*mask[:,:,None];alive&=~done
    rows=[]
    for g,(name,command) in enumerate(groups):
        v=sums[g*per:(g+1)*per].sum(0);count=v[:,0]
        def mean(i):return [float(v[f,i]/count[f]) if count[f]>0 else None for f in range(4)]
        rows.append(dict(name=name,command=command,survival_fraction=float(alive[g*per:(g+1)*per].float().mean()),
            swing_samples_per_foot=count.cpu().tolist(),actual_clearance_m=mean(1),target_FK_clearance_m=mean(2),
            clock_target_clearance_m=mean(3),target_FK_below_ground_fraction=mean(4),
            target_FK_below2mm_fraction=mean(5),actual_below2mm_fraction=mean(6),target_FK_shortfall_m=mean(7)))
    assert worst<3e-5,'Actual FK and native capsule clearance disagree'
    atomic_json(a.output,dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),groups=rows,
        actual_FK_native_max_error_m=worst,reference_execution_calls=env.reference_execution_calls,
        scope='Diagnostic only. Target FK is hypothetical geometry at current measured root pose with executed joint targets and exact opposed coupling; not actual clearance or evidence of gait. Original nonterminal samples after1s with commanded swing weight>0.5.',
        task_acceptance=False,horizon_s=6.,seed=419))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
