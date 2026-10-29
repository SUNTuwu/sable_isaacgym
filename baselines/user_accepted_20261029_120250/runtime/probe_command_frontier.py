"""Measure actual walking along command-space paths from the learned forward gait.

Short diagnostic only. Full S3 acceptance remains at the original endpoints.
No policy, controller, reward or curriculum is changed by this probe.
"""
import argparse
from pathlib import Path
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from history_policy import HistoryActorCritic
from model_mapping import heading_velocity,quat_apply
from step_events import StepEvents
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--alphas',type=float,nargs='+',default=[.1,.25,.5,.75,1.])
    p.add_argument('--full-families',action='store_true')
    a=p.parse_args();torch.manual_seed(439);per=4;horizon=a.seconds
    assert horizon>1. and all(0<=x<=1 for x in a.alphas)
    groups=[('forward_control',0.,[.105,0.,0.]),('combined_control',1.,[.07,.03,.25])]
    families=[('reverse',[-.04,0.,0.]),('left',[0.,.04,0.]),('turn_left',[0.,0.,.35])]
    if a.full_families:families.extend([('right',[0.,-.04,0.]),('turn_right',[0.,0.,-.35])])
    for name,goal in families:
        for alpha in a.alphas:
            command=[(1-alpha)*.105+alpha*goal[0],alpha*goal[1],alpha*goal[2]]
            groups.append((name,alpha,command))
    n=per*len(groups);cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.reference.file='/intentionally_absent_command_frontier_probe.json'
    cfg.commands.resampling_time=1000.;cfg.env.episode_length_s=horizon+1.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True;env.reset_noise=.015
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();commands=env.commands16.clone()
    for i,(_,_,c) in enumerate(groups):commands[i*per:(i+1)*per,:3]=commands.new_tensor(c)
    env.set_commands(commands);alive=torch.ones(n,device=env.device,dtype=torch.bool)
    sums=torch.zeros(n,7,device=env.device);steps=torch.zeros(n,4,device=env.device)
    last=torch.full_like(steps,-1.);interval_sum=torch.zeros_like(steps);interval_count=torch.zeros_like(steps)
    meter=StepEvents(n,env.device,env.dt);clearance=[];valid_rows=[]
    with torch.no_grad():
        for tick in range(round(horizon/env.dt)):
            _,_,_,done,_=env.step(policy.act_inference(env.get_observations()));s=env.eval_snapshot;t=(tick+1)*env.dt
            valid=alive&~done&(t>=1.)
            xy,yaw=heading_velocity(s['base_quat'],quat_apply(s['base_quat'],s['velocity_local']),
                quat_apply(s['base_quat'],s['angular_velocity_local']))
            actual=torch.cat((xy,yaw[:,None]),1);error=(actual-commands[:,:3]).square()
            sums+=torch.cat((error,actual,torch.ones(n,1,device=env.device)),1)*valid[:,None]
            moving=(commands[:,:2].norm(dim=1)>.01)|(commands[:,2].abs()>.08)
            landed=meter.update(s['foot_centers'][:,:,:2],s['force']>1.,s['clearance'],
                torch.full((n,),t,device=env.device),moving)&valid[:,None]
            intervals=landed&(last>=1.);interval_sum+=(t-last)*intervals;interval_count+=intervals
            last[landed]=t;steps+=landed;clearance.append(s['clearance']);valid_rows.append(valid);alive&=~done
    heights=torch.stack(clearance);valids=torch.stack(valid_rows);rows=[]
    for i,(name,alpha,command) in enumerate(groups):
        sl=slice(i*per,(i+1)*per);total=sums[sl].sum(0);count=float(total[-1]);den=interval_sum[sl].sum(0);cnt=interval_count[sl].sum(0)
        samples=heights[:,sl][valids[:,sl]]
        rows.append(dict(family=name,alpha=alpha,actual_command=command,survival_fraction=float(alive[sl].float().mean()),
            xy_rmse_m_s=float((total[:2].sum()/count).sqrt())if count else None,
            yaw_rmse_rad_s=float((total[2]/count).sqrt())if count else None,
            mean_actual=(total[3:6]/count).cpu().tolist()if count else None,steps_per_foot=steps[sl].sum(0).cpu().tolist(),
            minimum_steps_per_foot=max(2,int((horizon-2)*.5)),
            four_foot_stepping_fraction=float((steps[sl]>=max(2,int((horizon-2)*.5))).all(1).float().mean()),
            actual_frequency_from_mean_interval_hz=[float(cnt[f]/den[f])if cnt[f]>0 else None for f in range(4)],
            interval_count_per_foot=cnt.cpu().tolist(),clearance_p95_per_foot_m=torch.quantile(samples,.95,dim=0).cpu().tolist()if len(samples)else None))
    atomic_json(a.output,dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),seed=439,
        horizon_s=horizon,environments_per_group=per,groups=rows,reference_execution_calls=env.reference_execution_calls,
        task_acceptance=False,training_modified=False,scope='%gs direct-start diagnostic, original nonterminal samples after1s. Actual2mm/60ms/5mm flight-and-landing events. Alpha1 is full desired command; smaller alpha is an intermediate training possibility only, never a relaxed acceptance endpoint.'%horizon))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
