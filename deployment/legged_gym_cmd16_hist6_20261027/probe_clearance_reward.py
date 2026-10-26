"""Measure clearance-component sensitivity lost to positive reward clipping.

The counterfactual changes one measured clearance scalar by1mm, holding every
other reward component fixed. It is a reward diagnostic, not a physical action
or evidence of locomotion. No optimizer or reference execution is used.
"""
import argparse
from pathlib import Path
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from history_policy import HistoryActorCritic
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,sha256

POST=('velocity_error','waist_vertical_acceleration','ankle_event_angle','yaw_rate_error',
    'stride_shortfall','stride_variance','hind_landing_width','proximal_acceleration_peak',
    'swing_peak_balance','swing_peak_shortfall')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(419)
    groups=[('forward',[.105,0.,0.]),('reverse',[-.04,0.,0.]),('lateral',[0.,.04,0.]),
        ('yaw',[0.,0.,.35]),('combined',[.07,.03,.25])]
    per=4;n=len(groups)*per
    cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.reference.file='/intentionally_absent_clearance_probe.json'
    cfg.env.episode_length_s=7.;cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True;env.reset_noise=.015
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();commands=env.commands16.clone()
    for g,(_,c) in enumerate(groups):commands[g*per:(g+1)*per,:3]=commands.new_tensor(c)
    env.set_commands(commands)
    alive=torch.ones(n,dtype=torch.bool,device=env.device)
    # eligible samples, fully swallowed samples, raw gain, aggregate gain
    sums=torch.zeros(n,2,4,device=env.device)
    reconstruction_error=0.
    with torch.no_grad():
        for step in range(300):
            _,_,_,done,_=env.step(policy.act_inference(env.get_observations()))
            s=env.eval_snapshot;rewards=s['weighted_rewards']
            after_clip=POST+(('gait_clearance','target_limits') if cfg.migration.unclipped_gait_costs else ())
            base=sum(v for k,v in rewards.items() if k not in after_clip+('termination','total','positive_clip_adjustment'))
            post=sum(v for k,v in rewards.items() if k in after_clip+('termination',))
            reconstructed=base.clamp(min=0)+post
            reconstruction_error=max(reconstruction_error,float((reconstructed-rewards['total']).abs().max()))
            target=env.gait_target_clearance[:,2:]
            actual=s['clearance'][:,2:]
            scale=commands[:,10:11]
            before=((actual-target)/scale).square().clamp(max=9.)
            after=((actual+.001-target)/scale).square().clamp(max=9.)
            delta=(after-before)*env.reward_scales['gait_clearance']/4.
            gain=delta if cfg.migration.unclipped_gait_costs else (base[:,None]+delta).clamp(min=0)-base[:,None].clamp(min=0)
            valid=alive[:,None]&(target>.01)&(actual+.001<target)&(delta>1e-7)&(step*env.dt>1.)
            sums[:,:,0]+=valid;sums[:,:,1]+=(gain<1e-7)&valid
            sums[:,:,2]+=delta*valid;sums[:,:,3]+=gain*valid
            alive &= ~done
    assert reconstruction_error<1e-5
    rows=[]
    for g,(name,c) in enumerate(groups):
        row=sums[g*per:(g+1)*per].sum(0)
        rows.append({'name':name,'command':c,'survival_fraction':float(alive[g*per:(g+1)*per].float().mean()),
            'hind_feet':[{'foot':foot+2,'samples':int(v[0]),'fully_swallowed_fraction':float(v[1]/v[0].clamp(min=1)),
                'unclipped_reward_gain':float(v[2]/v[0].clamp(min=1)),
                'actual_aggregate_gain':float(v[3]/v[0].clamp(min=1))} for foot,v in enumerate(row)]})
    atomic_json(a.output,{'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),
        'seed':419,'horizon_s':6.,'envs_per_group':per,'groups':rows,'reconstruction_max_error':reconstruction_error,
        'reference_execution_calls':env.reference_execution_calls,'task_acceptance':False,
        'unclipped_gait_costs':cfg.migration.unclipped_gait_costs,
        'scope':'One-component1mm clearance counterfactual, holding other rewards fixed; no physical intervention'})
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
