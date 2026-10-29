"""Observe age and physical meaning of cached hind touchdown positions."""
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
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(419)
    groups=[('forward',[.105,0.,0.]),('reverse',[-.04,0.,0.]),('lateral',[0.,.04,0.]),
        ('yaw',[0.,0.,.35]),('combined',[.07,.03,.25])]
    per=4;n=len(groups)*per
    cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.reference.file='/intentionally_absent_width_probe.json'
    cfg.env.episode_length_s=9.;cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True;env.reset_noise=.015
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();commands=env.commands16.clone()
    for g,(_,c) in enumerate(groups):commands[g*per:(g+1)*per,:3]=commands.new_tensor(c)
    env.set_commands(commands)
    alive=torch.ones(n,dtype=torch.bool,device=env.device);last=torch.zeros(n,2,device=env.device)
    sums=torch.zeros(n,8,device=env.device)
    with torch.no_grad():
        for step in range(400):
            previous_known=env.hind_width.known.clone()
            _,_,_,done,_=env.step(policy.act_inference(env.get_observations()));s=env.eval_snapshot
            age=(step+1)*env.dt
            contact=(s['force'][:,2:]>1.)&(s['clearance'][:,2:]<=.001)
            fresh=env.actual_steps.landed[:,2:]|(~previous_known&contact)
            last=torch.where(fresh,torch.full_like(last,age),last)
            stale=(age-last>2./commands[:,5:6]).any(1)
            x,y,z,w=s['base_quat'].unbind(1)
            forward=torch.stack((1-2*(y*y+z*z),2*(x*y+w*z)),1)
            forward/=forward.norm(dim=1,keepdim=True).clamp(min=1e-6)
            left=torch.stack((-forward[:,1],forward[:,0]),1)
            instant=((env.foot_contact_point[:,2,:2]-env.foot_contact_point[:,3,:2])*left).sum(1)
            penalty=-s['weighted_rewards']['hind_landing_width']/env.dt
            valid=alive&~done&(age>=1.)
            rows=torch.stack((torch.ones_like(penalty),stale.float(),penalty,penalty*stale,
                s['hind_landing_width'],instant,(instant>=.06).float()*stale* (penalty>1e-5),
                (age-last).max(1).values),1)
            sums+=rows*valid[:,None];alive &= ~done
    rows=[]
    for g,(name,c) in enumerate(groups):
        v=sums[g*per:(g+1)*per].sum(0);count=v[0].clamp(min=1)
        rows.append({'name':name,'command':c,'samples':int(v[0]),'stale_pair_fraction':float(v[1]/count),
            'weighted_width_cost_per_second':float(v[2]/count),'width_cost_fraction_on_stale_pairs':float(v[3]/v[2].clamp(min=1e-8)),
            'mean_cached_width_m':float(v[4]/count),'mean_instantaneous_contact_point_width_m':float(v[5]/count),
            'fraction_penalized_on_stale_pair_despite_instant_width60mm':float(v[6]/count),
            'mean_oldest_record_age_s':float(v[7]/count),'survival_fraction':float(alive[g*per:(g+1)*per].float().mean())})
    atomic_json(a.output,{'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),
        'seed':419,'horizon_s':8.,'envs_per_group':per,'groups':rows,
        'reference_execution_calls':env.reference_execution_calls,'task_acceptance':False,
        'scope':'Original episodes; no intervention. Instantaneous capsule-lowest-point spacing is a different quantity from cached actual touchdowns; both are reported explicitly.'})
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
