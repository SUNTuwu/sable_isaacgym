"""Compare exploration noise in one physics batch; not task acceptance."""
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
    a=p.parse_args()
    torch.manual_seed(419)
    groups=[(name,command,std) for std in (.02,.04,.06) for name,command in
        [('forward',[.105,0.,0.]),('reverse',[-.04,0.,0.]),('lateral',[0.,.04,0.]),('yaw',[0.,0.,.35])]]
    per=8;n=len(groups)*per
    cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.reference.file='/intentionally_absent_exploration_probe.json'
    cfg.env.episode_length_s=11.;cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True;env.reset_noise=.015
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();commands=env.commands16.clone();stds=torch.zeros(n,1,device=env.device)
    for g,(_,c,std) in enumerate(groups):
        commands[g*per:(g+1)*per,:3]=commands.new_tensor(c);stds[g*per:(g+1)*per]=std
    env.set_commands(commands)
    alive=torch.ones(n,dtype=torch.bool,device=env.device)
    duration=torch.zeros(n,device=env.device);reward_sum=duration.clone()
    peaks=torch.zeros(n,4,device=env.device);costs={}
    with torch.no_grad():
        for _ in range(500):
            action=policy.act_inference(env.get_observations())+torch.randn(n,15,device=env.device)*stds*policy.execution_mask
            _,_,reward,done,_=env.step(action);snapshot=env.eval_snapshot
            duration+=alive*env.dt;reward_sum+=reward*alive
            peaks=torch.maximum(peaks,snapshot['clearance']*alive[:,None])
            for name,value in snapshot['weighted_rewards'].items():
                if name not in costs:costs[name]=torch.zeros(n,device=env.device)
                costs[name]+=value*alive
            alive &= ~done
    rows=[]
    for g,(name,command,std) in enumerate(groups):
        sl=slice(g*per,(g+1)*per)
        rows.append({'command':command,'name':name,'std':std,'survival_fraction':float(alive[sl].float().mean()),
            'mean_original_duration_s':float(duration[sl].mean()),'mean_peak_per_foot_m':peaks[sl].mean(0).cpu().tolist(),
            'weighted_reward_per_second':{k:float(v[sl].sum()/duration[sl].sum().clamp(min=.02)) for k,v in costs.items()}})
    atomic_json(a.output,{'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),
        'seed':419,'horizon_s':10.,'envs_per_group':per,'groups':rows,
        'reference_execution_calls':env.reference_execution_calls,'task_acceptance':False,
        'scope':'Stochastic exploration diagnostic; original episodes only, no learning, direct starts'})
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
