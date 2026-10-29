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
from step_events import StepEvents


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hind-only',action='store_true',help='Raise only six hind pitch/knee/ankle noise dimensions; others remain .04')
    p.add_argument('--stds',type=float,nargs='+',default=[.02,.04,.06])
    p.add_argument('--swing-only',action='store_true',help='Smoothly raise hind noise only during its commanded swing')
    p.add_argument('--scenarios',nargs='+',choices=['forward','reverse','lateral','yaw'],default=['forward','reverse','lateral','yaw'])
    a=p.parse_args()
    if a.swing_only and not a.hind_only:p.error('--swing-only requires --hind-only')
    torch.manual_seed(419)
    groups=[(name,command,std) for std in a.stds for name,command in
        [('forward',[.105,0.,0.]),('reverse',[-.04,0.,0.]),('lateral',[0.,.04,0.]),('yaw',[0.,0.,.35])] if name in a.scenarios]
    per=8;n=len(groups)*per
    cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.reference.file='/intentionally_absent_exploration_probe.json'
    cfg.env.episode_length_s=11.;cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True;env.reset_noise=.015
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();commands=env.commands16.clone();stds=torch.full((n,15),.04,device=env.device)
    for g,(_,c,std) in enumerate(groups):
        commands[g*per:(g+1)*per,:3]=commands.new_tensor(c)
        if a.hind_only:stds[g*per:(g+1)*per,[5,6,10,11,12,13]]=std
        else:stds[g*per:(g+1)*per]=std
    env.set_commands(commands)
    alive=torch.ones(n,dtype=torch.bool,device=env.device)
    duration=torch.zeros(n,device=env.device);reward_sum=duration.clone()
    peaks=torch.zeros(n,4,device=env.device);costs={}
    steps=StepEvents(n,env.device,env.dt);counts=torch.zeros(n,4,device=env.device)
    with torch.no_grad():
        for tick in range(500):
            raw=env.get_observations();actual_stds=stds
            if a.swing_only:
                gate=torch.zeros_like(stds)
                gate[:,[5,6,12]]=(-raw[:,324:325]).clamp(0,1)
                gate[:,[10,11,13]]=(-raw[:,325:326]).clamp(0,1)
                actual_stds=.04+(stds-.04)*gate
            action=policy.act_inference(raw)+torch.randn(n,15,device=env.device)*actual_stds*policy.execution_mask
            _,_,reward,done,_=env.step(action);snapshot=env.eval_snapshot
            duration+=alive*env.dt;reward_sum+=reward*alive
            peaks=torch.maximum(peaks,snapshot['clearance']*alive[:,None])
            landed=steps.update(snapshot['foot_centers'][:,:,:2],snapshot['force']>1.,snapshot['clearance'],
                torch.full((n,),(tick+1)*env.dt,device=env.device),torch.ones(n,dtype=torch.bool,device=env.device))
            counts+=landed*alive[:,None]
            for name,value in snapshot['weighted_rewards'].items():
                if name not in costs:costs[name]=torch.zeros(n,device=env.device)
                costs[name]+=value*alive
            alive &= ~done
    rows=[]
    for g,(name,command,std) in enumerate(groups):
        sl=slice(g*per,(g+1)*per)
        rows.append({'command':command,'name':name,'std':std,'survival_fraction':float(alive[sl].float().mean()),
            'mean_original_duration_s':float(duration[sl].mean()),'mean_peak_per_foot_m':peaks[sl].mean(0).cpu().tolist(),
            'actual_completed_steps_per_foot':counts[sl].sum(0).cpu().tolist(),
            'weighted_reward_per_second':{k:float(v[sl].sum()/duration[sl].sum().clamp(min=.02)) for k,v in costs.items()}})
    atomic_json(a.output,{'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),
        'seed':419,'horizon_s':10.,'envs_per_group':per,'groups':rows,
        'reference_execution_calls':env.reference_execution_calls,'task_acceptance':False,
        'noise_scope':'six hind pitch/knee/ankle dimensions, other dimensions .04' if a.hind_only else 'all executed dimensions',
        'swing_phase_gated':a.swing_only,
        'scope':'Stochastic exploration diagnostic; original episodes only, no learning, direct starts'})
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
