"""Measure a shadow command-directed foot-velocity cost without changing rewards."""
import argparse
from pathlib import Path
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from history_policy import HistoryActorCritic
from teacher_adapter import verify_teacher_manifest
from command_foot_motion import desired_foot_velocity,command_velocity_at_feet,FootCenterVelocity,foot_velocity_cost
from run_io import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(419)
    groups=[('forward09',[.09,0.,0.]),('forward105',[.105,0.,0.]),('forward12',[.12,0.,0.]),
        ('stop',[0.,0.,0.]),('reverse',[-.04,0.,0.]),('left',[0.,.04,0.]),
        ('turn_left',[0.,0.,.35]),('combined',[.07,.03,.25])]
    per=4;n=len(groups)*per;cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.reference.file='/intentionally_absent_foot_motion_probe.json'
    cfg.env.episode_length_s=7.;cfg.commands.resampling_time=1000.
    env=create_migration_env(cfg,noise=False);env.capture_eval=True;env.reset_noise=.015
    policy=HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(torch.load(str(a.checkpoint),map_location=env.device)['model_state_dict'],strict=True)
    env.reset();commands=env.commands16.clone()
    for g,(_,c) in enumerate(groups):commands[g*per:(g+1)*per,:3]=commands.new_tensor(c)
    env.set_commands(commands);meter=FootCenterVelocity(n,env.device,env.dt)
    alive=torch.ones(n,dtype=torch.bool,device=env.device);count=torch.zeros(n,device=env.device)
    costs=torch.zeros(n,4,device=env.device);aligned=torch.zeros_like(costs);targets=torch.zeros_like(costs)
    swing_costs=torch.zeros_like(costs);normalized_costs=torch.zeros_like(costs);dense_gain=torch.zeros_like(costs)
    rewards={}
    with torch.no_grad():
        for tick in range(300):
            _,_,_,done,_=env.step(policy.act_inference(env.get_observations()));s=env.eval_snapshot
            actual=meter.update(s['foot_centers'][:,:,:2]);x,y,z,w=s['base_quat'].unbind(1)
            forward=torch.stack((1-2*(y*y+z*z),2*(x*y+w*z)),1)
            forward/=forward.norm(dim=1,keepdim=True).clamp(min=1e-6)
            phase=torch.remainder(env.clock.local_phases(commands)-commands[:,5:6]*env.dt*.5,1.)
            target=desired_foot_velocity(commands[:,:3],forward,s['xyz'][:,:2],s['foot_centers'][:,:,:2],
                phase,commands[:,9:10],env.task_contact_activation)
            progress=((phase-commands[:,9:10])/(1-commands[:,9:10])).clamp(0,1)
            swing_weight=(progress/.15).clamp(0,1)*((1-progress)/.15).clamp(0,1)*env.task_contact_activation[:,None]
            scale=command_velocity_at_feet(commands[:,:3],forward,s['xyz'][:,:2],s['foot_centers'][:,:,:2]).norm(dim=-1).clamp(min=.04)
            valid=alive&~done&meter.valid&((tick+1)*env.dt>=1.)
            count+=valid;costs+=foot_velocity_cost(actual,target)*valid[:,None]
            direction=target/target.norm(dim=-1,keepdim=True).clamp(min=1e-8)
            normalized=foot_velocity_cost(actual,target,scale)*swing_weight
            normalized_costs+=normalized*valid[:,None]
            swing_costs+=foot_velocity_cost(actual,target)*swing_weight*valid[:,None]
            dense_gain+=(normalized-foot_velocity_cost(actual+.01*direction,target,scale)*swing_weight)*valid[:,None]
            aligned+=(actual*direction).sum(-1)*valid[:,None];targets+=target.norm(dim=-1)*valid[:,None]
            for key,value in s['weighted_rewards'].items():
                if key not in rewards:rewards[key]=torch.zeros(n,device=env.device)
                rewards[key]+=value/env.dt*valid
            alive&=~done
    rows=[]
    for g,(name,command) in enumerate(groups):
        sl=slice(g*per,(g+1)*per);den=count[sl].sum().clamp(min=1)
        rows.append(dict(name=name,command=command,survival_fraction=float(alive[sl].float().mean()),samples=int(count[sl].sum()),
            shadow_cost_per_foot=(costs[sl].sum(0)/den).cpu().tolist(),
            aligned_actual_velocity_per_foot_m_s=(aligned[sl].sum(0)/den).cpu().tolist(),
            target_speed_per_foot_m_s=(targets[sl].sum(0)/den).cpu().tolist(),
            shadow_swing_cost_per_foot=(swing_costs[sl].sum(0)/den).cpu().tolist(),
            shadow_normalized_swing_cost_per_foot=(normalized_costs[sl].sum(0)/den).cpu().tolist(),
            normalized_swing_cost_reduction_for_1cm_s_aligned_velocity=(dense_gain[sl].sum(0)/den).cpu().tolist(),
            actual_rewards_per_second={k:float(v[sl].sum()/den)for k,v in rewards.items()}))
    atomic_json(a.output,dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),groups=rows,
        seed=419,horizon_s=6.,envs_per_group=per,reference_execution_calls=env.reference_execution_calls,
        shadow_scale_m_s=.1,task_acceptance=False,training_reward_changed=False,
        scope='Shadow cost only; original nonterminal samples after1s; center displacement at20ms, not contact slip or hardware velocity.'))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
