"""Isolated actor-learning experiment with differentiable task-space clearance.

This does not modify PPO or a running model. Private measured root/q are labels
only. The exported candidate keeps the same inference graph, with no execution
reference. All geometry is hypothetical until a separate closed-loop evaluation.
"""
import argparse
import copy
import json
import math
from pathlib import Path
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from history_policy import HistoryActorCritic
from action_schema import ActionSpec
from target_pose_clearance import executed_target_coordinates,target_pose_clearance_cost
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,append_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--updates',type=int,default=128)
    p.add_argument('--reuse-collection',type=Path)
    p.add_argument('--old-kl-weight',type=float,default=.5)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(431)
    commands=[(.105,0,0),(0,0,0),(-.04,0,0),(0,.04,0),(0,-.04,0),(0,0,.35),(0,0,-.35),(.07,.03,.25)]
    per=8;n=per*len(commands);cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.reference.file='/intentionally_absent_clearance_gradient_probe.json'
    cfg.commands.resampling_time=1000.;cfg.env.episode_length_s=7.
    env=create_migration_env(cfg,noise=False);env.reset_noise=.015
    parent=torch.load(str(a.checkpoint),map_location=env.device)
    policy=HistoryActorCritic().to(env.device).eval();policy.load_state_dict(parent['model_state_dict'],strict=True)
    policy.novel_command_std_floor=.04;policy.reverse_hind_std_floor=.08
    for parameter in policy.parameters():parameter.requires_grad_(False)
    for parameter in policy.actor.parameters():parameter.requires_grad_(True)
    env.reset();c=env.commands16.clone()
    for i,cmd in enumerate(commands):c[i*per:(i+1)*per,:3]=c.new_tensor(cmd)
    env.set_commands(c);alive=torch.ones(n,device=env.device,dtype=torch.bool);rows=[];fk_error=0.
    collection_source=None
    if a.reuse_collection:
        receipt=json.loads((a.reuse_collection/'result.json').read_text())
        assert receipt['parent_sha256']==sha256(a.checkpoint)
        fk_error=receipt['actual_FK_native_max_error_m']
        collection_survival=receipt['original_collection_survival_fraction']
        data=torch.load(a.reuse_collection/'dataset.pt',map_location=env.device)
        collection_source=dict(path=str(a.reuse_collection),dataset_sha256=sha256(a.reuse_collection/'dataset.pt'),
            receipt_sha256=sha256(a.reuse_collection/'result.json'))
    else:
        with torch.no_grad():
            for step in range(300):
                if step%5==0:
                    valid=alive&(env.episode_length_buf*env.dt>=max(1.,cfg.reference.ramp_s))
                    local=torch.remainder(env.clock.local_phases(c)+c[:,5:6]*env.dt,1.)
                    beta=c[:,9:10];progress=((local-beta)/(1-beta)).clamp(0,1)
                    moving=(c[:,:2].norm(dim=1)>.01)|(c[:,2].abs()>.08)
                    swing=(local>=beta)&moving[:,None]
                    desired=torch.sin(math.pi*progress).square()*c[:,10:11]*swing
                    weight=(progress/.15).clamp(0,1)*((1-progress)/.15).clamp(0,1)*swing
                    if valid.any():
                        actual_fk=env.mapping.task_measurements(env.root_states[:,:3],env.base_quat,env.dof_pos,env.dof_names)[3]
                        fk_error=max(fk_error,float((actual_fk[valid]-env.foot_clearance[valid]).abs().max()))
                        old=torch.arange(n,device=env.device)<2*per
                        rows.append(dict(raw=env.get_observations()[valid].clone(),q=env.dof_pos[valid].clone(),
                            root=env.root_states[valid,:3].clone(),quat=env.base_quat[valid].clone(),
                            desired=desired[valid].clone(),weight=weight[valid].float().clone(),
                            height=c[valid,10:11].clone(),old=old[valid].clone()))
                _,_,_,done,_=env.step(policy.act_inference(env.get_observations()));alive&=~done
        data={k:torch.cat([r[k]for r in rows])for k in rows[0]}
        collection_survival=float(alive.float().mean())
    assert fk_error<3e-5 and env.reference_execution_calls==0
    torch.save({k:v.cpu()for k,v in data.items()},a.output_dir/'dataset.pt')
    with torch.no_grad():
        data['anchor_action']=policy.act_inference(data['raw']).clone()
        data['anchor_std']=policy.distribution_std(data['raw']).clone()
    ids=torch.randperm(len(data['raw']),device=env.device)[:1024]
    validation={k:v[ids]for k,v in data.items()}
    def prediction(d):
        action=policy.act_inference(d['raw'])
        target=ActionSpec.target(action,env.default_dof_pos[:,env.active],env.hard_limits[env.active],True)
        q=executed_target_coordinates(d['q'],env.active,target,env.hip,env.dependent)
        height=env.mapping.task_measurements(d['root'],d['quat'],q,env.dof_names)[3]
        cost=target_pose_clearance_cost(height,d['desired'],d['weight'],d['height']).mean(1)
        return action,cost,height
    with torch.no_grad():
        base_action,base_cost,base_height=prediction(validation)
        std=policy.distribution_std(validation['raw'])[:,:14].clone()
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=1e-4)
    accepted=0
    for update in range(a.updates):
        ids=torch.randint(len(data['raw']),(512,),device=env.device)
        batch={k:v[ids]for k,v in data.items()}
        before=copy.deepcopy(policy.actor.state_dict());before_optimizer=copy.deepcopy(optimizer.state_dict())
        with torch.no_grad():previous_action=policy.act_inference(validation['raw'])
        action,cost,_=prediction(batch)
        anchor_kl=.5*((action[:,:14]-batch['anchor_action'][:,:14])/batch['anchor_std'][:,:14]).square().sum(1)
        loss=cost[~batch['old']].mean()+a.old_kl_weight*anchor_kl[batch['old']].mean()
        optimizer.zero_grad();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all()for p in policy.actor.parameters())
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
        with torch.no_grad():
            action=policy.act_inference(validation['raw'])
            kl=.5*((action[:,:14]-previous_action[:,:14])/std).square().sum(1)
            old_total=.5*((action[:,:14]-base_action[:,:14])/std).square().sum(1)
            mean_kl=float(kl.mean());old_kl=float(kl[validation['old']].mean())
            old_cumulative=float(old_total[validation['old']].mean())
        ok=mean_kl<=.01 and old_kl<=.002 and old_cumulative<=.1
        lr=optimizer.param_groups[0]['lr']
        if ok:accepted+=1;next_lr=min(lr*1.02,1e-4)
        else:
            policy.actor.load_state_dict(before);optimizer.load_state_dict(before_optimizer);next_lr=max(lr*.5,1e-7)
        for group in optimizer.param_groups:group['lr']=next_lr
        if update%16==0 or update+1==a.updates:
            append_json(a.output_dir/'updates.jsonl',dict(update=update+1,loss=float(loss),accepted=ok,
                proposal_mean_kl=mean_kl,proposal_old_kl=old_kl,proposal_old_total_kl=old_cumulative,learning_rate=next_lr))
    with torch.no_grad():
        final_action,final_cost,final_height=prediction(validation)
        final_old_kl=float((.5*((final_action[:,:14]-base_action[:,:14])/std).square().sum(1))[validation['old']].mean())
    candidate=a.output_dir/'candidate.pt'
    torch.save(dict(model_state_dict=policy.state_dict(),iter=a.updates,schema=parent['schema'],
        teacher_sha256=parent['teacher_sha256'],stage='S3',reason='isolated_task_space_actor_gradient_experiment',
        parent_checkpoint=str(a.checkpoint),parent_sha256=sha256(a.checkpoint)),candidate)
    novel=~validation['old'];swing=novel[:,None]&(validation['weight']>.5)
    result=dict(parent_checkpoint=str(a.checkpoint),parent_sha256=sha256(a.checkpoint),candidate=str(candidate),
        candidate_sha256=sha256(candidate),seed=431,collected_examples=len(data['raw']),
        optimization_monitoring_examples=len(validation['raw']),
        monitoring_split='Subset of the collected optimization states, not held-out generalization evidence',
        original_collection_survival_fraction=collection_survival,actual_FK_native_max_error_m=fk_error,
        collection_source=collection_source,old_domain_anchor_KL_weight=a.old_kl_weight,
        attempted_updates=a.updates,accepted_updates=accepted,old_domain_total_KL=final_old_kl,
        novel_hypothetical_cost_before=float(base_cost[novel].mean()),novel_hypothetical_cost_after=float(final_cost[novel].mean()),
        hypothetical_swing_clearance_mean_before_m=float(base_height[swing].mean()),
        hypothetical_swing_clearance_mean_after_m=float(final_height[swing].mean()),
        reference_execution_calls=env.reference_execution_calls,task_acceptance=False,
        scope='Offline geometric actor learning experiment, not PPO. True root/q are loss labels only. Same326D inference; no FK/teacher/reference execution in inference. Hypothetical clearance must be validated in a separate live simulation.')
    atomic_json(a.output_dir/'result.json',result);print(result,flush=True);env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
