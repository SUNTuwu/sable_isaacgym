"""Paired eight-environment frozen-policy reward diagnosis; no optimizer.

Two policies run simultaneously, four environments each, with identical clip
and reset perturbations. Default is a CPU-only plan; --run-simulation opts in.
Every 200 steps persists partial metrics plus a receipt outside all run folders.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def argument_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--teacher',choices=('04_TURN_LEFT','05_TURN_RIGHT'),required=True)
    parser.add_argument('--checkpoint',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--control',type=Path)
    parser.add_argument('--reward-module',type=Path)
    parser.add_argument('--cycles-module',type=Path)
    parser.add_argument('--seed',type=int,default=947)
    parser.add_argument('--max-wall-s',type=float,default=600.)
    parser.add_argument('--run-simulation',action='store_true')
    return parser


def paired_seed_rows(teacher_index):
    """The source bank's four reset-noise rows, duplicated in A/B order."""
    assert 0<=teacher_index<8
    rows=list(range(teacher_index*4,(teacher_index+1)*4))
    return rows+rows


def validate_output(project,out):
    target=Path(out).resolve();runs=(Path(project)/'runs').resolve()
    if target==runs or runs in target.parents:
        raise ValueError('Diagnostic output must be outside every training run directory')
    return target


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(path)


def write_progress(out,plan,reports,status,executed,max_steps,wall_s,error=None):
    complete=status=='completed' and len(reports)==2 and all(r.get('complete') for r in reports)
    payload=dict(plan=plan,reports=reports,status=status,complete=complete,partial=not complete,
        executed_steps=executed,requested_steps=max_steps,wall_s=wall_s,updated_unix=time.time(),
        comparison='Two frozen policies concurrently in one 8-env scene; four paired resets each; same immutable runtime/reference/rewards; no optimizer.',
        scope='Complete first natural episodes' if complete else 'INCOMPLETE diagnostic; never evidence of full-episode acceptance')
    if error:payload['error']=error
    atomic_json(out,payload)
    receipt=Path(out).with_name(Path(out).stem+'.receipt.json')
    atomic_json(receipt,dict(status=status,complete=complete,partial=not complete,pid=os.getpid(),
        output=str(Path(out).resolve()),checkpoint_receipts=plan['checkpoints'],
        executed_steps=executed,requested_steps=max_steps,wall_s=wall_s,
        updated_unix=payload['updated_unix'],error=error,optimizer_created=False,training_files_mutated=False))
    print(json.dumps(dict(status=status,complete=complete,step=executed,requested_steps=max_steps,
        wall_s=round(wall_s,2),output=str(out),survival_so_far=[r.get('survival_so_far') for r in reports]),ensure_ascii=False),flush=True)
    return payload


def run_simulation(args,plan,state,project,bank,reference,teacher_index,control):
    started=time.time();env=None;reports=[];executed=0;max_steps=None;build_reports=None
    def persist(status,error=None):
        nonlocal reports
        if build_reports is not None:
            try:reports=build_reports()
            except Exception as exc:
                error=(error+'; ' if error else '')+'latest snapshot unavailable: '+repr(exc)
        return write_progress(args.out,plan,reports,status,executed,max_steps,time.time()-started,error)
    def interrupted(signum,frame):raise InterruptedError('Evaluator received signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    persist('loading')
    try:
        # Import order is required by Isaac Gym. No GPU imports occur in plan mode.
        from isaacgym import gymtorch
        import numpy as np
        import torch
        corepath=bank/'runtime/teacher_bank_core.py';sys.path.insert(0,str(corepath.parent))
        for name,path in [('teacher_turn_cycles',args.cycles_module),('teacher_reference_rewards_v2',args.reward_module)]:
            if path:
                spec=importlib.util.spec_from_file_location(name,str(path.resolve()))
                module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        spec=importlib.util.spec_from_file_location('frozen_turn_diagnostic_core',str(corepath))
        core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
        parents=[torch.load(item['path'],map_location='cpu') for item in plan['checkpoints']]
        for parent in parents:
            assert parent['observations']==94 and parent['action_names']==list(core.ACTION_NAMES)
            assert parent['teacher_control_mode']=='reference_plus_residual'
            assert parent['reference_strength']==parent['reference_time_scale']==1.
            assert parent['offset_limit_deg']==state['offset_limit_deg']
        manifest=json.loads((reference/'manifest.json').read_text())
        clip=next(c for c in manifest['clips'] if c['id']==args.teacher)
        bootstrap=corepath.parent/'bootstrap'
        if not (bootstrap/'config.json').exists():bootstrap=project/'runs/stand5_20261027_v44_stride80_all'
        cfg=core.student_config(bootstrap,stage='S3',num_envs=8)
        cfg.migration.freeze_waist=False;cfg.migration.moving_start_training=False
        cfg.migration.command_frontier_alpha=1.;cfg.migration.history_noise_level=0.
        cfg.commands.resampling_time=60.;cfg.env.episode_length_s=24.
        for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
        cfg.rewards.scales.alive=1.
        cfg.teacher_bank=dict(clips=[copy.deepcopy(clip),copy.deepcopy(clip)],train_per=0,eval_per=4,reference=str(reference),
            control_mode='reference_plus_residual',reference_strength=1.,time_scale=1.,
            imitation_scale=parents[0].get('imitation_reward_scale',0.),offset_limit_deg=state['offset_limit_deg'],
            balance_first=False,adaptive_enabled=False,motion_reward_module_enabled=False,walk_event_weight_scale=.25,
            task_tracking_profile=state.get('task_tracking_profile','legacy'),reward_cleanup_profile=state['reward_cleanup_profile'],
            step_quality_profile=state.get('step_quality_profile','legacy'),translation_guard_enabled=state.get('translation_guard_enabled',False),
            turn_speed_scale=state.get('turn_speed_scale',1.),turn_reference_time_scale=state.get('turn_reference_time_scale',1.),
            turn_offset_rate_deg_s=state.get('turn_offset_rate_deg_s',0.),turn_lift_config=state.get('turn_lift_config'),pace_stability_enabled=False)
        assert not state.get('motion_reward_module_enabled',False) and not state.get('pace_stability_enabled',False)
        persist('creating_simulation')
        env=core.create_env(cfg);env.night_control=copy.deepcopy(control)
        assert env.num_envs==8 and env.teacher_ids.tolist()==[0]*4+[1]*4
        policies=[core.Policy().to(env.device) for _ in parents]
        for policy,parent in zip(policies,parents):
            policy.load_state_dict(parent['model_state_dict'],strict=True)
            policy.minimum_std=parent.get('minimum_action_std',.02);policy.maximum_std=.6;policy.eval()
        allids=torch.arange(8,device=env.device)
        generator=torch.Generator(device=env.device).manual_seed(args.seed)
        original_noise=(torch.rand((32,15),device=env.device,generator=generator)-.5)*.01
        perturb=original_noise[torch.tensor(paired_seed_rows(teacher_index),device=env.device)]
        assert torch.equal(perturb[:4],perturb[4:])
        env.eval_reset_phase=0.;env.reset_idx(allids)
        env.dof_pos[allids[:,None],env.active]+=perturb
        env.dof_pos[allids,env.dependent]=-env.dof_pos[allids,env.hip]
        actors=env.robot_actor_indices[allids]
        env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(actors),8)
        env._refresh_proprio_state();env.compute_observations();env.eval_reset_phase=None
        alive=torch.ones(8,device=env.device,dtype=torch.bool);count=torch.zeros(8,device=env.device)
        undiscounted=count.clone();discounted=count.clone();radius=count.clone();net_yaw=count.clone()
        actual=torch.zeros(8,3,device=env.device);initial_xy=env.root_states[:,:2].clone()
        end_step=torch.zeros(8,device=env.device,dtype=torch.long);end_fall=~alive;end_timeout=~alive
        term_names=None;term_sum=None;term_discounted=None
        kl_sum=count.clone();action_delta_sum=count.clone()
        initial_obs=env.teacher_observations()
        with torch.no_grad():initial_values=[p.value(initial_obs[i*4:(i+1)*4]).tolist() for i,p in enumerate(policies)]
        snapshots16=[None,None];max_steps=int(env.max_episode_length)+2
        stds=[p.log_std.exp().clamp(p.minimum_std,.6)[:14].detach() for p in policies]
        commands=env.teacher_commands.tolist()
        def capture_reports():
            counts=count.tolist();ends=end_step.tolist();falls=end_fall.tolist();timeouts=end_timeout.tolist()
            actual_cpu=actual.tolist();radii=radius.tolist();yaw=net_yaw.tolist()
            returns=undiscounted.tolist();discounted_returns=discounted.tolist()
            kl=kl_sum.tolist();action_difference=action_delta_sum.tolist()
            terms=term_sum.tolist() if term_sum is not None else [[] for _ in range(8)]
            terms_discounted=term_discounted.tolist() if term_discounted is not None else [[] for _ in range(8)]
            result=[]
            for i,parent in enumerate(parents):
                group=list(range(i*4,(i+1)*4));samples=sum(counts[j] for j in group);complete=all(ends[j]>0 for j in group)
                ended=[dict(time_s=ends[j]*env.dt,fall=falls[j],timeout=timeouts[j],radius_m=radii[j],net_yaw_deg=math.degrees(yaw[j])) if ends[j] else None for j in group]
                result.append(dict(checkpoint=plan['checkpoints'][i],iteration=parent['iteration'],candidate_index=i,env_indices=group,
                    source_teacher_id=parent['teacher_id'],target_command=commands[i],complete=complete,partial=not complete,
                    survived_complete_episode_fraction=sum(ends[j]>0 and not falls[j] for j in group)/4,
                    survival_fraction=sum(not falls[j] for j in group)/4 if complete else None,
                    survival_so_far=sum(not falls[j] for j in group)/4,completed_episodes=sum(ends[j]>0 for j in group),
                    optimizer_lr_saved=[g.get('lr') for g in parent.get('optimizer_state_dict',{}).get('param_groups',[])],
                    normalized_action_std=stds[i].tolist(),snapshot_16s=snapshots16[i],first_episode_end=ended,
                    simulated_samples_per_env=[counts[j] for j in group],
                    mean_actual=[sum(actual_cpu[j][axis]/max(counts[j],1) for j in group)/4 for axis in range(3)],
                    max_turn_radius_m=max(radii[j] for j in group),return_per_env=[returns[j] for j in group],
                    discounted_return_per_env=[discounted_returns[j] for j in group],initial_critic_value=initial_values[i],
                    reward_terms_per_second={name:sum(terms[j][k] for j in group)/max(samples*env.dt,1e-9) for k,name in enumerate(term_names or [])},
                    reward_term_returns={name:[terms[j][k] for j in group] for k,name in enumerate(term_names or [])},
                    reward_term_discounted_returns={name:[terms_discounted[j][k] for j in group] for k,name in enumerate(term_names or [])},
                    baseline_to_candidate_gaussian_kl_on_this_rollout=[0.,sum(kl[j] for j in group)/max(samples,1)],
                    mean_residual_difference_from_baseline_rad=[0.,sum(action_difference[j] for j in group)/max(samples,1)],
                    kl_scope='Cumulative checkpoint divergence on this candidate rollout states, not KL of one PPO update',
                    omitted_timeout_value_bootstrap='Measured returns include terminal settlement, without critic bootstrap',
                    control_unchanged=control==env.night_control))
            return result
        build_reports=capture_reports;persist('evaluating')
        with torch.no_grad():
            for step in range(max_steps):
                if time.time()-started>args.max_wall_s:raise TimeoutError('Bounded diagnostic exceeded wall-time budget')
                obs=env.teacher_observations();mask=alive.float()
                # Only two actor forwards per simulation step, reused for action and KL.
                means=[policy.actor(obs) for policy in policies]
                ratio=stds[0]/stds[1]
                kl=((stds[1]/stds[0]).log()+.5*(ratio.square()+(means[0][:,:14]-means[1][:,:14]).square()/stds[1].square())-.5).sum(1)
                kl_sum+=kl*mask
                action_delta_sum+=((means[1][:,:14]-means[0][:,:14])*.35).square().mean(1).sqrt()*mask
                action=torch.cat((means[0][:4],means[1][4:]),0);action[:,-1]=0.
                env.step(action);data=env.pre_reset_teacher_state
                count+=mask;actual+=data['actual_velocity']*mask[:,None]
                net_yaw+=data['actual_velocity'][:,2]*env.dt*mask
                radius=torch.maximum(radius,(data['root'][:,:2]-initial_xy).norm(dim=1)*mask)
                names=list(env.teacher_terms)
                if term_names is None:
                    term_names=names;term_sum=torch.zeros(8,len(names),device=env.device);term_discounted=torch.zeros_like(term_sum)
                assert term_names==names
                contributions=torch.stack([env.teacher_terms[name] for name in names],1)*mask[:,None]*env.dt
                term_sum+=contributions;term_discounted+=contributions*(.99**step)
                reward=contributions.sum(1);undiscounted+=reward;discounted+=reward*(.99**step)
                newly=alive&env.reset_buf.bool()
                end_step=torch.where(newly,torch.full_like(end_step,step+1),end_step)
                end_fall|=newly&data['fall'];end_timeout|=newly&env.time_out_buf
                alive&=~env.reset_buf.bool();executed=step+1
                if executed==800:
                    snapshots16=[{k:r[k] for k in ('survival_so_far','mean_actual','max_turn_radius_m','return_per_env','discounted_return_per_env','reward_terms_per_second')} for r in capture_reports()]
                # GPU-to-host synchronization and durable output only at bounded intervals.
                if executed%200==0 or executed==max_steps:
                    persist('evaluating')
                    if not bool(alive.any()):break
        if not bool((end_step>0).all()):
            persist('incomplete_horizon','Some original episodes did not end within the requested control steps');return 2
        persist('completed');return 0
    except BaseException as exc:
        status='timed_out' if isinstance(exc,TimeoutError) else 'interrupted' if isinstance(exc,InterruptedError) else 'failed'
        persist(status,type(exc).__name__+': '+str(exc));return 2
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


def main():
    args=argument_parser().parse_args();project=Path(os.environ['SABLE_GYM_ROOT']).resolve();bank=args.bank.resolve()
    assert bank.parent==project/'runs'
    assert len(args.checkpoint)==2,'Exactly two checkpoints are required for paired 8-env A/B'
    args.out=validate_output(project,args.out)
    state=json.loads((bank/'teacher_bank.json').read_text())
    controlpath=args.control or project/'deployment/night_20261029/control.json';control=json.loads(controlpath.read_text())
    checkpoints=[p.resolve() for p in args.checkpoint]
    for path in checkpoints:assert project/'runs' in path.parents and path.is_file()
    teacher_index=next(i for i,item in enumerate(state['teachers']) if item['id']==args.teacher)
    reference=Path(state['teachers'][teacher_index]['reference_metadata']).parent
    plan=dict(bank=str(bank),teacher=args.teacher,envs=8,envs_per_policy=4,policies_concurrent=True,seed=args.seed,
        reset_noise_rows=paired_seed_rows(teacher_index),checkpoints=[dict(path=str(p),sha256=digest(p)) for p in checkpoints],
        runtime={str(p):digest(p) for p in sorted((bank/'runtime').glob('*.py'))},control_path=str(controlpath),
        control_sha256=digest(controlpath),control_snapshot=control,reference=str(reference),
        reference_sha256=digest(reference/(args.teacher+'.npz')),horizon='One simultaneous 24s first episode per env, natural terminal settlement; partial checkpoint every 200 steps',
        optimizer_created=False,training_files_mutated=False,output=str(args.out),max_wall_s=args.max_wall_s)
    plan['module_overrides']={name:dict(path=str(path.resolve()),sha256=digest(path)) for name,path in
        [('teacher_turn_cycles',args.cycles_module),('teacher_reference_rewards_v2',args.reward_module)] if path}
    if not args.run_simulation:
        print(json.dumps(dict(mode='cpu_only_plan',**plan),ensure_ascii=False,indent=2));return 0
    return run_simulation(args,plan,state,project,bank,reference,teacher_index,control)


if __name__=='__main__':raise SystemExit(main())
