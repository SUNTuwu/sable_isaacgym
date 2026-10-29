"""Bounded 32-environment evaluation of frozen same-iteration teacher snapshots.

No optimizer is created. The imported bank runtime defines original physics and
the full reference-plus-bounded-offset control. Checkpoints remain immutable.
"""
import argparse,hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
from isaacgym import gymtorch
import numpy as np
import torch

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.small_eval.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(path)

def fail_signal(signum,frame):raise RuntimeError('Evaluator interrupted by signal '+str(signum))

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--core',type=Path,required=True);p.add_argument('--horizon-s',type=float,default=16.)
    p.add_argument('--seed',type=int,default=947);p.add_argument('--max-wall-s',type=float,default=170.)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    started=time.time();bank=a.bank.resolve();corepath=a.core.resolve()
    project=Path(os.environ['SABLE_GYM_ROOT']).resolve()
    assert bank.parent==project/'runs' and bank.name.startswith('teacher_bank_')
    assert bank/'runtime' in corepath.parents and corepath.is_file()
    statuspath=bank/('small_evaluation_smoke.json' if a.smoke else 'small_evaluation_status.json')
    lock=bank/'small_evaluation.lock';env=None;locked=False
    signal.signal(signal.SIGTERM,fail_signal);signal.signal(signal.SIGINT,fail_signal)
    def status(state,**extra):
        atomic(statuspath,dict(status=state,pid=os.getpid(),bank=str(bank),core=str(corepath),
                              started_unix=started,updated_unix=time.time(),wall_s=time.time()-started,**extra))
    def wall_check():
        if time.time()-started>a.max_wall_s:raise TimeoutError('Small evaluator wall-time budget exceeded')
    try:
        if lock.exists():
            old=json.loads(lock.read_text(encoding='utf8'));pid=int(old['pid'])
            proc=Path('/proc')/str(pid)/'cmdline'
            if proc.exists() and 'evaluate_teacher_bank_small' in proc.read_bytes().decode(errors='ignore'):
                raise RuntimeError('Another small evaluator is active for this bank')
            lock.unlink()
        with lock.open('x',encoding='utf8') as f:f.write(json.dumps({'pid':os.getpid(),'created_unix':started}))
        locked=True;status('loading_checkpoints')
        state=json.loads((bank/'teacher_bank.json').read_text(encoding='utf8'))
        assert state['teacher_control_mode']=='reference_plus_residual'
        assert state['reference_strength']==1. and state['reference_time_scale']==1.
        teachers=state['teachers'];assert len(teachers)==8
        ids=[t['id'] for t in teachers];assert len(set(ids))==8
        # Read immutable CPs through atomic per-teacher receipts. Retry only the
        # short window in which a live training process is publishing eight CPs.
        for attempt in range(20):
            receipts=[];parents=[]
            for ident in ids:
                receipt=json.loads((bank/ident/'checkpoint.json').read_text(encoding='utf8'))
                cp=Path(receipt['path']).resolve();assert bank/ident in cp.parents
                assert digest(cp)==receipt['sha256']
                parent=torch.load(str(cp),map_location='cpu')
                assert parent['teacher_id']==ident and parent['iteration']==receipt['iteration']
                assert parent['teacher_control_mode']=='reference_plus_residual'
                assert parent['reference_strength']==1. and parent['reference_time_scale']==1.
                assert parent['offset_limit_deg']==state['offset_limit_deg']
                assert digest(parent['reference'])==parent['reference_sha256']
                receipts.append(receipt);parents.append(parent)
            iterations={x['iteration'] for x in parents}
            if len(iterations)==1:break
            wall_check();time.sleep(.5)
        else:raise RuntimeError('Eight checkpoint receipts did not reach a common iteration')
        iteration=parents[0]['iteration'];reference=Path(parents[0]['reference']).parent
        assert all(Path(x['reference']).parent==reference for x in parents)
        snapshot=dict(iteration=iteration,checkpoint_receipts=receipts,core=str(corepath),core_sha256=digest(corepath),
                      reference=str(reference),created_unix=time.time(),same_iteration_verified=True)
        snapshots=bank/'evaluations';snapshots.mkdir(exist_ok=True)
        atomic(snapshots/('small_snapshot_%06d.json'%iteration),snapshot)
        sys.path.insert(0,str(corepath.parent))
        module_spec=importlib.util.spec_from_file_location('frozen_small_eval_core',str(corepath))
        core=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(core)
        torch.set_num_threads(2);torch.manual_seed(a.seed+iteration);np.random.seed(a.seed+iteration)
        manifest=json.loads((reference/'manifest.json').read_text(encoding='utf8'))
        clips=[next(c for c in manifest['clips'] if c['id']==ident) for ident in ids]
        cfg=core.student_config(project/'runs/stand5_20261027_v44_stride80_all',stage='S3',num_envs=32)
        cfg.migration.freeze_waist=False;cfg.migration.moving_start_training=False
        cfg.migration.command_frontier_alpha=1.;cfg.migration.history_noise_level=0.
        cfg.commands.resampling_time=60.;cfg.env.episode_length_s=max(24.,a.horizon_s+4.)
        for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
        cfg.rewards.scales.alive=1.
        cfg.teacher_bank=dict(clips=clips,train_per=0,eval_per=4,reference=str(reference),
             control_mode='reference_plus_residual',reference_strength=1.,time_scale=1.,
             imitation_scale=parents[0].get('imitation_reward_scale',.5),offset_limit_deg=state['offset_limit_deg'])
        wall_check();status('creating_simulation',iteration=iteration)
        env=core.create_env(cfg);env.eval_reset_phase=0.;env.reset()
        policies=[core.Policy().to(env.device) for _ in ids]
        for policy,parent in zip(policies,parents):
            assert parent['action_names']==list(core.ACTION_NAMES)
            policy.load_state_dict(parent['model_state_dict'],strict=True);policy.eval()
        generator=torch.Generator(device=env.device).manual_seed(a.seed+iteration)
        perturb=(torch.rand((32,15),device=env.device,generator=generator)-.5)*.01
        allids=torch.arange(32,device=env.device)
        env.dof_pos[allids[:,None],env.active]+=perturb
        env.dof_pos[allids,env.dependent]=-env.dof_pos[allids,env.hip]
        actors=env.robot_actor_indices[allids]
        env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(actors),32)
        env._refresh_proprio_state();env.compute_observations()
        alive=torch.ones(32,device=env.device,dtype=torch.bool)
        count=torch.zeros(32,device=env.device);qsum=count.clone();xysum=count.clone();yawsum=count.clone()
        actual=torch.zeros(32,3,device=env.device);steps=torch.zeros(32,4,device=env.device)
        previous_contact=None;falls=[None]*32;trace=[[] for _ in ids];executed=0
        max_steps=2 if a.smoke else int(round(a.horizon_s/env.dt))
        status('evaluating',iteration=iteration,requested_horizon_s=a.horizon_s)
        with torch.no_grad():
            for step in range(max_steps):
                wall_check();obs=env.teacher_observations();assert obs.shape==(32,86)
                action=torch.cat([policy.actor(obs[i*4:(i+1)*4]) for i,policy in enumerate(policies)])
                assert torch.isfinite(action).all()
                env.step(action);d=env.pre_reset_teacher_state;mask=alive.float()
                count+=mask;qsum+=d['q_rmse'].square()*mask;xysum+=d['xy_error'].square()*mask;yawsum+=d['yaw_error'].square()*mask
                actual+=d['actual_velocity']*mask[:,None]
                if previous_contact is not None:steps+=((d['contact']&~previous_contact)&alive[:,None]).float()
                previous_contact=d['contact'].clone()
                newly=torch.nonzero(alive&d['fall'],as_tuple=False).flatten().tolist()
                for j in newly:
                    details={k:v[j].item() for k,v in d.get('fall_diagnostics',{}).items()}
                    falls[j]=dict(time_s=(step+1)*env.dt,**details)
                alive&=~d['fall'];executed=step+1
                if step%5==0 or newly:
                    for i in range(8):
                        j=i*4;r=d['root'][j]
                        trace[i].append(dict(time_s=(step+1)*env.dt,q=d['q'][j].tolist(),
                            root_position=(r[:3]-env.env_origins[j]).tolist(),root_quaternion_xyzw=r[3:7].tolist(),
                            toe_world=(d['toe'][j]-env.env_origins[j]).tolist(),
                            reference_toe_world=(d['reference_toe'][j]-env.env_origins[j]).tolist(),
                            contact_normal_force=d['contact_force'][j].tolist(),commands=env.teacher_commands[i].tolist(),
                            actual_velocity=d['actual_velocity'][j].tolist(),active=bool(alive[j]),
                            executed_target_full=d['executed_target_full'][j].tolist(),residual_action=d['residual_action'][j].tolist(),
                            offset_joint_rad=d['offset_joint_rad'][j].tolist(),reference_joint_target=d['reference_joint_target'][j].tolist(),
                            teacher_control_mode='reference_plus_residual'))
                if not bool(alive.any()):break
        if a.smoke:
            status('smoke_passed',iteration=iteration,steps=executed,shape=[32,86],no_optimizer_created=True)
            print(json.dumps({'status':'smoke_passed','iteration':iteration,'steps':executed}),flush=True);return
        reports=[]
        for i,ident in enumerate(ids):
            sl=slice(i*4,(i+1)*4);n=count[sl].clamp(min=1)
            report=dict(teacher_id=ident,iteration=iteration,checkpoint_iteration=iteration,checkpoint=receipts[i],
              independent=True,evaluation_seed=a.seed+iteration,reset='reference_phase0 plus seeded joint perturbation +/-0.005rad',
              evaluator='independent32env_small_sim',horizon_s=a.horizon_s,executed_simulation_s=executed*env.dt,num_envs=4,
              survival_fraction=float(alive[sl].float().mean()),q_tracking_rmse_rad=float((qsum[sl].sum()/n.sum()).sqrt()),
              xy_rmse_m_s=float((xysum[sl].sum()/n.sum()).sqrt()),yaw_rmse_rad_s=float((yawsum[sl].sum()/n.sum()).sqrt()),
              mean_command=env.teacher_commands[i].tolist(),mean_actual=(actual[sl]/n[:,None]).mean(0).tolist(),
              steps_per_foot=steps[sl].sum(0).tolist(),step_scope='Raw normal-force contact onsets excluding first sample; not genuine-air-step acceptance',
              samples=int(n.sum()),accepted=False,first_fall_diagnostics=falls[i*4:(i+1)*4],
              teacher_control_mode='reference_plus_residual',reference_strength=1.,reference_time_scale=1.,
              offset_limit_deg=state['offset_limit_deg'],created_unix=time.time(),
              scope='First original episodes only. All-first-episodes-failed permits early completion; no optimizer and no checkpoint mutation.')
            folder=bank/ident
            atomic(folder/'evaluations'/('independent_small_%06d.json'%iteration),report)
            atomic(folder/'latest_evaluation.json',report)
            tracepath=folder/'videos'/('trace_small_%06d.json'%iteration)
            atomic(tracepath,dict(teacher_id=ident,iteration=iteration,joint_names=list(core.ACTION_NAMES),type='kinematic_trace',frames=trace[i]))
            atomic(folder/'video_receipt.json',dict(type='kinematic_trace',path=str(tracepath),iteration=iteration,
                   checkpoint_path=receipts[i]['path'],checkpoint_sha256=receipts[i]['sha256'],
                   source='Independent32env deterministic rollout, no optimizer',created_unix=time.time()))
            reports.append(report)
        atomic(bank/'small_evaluation_result.json',dict(snapshot=snapshot,reports=reports,created_unix=time.time()))
        status('completed',iteration=iteration,teachers=8,executed_simulation_s=executed*env.dt,
               early_all_first_episodes_failed=not bool(alive.any()),all_checkpoint_hashes_verified=True)
        print(json.dumps({'status':'completed','iteration':iteration,'survival':[r['survival_fraction'] for r in reports],'wall_s':time.time()-started}),flush=True)
    except BaseException as error:
        status('failed',error=type(error).__name__+': '+str(error));raise
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)
        if locked and lock.exists():lock.unlink()

if __name__=='__main__':main()
