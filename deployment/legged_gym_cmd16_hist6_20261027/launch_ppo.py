"""Launch one authorized frozen PPO round after a verified parent archive."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from run_io import atomic_json,sha256,read_json
from migration_config import student_config

ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--stage',choices=['S2','S3','S5'],required=True)
    p.add_argument('--acceptance',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--select-best-regression',action='store_true')
    p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--frontier-alpha',type=float)
    p.add_argument('--frontier-evidence',type=Path)
    p.add_argument('--require-normal-completion',action='store_true')
    p.add_argument('--yaw-error-scale',type=float)
    p.add_argument('--preserve-learning-state-evidence',type=Path)
    a=p.parse_args()
    parent=a.parent_run.resolve()
    until=time.time()+1800
    while time.time()<until:
        state=read_json(parent/'status.json',{})
        if a.require_normal_completion:
            if state.get('status') in ('stopped_requested','failed') or (parent/'STOP').exists():
                raise RuntimeError('Queued progression cancelled: parent stopped or failed')
            if read_json(ROOT/'results/cmd16_hist6/current_migration.json',{}).get('run')!=str(parent):
                raise RuntimeError('Queued progression cancelled: active run changed')
        if state.get('status') in ('stopped_requested','stopped_at_deadline','iterations_complete') and (parent/'backup_receipt.json').exists():break
        time.sleep(3)
    else:raise RuntimeError('Parent did not finish its saved/archive cycle')
    receipt=read_json(parent/'backup_receipt.json')
    assert sha256(receipt['archive'])==receipt['sha256']
    cp=read_json(parent/'checkpoint.json')
    selection=None
    if a.select_best_regression:
        if a.stage!='S2' or a.checkpoint:raise ValueError('Automatic regression selection is S2-only and excludes --checkpoint')
        from select_regression_checkpoint import select_checkpoint
        cp,selection=select_checkpoint(parent,read_json(parent/'baseline_metrics.json'))
    if a.checkpoint:
        assert a.checkpoint.resolve().parent==parent/'checkpoints'
        cp=dict(path=str(a.checkpoint.resolve()),sha256=sha256(a.checkpoint))
    assert sha256(cp['path'])==cp['sha256']
    with tarfile.open(receipt['archive'],'r:gz')as tar:
        member=parent.name+'/checkpoints/'+Path(cp['path']).name
        assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==cp['sha256']
    parent_config=read_json(parent/'config.json',{})
    continuation=parent_config.get('stage')==a.stage and 'on-policy PPO' in parent_config.get('algorithm','')
    expected_config=student_config(read_json(parent/'teacher_manifest.json')['run'],a.stage,a.num_envs)
    if a.yaw_error_scale is None and a.stage in ('S3','S5') and parent_config.get('stage') in ('S3','S5'):
        # Carry the chosen yaw objective through continuation and the frequency
        # stage; changing it must be explicit, not a reset to the source default.
        a.yaw_error_scale=parent_config.get('yaw_error_scale_rad_s',1.)
    if a.yaw_error_scale is not None:
        if not math.isfinite(a.yaw_error_scale)or a.yaw_error_scale<=0:raise ValueError('Invalid yaw error scale')
        expected_config.migration.yaw_error_scale_rad_s=a.yaw_error_scale
    parent_alpha=parent_config.get('command_frontier',{}).get('training_alpha') if parent_config.get('stage')=='S3' else None
    frontier_alpha=1.;frontier_gate=None
    if a.stage=='S3':
        frontier_alpha=a.frontier_alpha if a.frontier_alpha is not None else (
            parent_alpha if parent_alpha is not None else expected_config.migration.training_frontier_alpha)
        if not 0.<=frontier_alpha<=1.:raise ValueError('Frontier alpha must be in[0,1]')
        if parent_alpha is not None and frontier_alpha!=parent_alpha:
            if not a.frontier_evidence:raise ValueError('Changed command frontier requires measured progression evidence')
            frontier_gate=read_json(a.frontier_evidence)
            assert frontier_gate['schema']=='s3_frontier_progression_v1' and frontier_gate['passed']
            assert not frontier_gate['full_S3_accepted'] and frontier_gate['training_alpha']==parent_alpha
            assert frontier_gate['checkpoint_sha256']==cp['sha256']
            assert parent_alpha<frontier_alpha<=min(1.,parent_alpha+.125)
            sources=frontier_gate['sources']
            for item in sources.values():assert sha256(item['path'])==item['sha256']
            assert sources['baseline']['sha256']==sha256(parent/'baseline_metrics.json')
            from evaluate_frontier import assess_frontier
            verified=assess_frontier(read_json(sources['frontier']['path']),read_json(sources['endpoint']['path']),
                read_json(parent/'baseline_metrics.json'),parent_alpha)
            assert verified['passed'] and verified['checkpoint_sha256']==cp['sha256']
    elif a.frontier_alpha not in (None,1.):raise ValueError('Only S3 permits intermediate velocity commands')
    same_rewards=(parent_config.get('reward_weights')==vars(expected_config.rewards.scales)
                  and parent_config.get('velocity_error_scale_m_s')==expected_config.migration.velocity_error_scale_m_s)
    same_rewards &= parent_config.get('unclipped_gait_costs',False)==expected_config.migration.unclipped_gait_costs
    same_rewards &= parent_config.get('yaw_error_scale_rad_s',1.)==expected_config.migration.yaw_error_scale_rad_s
    same_curriculum=parent_config.get('moving_start_curriculum',{}).get('enabled',False)==(a.stage=='S3')
    same_exploration=parent_config.get('minimum_action_std',.02)==expected_config.migration.minimum_action_std
    same_exploration &= parent_config.get('novel_command_std_floor',0.)==expected_config.migration.novel_command_std_floor
    same_exploration &= parent_config.get('reverse_hind_std_floor',0.)==expected_config.migration.reverse_hind_std_floor
    gate_source=a.acceptance
    if a.stage in ('S3','S5'):
        if continuation:
            gate_source=parent/'parent_acceptance.json'
        if not gate_source:raise ValueError('New curriculum requires prior stage acceptance')
        gate=read_json(gate_source)
        assert gate['passed']
        if not continuation:assert gate['checkpoint_sha256']==cp['sha256']
        assert gate['stage']==('S2' if a.stage=='S3' else 'S3')
    source=Path(__file__).resolve().parent
    # Exclude process/diagnostic entrypoints, never physics, policy, curriculum
    # or optimizer implementations. A code change deliberately resets Adam.
    nontraining={'train_ppo.py','launch_ppo.py','watch_migration.py','watch_frontier.py','export_policy.py','select_regression_checkpoint.py','queue_continuation.py'}
    training_files=[f for f in source.glob('*.py') if f.name not in nontraining
        and not f.name.startswith(('evaluate_','evaluation_','probe_','check_'))]
    same_training_code=all((parent/'runtime'/f.name).exists() and sha256(parent/'runtime'/f.name)==sha256(f) for f in training_files)
    restore_optimizer=(continuation and same_rewards and same_curriculum and same_exploration and same_training_code
        and parent_config.get('num_envs')==a.num_envs and parent_config.get('learning_rate')==5e-5)
    preservation=None
    if a.preserve_learning_state_evidence:
        assert a.stage=='S3' and continuation and same_curriculum and same_exploration
        assert parent_config.get('num_envs')==a.num_envs and parent_config.get('learning_rate')==5e-5
        assert frontier_alpha==parent_alpha,'Do not combine this reward preflight with a new command range'
        from check_reward_compatibility import verify_preservation_evidence
        preservation=verify_preservation_evidence(a.preserve_learning_state_evidence,parent,source,
            frontier_alpha,expected_config.migration.yaw_error_scale_rad_s)
        restore_optimizer=True
    critic_warmup=0 if continuation and same_rewards and same_curriculum and same_exploration and same_training_code else 10
    run=ROOT/'runs'/('cmd16_hist6_'+a.stage.lower()+'_ppo_'+time.strftime('%Y%m%d_%H%M%S'))
    runtime=run/'runtime';runtime.mkdir(parents=True)
    for f in source.glob('*.py'):shutil.copy2(f,runtime/f.name)
    shutil.copytree(source/'assets',runtime/'assets')
    shutil.copytree(source/'tests',runtime/'tests',ignore=shutil.ignore_patterns('__pycache__'))
    for n in ('teacher_manifest.json','baseline_metrics.json'):shutil.copy2(parent/n,run/n)
    if gate_source:shutil.copy2(gate_source,run/'parent_acceptance.json')
    if selection:atomic_json(run/'parent_checkpoint_selection.json',selection)
    if frontier_gate:atomic_json(run/'frontier_progression.json',frontier_gate)
    if preservation:shutil.copy2(a.preserve_learning_state_evidence,run/'learning_state_compatibility.json')
    files={str(f.relative_to(runtime)):sha256(f) for f in runtime.rglob('*') if f.is_file()}
    atomic_json(run/'runtime_manifest.json',{'files':files,'created_unix':time.time()})
    prefs=read_json(ROOT/'results/legged_gym/training_preferences.json')
    start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    deadline=(start+datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
    cmd=[sys.executable,'-u',str(runtime/'train_ppo.py'),'--run',str(run),'--teacher',str(run/'teacher_manifest.json'),
        '--resume',cp['path'],'--deadline',deadline,'--stage',a.stage,'--num-envs',str(a.num_envs),
        '--critic-warmup-updates',str(critic_warmup)]
    if restore_optimizer:cmd.append('--restore-optimizer')
    if a.stage=='S3':cmd+=['--frontier-alpha',str(frontier_alpha)]
    if a.yaw_error_scale is not None:cmd+=['--yaw-error-scale',str(a.yaw_error_scale)]
    if a.require_normal_completion:
        assert read_json(ROOT/'results/cmd16_hist6/current_migration.json',{}).get('run')==str(parent)
        assert not (parent/'STOP').exists()
    atomic_json(run/'launch.json',{'command':cmd,'deadline':deadline,'preferences':prefs,'parent':cp,'stage':a.stage+'_PPO',
        'frontier_alpha':frontier_alpha,'parent_frontier_alpha':parent_alpha,'automatic_resume':False,
        'learning_state_preservation':preservation})
    with (run/'train.log').open('w') as log:training=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    until=time.time()+150
    while time.time()<until and not (run/'status.json').exists():
        if training.poll() is not None:raise RuntimeError('PPO failed to start: '+str(run))
        time.sleep(.5)
    if not (run/'status.json').exists():
        training.terminate()
        raise RuntimeError('PPO startup did not produce a status record')
    atomic_json(ROOT/'runs/current.json',{'run_dir':str(run)})
    atomic_json(ROOT/'results/cmd16_hist6/current_migration.json',{'run':str(run),'stage':a.stage+'_PPO','launcher_pid':os.getpid()})
    atomic_json(ROOT/'results/legged_gym/active_training_plan.json',{'run':run.name,'run_dir':str(run),'stage':a.stage+'_PPO',
        'status':'training','deadline':deadline,'headline':'学生独立策略训练',
        'description':'完整目标控制，独立PPO采样，无教师/参考执行；独立评估与视频共用实际回放。',
        'automatic_resume':False,'progress_interval_s':180,'updated_unix':time.time()})
    watch=[sys.executable,'-u',str(runtime/'watch_migration.py'),'--run',str(run),'--stage',a.stage,'--interval',str(prefs['video_interval_s'])]
    with (run/'monitor.log').open('w') as log:monitor=subprocess.Popen(watch,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(run/'processes.json',{'training_pid':training.pid,'monitor_pid':monitor.pid,'supervisor_pid':os.getpid()})
    print(json.dumps({'run':str(run),'deadline':deadline,'training_pid':training.pid,'monitor_pid':monitor.pid}),flush=True)
    code=training.wait()
    try:monitor_code=monitor.wait(timeout=600)
    except subprocess.TimeoutExpired:monitor_code=None
    atomic_json(run/'completion.json',{'training_exit_code':code,'monitor_exit_code':monitor_code,'completed_unix':time.time(),
        'deadline':deadline,'automatic_resume':False})
    # Check all frozen runtime files before archiving and verify their archived bytes.
    for name,digest in files.items():assert sha256(runtime/name)==digest
    archive=ROOT/'results/cmd16_hist6'/(run.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(run,arcname=run.name)
    final=read_json(run/'checkpoint.json')
    checks={'runtime/'+name:digest for name,digest in files.items()}
    checks.update({name:sha256(run/name) for name in ('checkpoint.json','completion.json','config.json','runtime_manifest.json')})
    checks['checkpoints/'+Path(final['path']).name]=final['sha256']
    with tarfile.open(archive,'r:gz') as tar:
        for name,digest in checks.items():assert hashlib.sha256(tar.extractfile(run.name+'/'+name).read()).hexdigest()==digest
    atomic_json(run/'backup_receipt.json',{'archive':str(archive),'sha256':sha256(archive),'verified_files':len(checks),
        'final_checkpoint_and_frozen_runtime_verified':True,'created_unix':time.time()})


if __name__=='__main__':main()
