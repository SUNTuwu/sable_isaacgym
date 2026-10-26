"""Launch one authorized frozen PPO round after a verified parent archive."""
import argparse
import datetime
import hashlib
import json
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
    a=p.parse_args()
    parent=a.parent_run.resolve()
    until=time.time()+1800
    while time.time()<until:
        state=read_json(parent/'status.json',{})
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
    parent_config=read_json(parent/'config.json',{})
    continuation=parent_config.get('stage')==a.stage and 'on-policy PPO' in parent_config.get('algorithm','')
    expected_config=student_config(read_json(parent/'teacher_manifest.json')['run'],a.stage,a.num_envs)
    same_rewards=(parent_config.get('reward_weights')==vars(expected_config.rewards.scales)
                  and parent_config.get('velocity_error_scale_m_s')==expected_config.migration.velocity_error_scale_m_s)
    same_rewards &= parent_config.get('unclipped_gait_costs',False)==expected_config.migration.unclipped_gait_costs
    same_curriculum=parent_config.get('moving_start_curriculum',{}).get('enabled',False)==(a.stage=='S3')
    same_exploration=parent_config.get('minimum_action_std',.02)==expected_config.migration.minimum_action_std
    same_exploration &= parent_config.get('novel_command_std_floor',0.)==expected_config.migration.novel_command_std_floor
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
    nontraining={'train_ppo.py','launch_ppo.py','watch_migration.py','export_policy.py','select_regression_checkpoint.py'}
    training_files=[f for f in source.glob('*.py') if f.name not in nontraining
        and not f.name.startswith(('evaluate_','evaluation_','probe_','check_'))]
    same_training_code=all((parent/'runtime'/f.name).exists() and sha256(parent/'runtime'/f.name)==sha256(f) for f in training_files)
    restore_optimizer=(continuation and same_rewards and same_curriculum and same_exploration and same_training_code
        and parent_config.get('num_envs')==a.num_envs and parent_config.get('learning_rate')==5e-5)
    run=ROOT/'runs'/('cmd16_hist6_'+a.stage.lower()+'_ppo_'+time.strftime('%Y%m%d_%H%M%S'))
    runtime=run/'runtime';runtime.mkdir(parents=True)
    for f in source.glob('*.py'):shutil.copy2(f,runtime/f.name)
    shutil.copytree(source/'assets',runtime/'assets')
    shutil.copytree(source/'tests',runtime/'tests',ignore=shutil.ignore_patterns('__pycache__'))
    for n in ('teacher_manifest.json','baseline_metrics.json'):shutil.copy2(parent/n,run/n)
    if gate_source:shutil.copy2(gate_source,run/'parent_acceptance.json')
    if selection:atomic_json(run/'parent_checkpoint_selection.json',selection)
    files={str(f.relative_to(runtime)):sha256(f) for f in runtime.rglob('*') if f.is_file()}
    atomic_json(run/'runtime_manifest.json',{'files':files,'created_unix':time.time()})
    prefs=read_json(ROOT/'results/legged_gym/training_preferences.json')
    start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    deadline=(start+datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
    cmd=[sys.executable,'-u',str(runtime/'train_ppo.py'),'--run',str(run),'--teacher',str(run/'teacher_manifest.json'),
        '--resume',cp['path'],'--deadline',deadline,'--stage',a.stage,'--num-envs',str(a.num_envs),
        '--critic-warmup-updates','0' if continuation and same_rewards and same_curriculum and same_exploration and same_training_code else '10']
    if restore_optimizer:cmd.append('--restore-optimizer')
    atomic_json(run/'launch.json',{'command':cmd,'deadline':deadline,'preferences':prefs,'parent':cp,'stage':a.stage+'_PPO','automatic_resume':False})
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
