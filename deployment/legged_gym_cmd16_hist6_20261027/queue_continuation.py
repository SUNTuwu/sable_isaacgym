"""One bounded, compatible S3 continuation after a normal completed round."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from run_io import atomic_json,read_json,sha256

ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
OLD={'forward09','forward105','forward12','stop','start_stop','speed_switch'}


def candidates(parent):
    baseline=read_json(parent/'baseline_metrics.json');limits=baseline['new_capability_limits']
    rows=[]
    for path in (parent/'evaluations').glob('independent_*.json'):
        report=read_json(path);scenes=report.get('scenarios',[])
        if report.get('suite')!='S3' or len(scenes)!=14 or report.get('horizon_s',0)<16:continue
        if report.get('num_envs',0)<56 or report.get('environments_per_scenario',0)<4:continue
        if report.get('reference_execution_calls',1) or report.get('teacher_control'):continue
        if report.get('teacher_sha256')!=baseline['teacher_sha256']:continue
        stable=all(s.get('survival_fraction',0)>=.9 for s in scenes)
        old=[s for s in scenes if s['name']in OLD]
        def within(value,maximum):
            return isinstance(value,(float,int)) and math.isfinite(value) and 0<=value<=maximum
        tracking=len(old)==6 and all(within(s.get('xy_rmse_m_s'),limits['xy_rmse_m_s'])
            and within(s.get('yaw_rmse_rad_s'),limits['yaw_rmse_rad_s']) for s in old)
        cp=Path(report['checkpoint'])
        if cp.parent.resolve()!=parent/'checkpoints':continue
        if sha256(cp)!=report['checkpoint_sha256']:raise ValueError('Evaluated checkpoint changed')
        rows.append(dict(iteration=report['iteration'],checkpoint=str(cp),sha256=report['checkpoint_sha256'],
            evaluation=str(path),evaluation_sha256=sha256(path),all_scenarios_survival=stable,old_tracking=tracking))
    usable=[r for r in rows if r['all_scenarios_survival'] and r['old_tracking']]
    if not usable:raise RuntimeError('No stable independently evaluated continuation parent')
    return max(usable,key=lambda r:r['iteration']),sorted(rows,key=lambda r:r['iteration'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args();parent=a.parent_run.resolve()
    assert read_json(parent/'config.json')['stage']=='S3'
    if a.check_only:
        selected,rows=candidates(parent);print(json.dumps(dict(selected=selected,candidates=rows)));return
    output=ROOT/'results/cmd16_hist6'/('continuation_'+parent.name+'.json')
    lock=(ROOT/'results/cmd16_hist6/continuation.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    until=time.time()+1800
    while time.time()<until:
        state=read_json(parent/'status.json',{})
        if state.get('status') in ('stopped_requested','failed'):
            atomic_json(output,dict(status='cancelled',reason='Parent explicitly stopped or failed'));return
        if read_json(ROOT/'results/cmd16_hist6/current_migration.json',{}).get('run')!=str(parent):
            atomic_json(output,dict(status='cancelled',reason='Active run changed'));return
        if state.get('status')=='stopped_at_deadline' and (parent/'backup_receipt.json').exists():break
        time.sleep(3)
    else:raise RuntimeError('Parent completion/archive did not become ready')
    final=read_json(parent/'checkpoint.json');report=read_json(parent/'latest_evaluation.json')
    assert report['checkpoint_sha256']==final['sha256'],'Final evaluation missing'
    receipt=read_json(parent/'backup_receipt.json');assert sha256(receipt['archive'])==receipt['sha256']
    selected,rows=candidates(parent)
    with tarfile.open(receipt['archive'],'r:gz') as tar:
        name=parent.name+'/checkpoints/'+Path(selected['checkpoint']).name
        assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==selected['sha256']
    # Launch from the parent's exact immutable runtime, preserving compatible
    # Adam/auxiliary state. Development-only edits cannot leak into this round.
    cfg=read_json(parent/'config.json')
    command=[sys.executable,'-u',str(parent/'runtime/launch_ppo.py'),'--parent-run',str(parent),
        '--checkpoint',selected['checkpoint'],'--stage','S3','--num-envs',str(cfg['num_envs'])]
    with output.with_suffix('.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    choice=dict(selected=selected,candidates=rows,archive_selected_bytes_verified=True,
        rule='Latest checkpoint with all14 scenario survival>=90% and six old-domain XY/yaw errors within existing S3 limits',
        scope='Continuation selection only; three-seed task acceptance remains required')
    atomic_json(output,dict(status='launching',pid=process.pid,command=command,choice=choice,created_unix=time.time()))
    until=time.time()+180
    while time.time()<until:
        if process.poll() is not None:raise RuntimeError('Continuation launcher exited before startup')
        current=read_json(ROOT/'results/cmd16_hist6/current_migration.json',{})
        if current.get('run')!=str(parent) and current.get('launcher_pid')==process.pid:
            run=Path(current['run']);config=read_json(run/'config.json');assert config['optimizer_restore']
            atomic_json(run/'parent_checkpoint_selection.json',choice)
            atomic_json(output,dict(status='started',pid=process.pid,run=str(run),command=command,choice=choice,
                optimizer_restored=True,created_unix=time.time()));return
        time.sleep(1)
    raise RuntimeError('Continuation startup not confirmed')


if __name__=='__main__':main()
