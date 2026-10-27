"""One bounded S3 continuation after a normal completed round.

Default uses the parent's exact runtime. An explicitly supplied prepared runtime
must have a sibling manifest and passed preflight; it may reset optimizer state.
"""
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
from acceptance import assess,number

ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
OLD={'forward09','forward105','forward12','stop','start_stop','speed_switch'}


def task_progress(report,baseline):
    """Development ranking against unchanged gates, never task acceptance.

    Periodic reports have one seed. Only the missing-three-seed failure is
    excluded from this ranking; the acceptance function itself is unchanged.
    """
    result=assess('S3',[report],baseline)
    failures=[f for f in result['failures'] if f!='exactly_three_declared_independent_seeds_required']
    limits=baseline['new_capability_limits'];cadence=0;distance=0.
    for row in report['scenarios']:
        if row['name'] in OLD:continue
        cadence+=sum(number(f) and abs(f-1.)<=.15 for f in (row.get('steady_frequency_per_foot_hz') or []))
        for field,limit in [('xy_rmse_m_s',limits['xy_rmse_m_s']),('yaw_rmse_rad_s',limits['yaw_rmse_rad_s'])]:
            value=row.get(field);distance+=value/limit if number(value) else 100.
        actual=row.get('mean_actual_xyz_command_units') or [];desired=row.get('mean_command') or []
        if len(actual)!=3 or len(desired)!=3 or not all(number(x)for x in actual+desired):distance+=100.;continue
        for axis in range(3):
            if abs(desired[axis])>=(.02 if axis<2 else .1):
                gain=actual[axis]/desired[axis]
                distance+=max(limits['mean_command_gain_min']-gain,0.) if number(gain) else 100.
    return dict(single_report_requirement_failures=failures,task_failure_count=len(failures),
        novel_feet_with_correct_cadence=cadence,normalized_motion_distance=distance,
        three_seed_acceptance=False)


def candidates(parent):
    baseline=read_json(parent/'baseline_metrics.json');limits=baseline['new_capability_limits']
    warmup=read_json(parent/'config.json').get('critic_warmup_updates',0)
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
            evaluation=str(path),evaluation_sha256=sha256(path),all_scenarios_survival=stable,old_tracking=tracking,
            has_actor_updates=report['iteration']>warmup,
            task_progress=task_progress(report,baseline)))
    usable=[r for r in rows if r['all_scenarios_survival'] and r['old_tracking'] and r['has_actor_updates']]
    if not usable:raise RuntimeError('No stable independently evaluated continuation parent')
    # Preserve a learned novel skill when a later stable checkpoint forgets it.
    # The exact optimizer state still comes from the selected checkpoint.
    def rank(row):
        score=row['task_progress']
        return (score['task_failure_count'],-score['novel_feet_with_correct_cadence'],
            score['normalized_motion_distance'],-row['iteration'])
    return min(usable,key=rank),sorted(rows,key=lambda r:r['iteration'])


def frontier_candidates(parent, directory):
    """Select a measured curriculum state, advancing only after its gate passes."""
    from evaluate_frontier import assess_frontier
    cfg=read_json(parent/'config.json');alpha=cfg['command_frontier']['training_alpha']
    baseline=read_json(parent/'baseline_metrics.json');warmup=cfg.get('critic_warmup_updates',0)
    paths=set(directory.rglob('gate_alpha_*.json'))|set(directory.rglob('frontier_gate_*.json'))
    rows=[]
    for path in paths:
        stored=read_json(path)
        if stored.get('training_alpha')!=alpha:continue
        sources=stored['sources']
        for item in sources.values():assert sha256(item['path'])==item['sha256']
        assert sources['baseline']['sha256']==sha256(parent/'baseline_metrics.json')
        endpoint=read_json(sources['endpoint']['path']);frontier=read_json(sources['frontier']['path'])
        cp=Path(endpoint['checkpoint'])
        if cp.parent.resolve()!=parent/'checkpoints':continue
        assert sha256(cp)==endpoint['checkpoint_sha256']
        verified=assess_frontier(frontier,endpoint,baseline,alpha)
        assert stored['passed']==verified['passed'] and stored['failures']==verified['failures']
        if endpoint['iteration']<=warmup:continue
        # Structural errors and lost old-domain behavior never qualify even for
        # continuation at the same alpha. Failed task gains/cadence may learn.
        families={'reverse','left','right','turn_left','turn_right'}
        if any(f.split(':',1)[0]not in families for f in verified['failures']):continue
        measured=verified['selected_rows']
        if len(measured)!=5 or any(r.get('survival_fraction',0)<.9 for r in measured):continue
        score=task_progress(endpoint,baseline)
        rows.append(dict(iteration=endpoint['iteration'],checkpoint=str(cp),sha256=endpoint['checkpoint_sha256'],
            evaluation=sources['endpoint']['path'],evaluation_sha256=sources['endpoint']['sha256'],
            frontier_gate=str(path.resolve()),frontier_gate_sha256=sha256(path),
            frontier_failures=verified['failures'],frontier_passed=verified['passed'],
            training_alpha=alpha,next_alpha=min(1.,alpha+.125)if verified['passed']else alpha,
            task_progress=score))
    if not rows:raise RuntimeError('No independently measured frontier candidate preserves old-domain gates and frontier survival')
    def rank(row):
        score=row['task_progress']
        return (len(row['frontier_failures']),score['task_failure_count'],
                score['normalized_motion_distance'],-row['iteration'])
    return min(rows,key=rank),sorted(rows,key=lambda r:r['iteration'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--source-runtime',type=Path)
    p.add_argument('--frontier-dir',type=Path)
    p.add_argument('--check-only',action='store_true');a=p.parse_args();parent=a.parent_run.resolve()
    assert read_json(parent/'config.json')['stage']=='S3'
    source=parent/'runtime'
    if a.source_runtime:
        source=a.source_runtime.resolve()
        manifest=read_json(source.parent/'runtime_manifest.json')
        preflight=read_json(source.parent/'preflight.json')
        assert preflight['passed'] and preflight['stage']=='S3'
        for name,digest in manifest['files'].items():assert sha256(source/name)==digest
        assert preflight['runtime_manifest_sha256']==sha256(source.parent/'runtime_manifest.json')
    if a.check_only:
        selected,rows=frontier_candidates(parent,a.frontier_dir)if a.frontier_dir else candidates(parent)
        print(json.dumps(dict(selected=selected,candidates=rows)));return
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
    selected,rows=frontier_candidates(parent,a.frontier_dir)if a.frontier_dir else candidates(parent)
    with tarfile.open(receipt['archive'],'r:gz') as tar:
        name=parent.name+'/checkpoints/'+Path(selected['checkpoint']).name
        assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==selected['sha256']
    # Recheck a prepared source after waiting; mutable development files cannot
    # leak into the new frozen round. The launcher checks optimizer compatibility.
    if a.source_runtime:
        for name,digest in manifest['files'].items():assert sha256(source/name)==digest
    cfg=read_json(parent/'config.json')
    command=[sys.executable,'-u',str(source/'launch_ppo.py'),'--parent-run',str(parent),
        '--checkpoint',selected['checkpoint'],'--stage','S3','--num-envs',str(cfg['num_envs'])]
    if a.frontier_dir:
        command+=['--frontier-alpha',str(selected['next_alpha']),'--require-normal-completion']
        if selected['frontier_passed']:
            command+=['--frontier-evidence',selected['frontier_gate']]
    with output.with_suffix('.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    choice=dict(selected=selected,candidates=rows,archive_selected_bytes_verified=True,
        rule='Require actor updates after critic warmup, all14 survival>=90% and six old-domain XY/yaw limits; rank by fewest unchanged S3 requirement failures, most novel feet at correct cadence, lowest normalized motion error/response shortfall, then latest iteration',
        scope='Continuation selection only; three-seed task acceptance remains required')
    if a.frontier_dir:
        choice['rule']='Require unchanged old-domain gates, all five actual frontier paths surviving>=90%, and actor updates. Rank by fewest current-frontier failures then full endpoint failures/error. Advance at most.125 only when the selected measured frontier gate passed; otherwise continue the same actual command range.'
    atomic_json(output,dict(status='launching',pid=process.pid,command=command,choice=choice,created_unix=time.time()))
    until=time.time()+180
    while time.time()<until:
        if process.poll() is not None:raise RuntimeError('Continuation launcher exited before startup')
        current=read_json(ROOT/'results/cmd16_hist6/current_migration.json',{})
        if current.get('run')!=str(parent) and current.get('launcher_pid')==process.pid:
            run=Path(current['run']);config=read_json(run/'config.json')
            if not a.source_runtime:assert config['optimizer_restore']
            atomic_json(run/'parent_checkpoint_selection.json',choice)
            atomic_json(output,dict(status='started',pid=process.pid,run=str(run),command=command,choice=choice,
                optimizer_restored=config['optimizer_restore'],source_runtime=str(source),created_unix=time.time()));return
        time.sleep(1)
    raise RuntimeError('Continuation startup not confirmed')


if __name__=='__main__':main()
