"""Fresh frozen DAgger round after a checked parent stop/save, with monitoring."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from run_io import atomic_json,sha256,read_json
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')


def main():
    from training_hold import require_training_enabled
    require_training_enabled()
    p=argparse.ArgumentParser();p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    parent=a.parent_run.resolve();end=time.time()+180
    while time.time()<end:
        state=read_json(parent/'status.json',{})
        if state.get('status') in ('stopped_requested','stopped_at_deadline','iterations_complete') and (parent/'backup_receipt.json').exists():break
        time.sleep(2)
    else:raise RuntimeError('Parent has not completed save/archive; refusing a new round')
    receipt=read_json(parent/'backup_receipt.json');assert sha256(receipt['archive'])==receipt['sha256']
    cp=read_json(parent/'checkpoint.json');assert sha256(cp['path'])==cp['sha256']
    source=Path(__file__).resolve().parent;run=ROOT/'runs'/('cmd16_hist6_s2_dagger_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir();runtime=run/'runtime';runtime.mkdir()
    for f in source.glob('*.py'):shutil.copy2(f,runtime/f.name)
    shutil.copytree(source/'assets',runtime/'assets');shutil.copytree(source/'tests',runtime/'tests',ignore=shutil.ignore_patterns('__pycache__'))
    for n in ('teacher_manifest.json','baseline_metrics.json'):shutil.copy2(parent/n,run/n)
    atomic_json(run/'runtime_manifest.json',{'files':{str(f.relative_to(runtime)):sha256(f) for f in runtime.rglob('*') if f.is_file()},'created_unix':time.time()})
    prefs=read_json(ROOT/'results/legged_gym/training_preferences.json');start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
    cmd=[sys.executable,'-u',str(runtime/'train_dagger.py'),'--run',str(run),'--teacher',str(run/'teacher_manifest.json'),'--resume',cp['path'],'--offline-dataset',str(a.dataset),'--deadline',deadline,'--num-envs','256','--initial-student-fraction','.25']
    atomic_json(run/'launch.json',{'command':cmd,'deadline':deadline,'preferences':prefs,'parent':cp,'stage':'S2_DAgger','automatic_resume':False})
    with (run/'train.log').open('w') as log:training=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    until=time.time()+120
    while time.time()<until and not (run/'status.json').exists():
        if training.poll() is not None:raise RuntimeError('DAgger failed to start: '+str(run))
        time.sleep(.5)
    assert (run/'status.json').exists()
    atomic_json(ROOT/'runs/current.json',{'run_dir':str(run)})
    atomic_json(ROOT/'results/cmd16_hist6/current_migration.json',{'run':str(run),'stage':'S2_DAgger','launcher_pid':os.getpid()})
    atomic_json(ROOT/'results/legged_gym/active_training_plan.json',{'run':run.name,'run_dir':str(run),'stage':'S2_DAgger','status':'training','deadline':deadline,'headline':'学生闭环数据聚合','description':'保留启动和学生访问状态，查询完整教师目标；混合执行用于监督学习，不当作PPO数据。独立评估不使用教师或参考。','automatic_resume':False,'progress_interval_s':180,'updated_unix':time.time()})
    watch_cmd=[sys.executable,'-u',str(runtime/'watch_migration.py'),'--run',str(run),'--stage','S2','--interval',str(prefs['video_interval_s'])]
    with (run/'monitor.log').open('w') as log:monitor=subprocess.Popen(watch_cmd,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(run/'processes.json',{'training_pid':training.pid,'monitor_pid':monitor.pid,'supervisor_pid':os.getpid()})
    print(json.dumps({'run':str(run),'deadline':deadline,'training_pid':training.pid,'monitor_pid':monitor.pid}),flush=True)
    code=training.wait()
    try:monitor_code=monitor.wait(timeout=600)
    except subprocess.TimeoutExpired:monitor_code=None
    atomic_json(run/'completion.json',{'training_exit_code':code,'monitor_exit_code':monitor_code,'completed_unix':time.time(),'deadline':deadline,'automatic_resume':False})
    archive=ROOT/'results/cmd16_hist6'/(run.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(run,arcname=run.name)
    with tarfile.open(archive,'r:gz') as tar:
        for name in ['checkpoint.json','completion.json','config.json','runtime_manifest.json']:
            assert hashlib.sha256(tar.extractfile(run.name+'/'+name).read()).hexdigest()==sha256(run/name)
        final=read_json(run/'checkpoint.json');member=run.name+'/checkpoints/'+Path(final['path']).name
        assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==final['sha256']
    atomic_json(run/'backup_receipt.json',{'archive':str(archive),'sha256':sha256(archive),'final_checkpoint_and_metadata_verified':True,'created_unix':time.time()})


if __name__=='__main__':main()
