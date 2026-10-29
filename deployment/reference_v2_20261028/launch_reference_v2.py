"""Default: CPU preflight only. --start explicitly enables remote GPU work."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    parser.add_argument('--duration',type=int,default=1200);parser.add_argument('--envs-per-teacher',type=int,default=1024)
    a=parser.parse_args();stage=Path(__file__).resolve().parent
    project=Path('/root/gpufree-data/projects/sable-isaacgym');venv=Path('/root/gpufree-data/envs/sable-isaacgym-py38')
    python=str(venv/'bin/python')
    if project/'deployment' not in stage.parents:raise RuntimeError('Run the staged remote launcher, not a local training job')
    manifest=json.loads((stage/'package_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        path=(stage/name).resolve()
        if stage not in path.parents:raise ValueError('Invalid manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Package changed: '+name)
    env=dict(os.environ,SABLE_GYM_ROOT=str(project),SABLE_GYM_ENV=str(venv),PYTHONUNBUFFERED='1')
    env['PATH']=str(venv/'bin')+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(venv/'lib')+os.pathsep+str(venv/'lib/python3.8/site-packages/torch/lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    env['TORCH_EXTENSIONS_DIR']='/root/gpufree-data/cache/sable-isaacgym/torch_extensions'
    cpu=dict(env,CUDA_VISIBLE_DEVICES='')
    result=subprocess.run([python,str(stage/'runtime/check_reference_v2.py')],env=cpu,capture_output=True,text=True,timeout=60)
    (stage/'remote_cpu_check.log').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:raise RuntimeError('CPU check failed: '+result.stderr[-3000:])
    receipt={'status':'prepared_no_gpu_started','cpu_tests_passed':True,'created_unix':time.time(),'profile':'reference_local_v2'}
    (stage/'readiness.json').write_text(json.dumps(receipt,indent=2))
    if not a.start:print(json.dumps(receipt));return
    if a.duration<=0 or a.envs_per_teacher<32:raise ValueError('Invalid duration or environment count')
    # Do not overlap another training bank, even when the dashboard pointer is stale.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:command=(proc/'cmdline').read_bytes().decode(errors='replace')
        except (OSError,ProcessLookupError):continue
        if 'teacher_bank_core.py' in command and str(project) in command:raise RuntimeError('Another teacher trainer is active')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    if gpu.returncode or not gpu.stdout.strip():raise RuntimeError('GPU is not available; no training started')
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    bank=project/'runs'/('teacher_bank_fullref_referencev2_'+stamp);bank.mkdir()
    shutil.copytree(stage/'runtime',bank/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(stage/'reference',bank/'reference')
    env['PYTHONPATH']=str(bank/'runtime')+os.pathsep+env.get('PYTHONPATH','')
    common=[python,str(bank/'runtime/teacher_bank_core.py'),'--reference',str(bank/'reference'),
        '--reward-cleanup-profile','reference_local_v2','--control-mode','reference_plus_residual',
        '--reference-strength','1','--time-scale','1','--offset-limit-deg','20','--imitation-scale','0',
        '--exploration-std','.08','--minimum-action-std','.02']
    with (bank/'ppo_smoke.log').open('w') as log:
        smoke=subprocess.run(common+['--bank',str(bank/'smoke'),'--envs-per-teacher','32','--smoke','--duration','120'],
                             env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if smoke.returncode:raise RuntimeError('GPU smoke failed; formal training NOT started: '+str(bank/'ppo_smoke.log'))
    state=json.loads((bank/'smoke/teacher_bank.json').read_text())
    if state['status']!='smoke_complete' or state['iteration']!=1:raise RuntimeError('GPU smoke did not complete one update')
    deadline=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(seconds=a.duration)
    command=common+['--bank',str(bank),'--envs-per-teacher',str(a.envs_per_teacher),'--deadline',deadline.isoformat()]
    with (bank/'train.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with (bank/'watchdog.log').open('w') as log:
        watcher=subprocess.Popen([python,str(bank/'runtime/watch_teacher_bank.py'),'--bank',str(bank),'--pid',str(process.pid),
             '--deadline',deadline.isoformat(),'--final-grace-s','120'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(status='starting',bank=str(bank),pid=process.pid,watchdog_pid=watcher.pid,deadline=deadline.isoformat(),
        fresh_weights=True,smoke_weights_reused=False,command=command)
    (bank/'launch_receipt.json').write_text(json.dumps(receipt,indent=2));(stage/'launch_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':main()
