import json,os,signal,time,shutil,datetime,subprocess
from pathlib import Path
p=Path('/root/gpufree-data/projects/sable-isaacgym');d=p/'deployment/night_20261029';yd=p/'deployment/yaw_sine_20261029'
r=json.loads((yd/'launch_receipt.json').read_text());old=Path(r['run']);pid=r['pid']
b=p/'runs'/('yaw_sine_night_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));b.mkdir()
shutil.copytree(old/'runtime',b/'runtime',ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(d/'yaw_sine_teacher.py',b/'runtime/yaw_sine_teacher.py')
evaluation=(yd/'evaluate_yaw.py').read_text()
evaluation=evaluation.replace("    cfg=student_config", "    set_task(meta['amplitude_deg'],meta['liftoff_budget_per_foot_per_episode'])\n    global EPISODE\n    EPISODE=SETTLE+2*(2*math.pi*math.radians(meta['amplitude_deg'])/math.radians(10))\n    cfg=student_config")
evaluation=evaluation.replace("bool((counts<=1).all())", "bool((counts<=meta['liftoff_budget_per_foot_per_episode']).all())")
(b/'runtime/evaluate_yaw.py').write_text(evaluation)
v=Path('/root/gpufree-data/envs/sable-isaacgym-py38');env=dict(os.environ,SABLE_GYM_ROOT=str(p),SABLE_GYM_ENV=str(v),PYTHONPATH=str(b/'runtime'),PYTHONUNBUFFERED='1',TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
env['PATH']=str(v/'bin')+':'+env.get('PATH','');env['LD_LIBRARY_PATH']=str(v/'lib')+':'+str(v/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
cmd=[str(v/'bin/python'),str(b/'runtime/yaw_sine_teacher.py')]
with (b/'smoke.log').open('w') as log:result=subprocess.run(cmd+['--run',str(b/'smoke'),'--envs','32','--smoke','--resume',str(old)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
assert result.returncode==0,'yaw resume smoke failed'
if Path('/proc',str(pid),'cmdline').exists():
 assert str(old) in Path('/proc',str(pid),'cmdline').read_bytes().decode();os.kill(pid,signal.SIGTERM)
for _ in range(120):
    proc=Path('/proc',str(pid))
    if not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
    time.sleep(.5)
else:raise RuntimeError('yaw parent did not stop')
deadline=datetime.datetime(2026,10,29,8,30,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
with (b/'train.log').open('w') as log:child=subprocess.Popen(cmd+['--run',str(b),'--envs','512','--resume',str(old),'--deadline',str(deadline)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with (b/'watchdog.log').open('w') as log:watch=subprocess.Popen(['python3',str(yd/'watch.py'),str(b),str(child.pid)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(run=str(b),pid=child.pid,watchdog_pid=watch.pid,source_run=str(old),deadline_unix=deadline,status='starting')
for f in [yd/'launch_receipt.json',b/'launch_receipt.json',d/'yaw_receipt.json']:f.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
