import json,os,signal,time,shutil,datetime,subprocess,hashlib
from pathlib import Path
p=Path('/root/gpufree-data/projects/sable-isaacgym');s=p/'deployment/reference_v2_20261028'
r=json.loads((s/'launch_receipt.json').read_text());old=Path(r['bank']);d=json.loads((old/'teacher_bank.json').read_text())
assert d['status']=='stopped_at_deadline'
end=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(seconds=2400)
proc=Path('/proc')/str(d['pid'])
assert not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:command=(process/'cmdline').read_bytes().decode(errors='replace')
 except OSError:continue
 assert not ('teacher_bank_core.py' in command and str(p/'runs') in command),'Another trainer is running'
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
b=p/'runs'/('teacher_bank_fullref_referencev2_hold40_'+stamp);b.mkdir();shutil.copytree(old/'runtime',b/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
status=d
cmd=list(r['command']);
if '--resume-bank' in cmd:
 idx=cmd.index('--resume-bank');del cmd[idx:idx+2]
cmd[1]=str(b/'runtime/teacher_bank_core.py');cmd[cmd.index('--bank')+1]=str(b);cmd[cmd.index('--deadline')+1]=end.isoformat();cmd+=['--resume-bank',str(old)]
if '--allow-objective-change' in cmd:cmd.remove('--allow-objective-change')
v=Path('/root/gpufree-data/envs/sable-isaacgym-py38')
env=dict(os.environ,SABLE_GYM_ROOT=str(p),SABLE_GYM_ENV=str(v),PYTHONUNBUFFERED='1',PYTHONPATH=str(b/'runtime'),TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
env['PATH']=str(v/'bin')+':'+env.get('PATH','');env['LD_LIBRARY_PATH']=str(v/'lib')+':'+str(v/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
assert all(hashlib.sha256(f.read_bytes()).hexdigest()==hashlib.sha256((b/'runtime'/f.relative_to(old/'runtime')).read_bytes()).hexdigest() for f in (old/'runtime').rglob('*') if f.is_file() and '__pycache__' not in f.parts)
with (b/'train.log').open('w') as f:child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
with (b/'watchdog.log').open('w') as f:watch=subprocess.Popen([str(v/'bin/python'),str(b/'runtime/watch_teacher_bank.py'),'--bank',str(b),'--pid',str(child.pid),'--deadline',end.isoformat(),'--final-grace-s','120'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r.update(bank=str(b),pid=child.pid,watchdog_pid=watch.pid,deadline=end.isoformat(),command=cmd,source_bank=str(old),source_iteration=status['iteration'],status='starting',fresh_weights=False,total_round_minutes=40,turn_curriculum_degrees=[10,20,30,45,60,90])
for f in [b/'launch_receipt.json',s/'launch_receipt.json']:f.write_text(json.dumps(r,indent=2))
print(json.dumps(r),flush=True)
