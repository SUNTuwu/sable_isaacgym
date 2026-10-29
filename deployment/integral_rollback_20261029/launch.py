import json,os,signal,time,shutil,datetime,subprocess
from pathlib import Path
p=Path('/root/gpufree-data/projects/sable-isaacgym');s=p/'deployment/reference_v2_20261028'
r=json.loads((s/'launch_receipt.json').read_text());old=Path(r['bank']);d=json.loads((old/'teacher_bank.json').read_text())
assert d['status'] in ('training','failed')
end=datetime.datetime.fromisoformat(r['deadline'])
source=Path(r['source_bank']);assert source.name=='teacher_bank_fullref_referencev2_freshfeet4_20261029_032315'
assert json.loads((source/'teacher_bank.json').read_text())['iteration']==121
if d['status']=='training':assert str(old) in Path('/proc',str(d['pid']),'cmdline').read_bytes().decode()
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
b=p/'runs'/('teacher_bank_fullref_referencev2_terminal_integral_'+stamp);b.mkdir();shutil.copytree(old/'runtime',b/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
patch=p/'deployment/integral_rollback_20261029'
for name in ['teacher_bank_core.py','teacher_reference_rewards_v2.py','teacher_turn_curriculum.py','teacher_turn_boundary.py']:shutil.copy2(patch/name,b/'runtime'/name)
reference=Path(r['command'][r['command'].index('--reference')+1]);shutil.copytree(reference,b/'reference')
v=Path('/root/gpufree-data/envs/sable-isaacgym-py38')
env=dict(os.environ,SABLE_GYM_ROOT=str(p),SABLE_GYM_ENV=str(v),PYTHONUNBUFFERED='1',PYTHONPATH=str(b/'runtime'),TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
env['PATH']=str(v/'bin')+':'+env.get('PATH','');env['LD_LIBRARY_PATH']=str(v/'lib')+':'+str(v/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
for name in ['check_reference_v2.py','teacher_turn_boundary.py']:
 result=subprocess.run([str(v/'bin/python'),str(b/'runtime'/name)],env=dict(env,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60)
 (b/(name+'.log')).write_text(result.stdout+'\n'+result.stderr);assert result.returncode==0,result.stderr
smoke_cmd=list(r['command'])
for flag in ['--resume-bank','--bank','--envs-per-teacher','--deadline']:
 if flag in smoke_cmd:
  index=smoke_cmd.index(flag);del smoke_cmd[index:index+2]
smoke_cmd[1]=str(b/'runtime/teacher_bank_core.py');smoke_cmd+=['--bank',str(b/'smoke'),'--envs-per-teacher','32','--smoke','--duration','120']
with (b/'ppo_smoke.log').open('w') as log:result=subprocess.run(smoke_cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert result.returncode==0,'GPU smoke failed; existing trainer is still running'
assert json.loads((b/'smoke/teacher_bank.json').read_text())['status']=='smoke_complete'
if d['status']=='training':
 os.kill(d['pid'],signal.SIGTERM)
 for _ in range(120):
  status=json.loads((old/'teacher_bank.json').read_text())
  proc=Path('/proc')/str(d['pid'])
  exited=not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
  if exited:break
  time.sleep(.5)
 else:raise RuntimeError('Parent did not stop cleanly; refusing overlap')
 assert status['status']=='stopped_requested',status['status']
else:status=d
cmd=list(r['command']);
if '--resume-bank' in cmd:
 idx=cmd.index('--resume-bank');del cmd[idx:idx+2]
cmd[1]=str(b/'runtime/teacher_bank_core.py');cmd[cmd.index('--bank')+1]=str(b);cmd[cmd.index('--deadline')+1]=end.isoformat();cmd+=['--resume-bank',str(source)];cmd[cmd.index('--minimum-action-std')+1]='.02'
if '--allow-objective-change' in cmd:cmd.remove('--allow-objective-change')
v=Path('/root/gpufree-data/envs/sable-isaacgym-py38')
env=dict(os.environ,SABLE_GYM_ROOT=str(p),SABLE_GYM_ENV=str(v),PYTHONUNBUFFERED='1',PYTHONPATH=str(b/'runtime'),TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
env['PATH']=str(v/'bin')+':'+env.get('PATH','');env['LD_LIBRARY_PATH']=str(v/'lib')+':'+str(v/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
with (b/'train.log').open('w') as f:child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
with (b/'watchdog.log').open('w') as f:watch=subprocess.Popen([str(v/'bin/python'),str(b/'runtime/watch_teacher_bank.py'),'--bank',str(b),'--pid',str(child.pid),'--deadline',end.isoformat(),'--final-grace-s','120'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r.update(bank=str(b),pid=child.pid,watchdog_pid=watch.pid,deadline=end.isoformat(),command=cmd,source_bank=str(source),source_iteration=121,discarded_bank=str(old),status='starting',fresh_weights=False,total_round_minutes=40,modified_teachers='all8 terminal-only integral boundary; turns +/-30deg/s',integral_boundary='translations30deg, turns0.20m; settled at natural end only')
for f in [b/'launch_receipt.json',s/'launch_receipt.json']:f.write_text(json.dumps(r,indent=2))
print(json.dumps(r),flush=True)
