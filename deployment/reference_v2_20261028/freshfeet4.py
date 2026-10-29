import json,os,time,shutil,datetime,subprocess,hashlib
from pathlib import Path
p=Path('/root/gpufree-data/projects/sable-isaacgym');s=p/'deployment/reference_v2_20261028'
r=json.loads((s/'launch_receipt.json').read_text());old=Path(r['bank']);d=json.loads((old/'teacher_bank.json').read_text());assert d['status']=='stopped_at_deadline'
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:cmdline=(proc/'cmdline').read_bytes().decode(errors='replace')
 except OSError:continue
 assert not ('teacher_bank_core.py' in cmdline and str(p/'runs') in cmdline),'Another trainer is active'
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
b=p/'runs'/('teacher_bank_fullref_referencev2_freshfeet4_'+stamp);b.mkdir()
shutil.copytree(old/'runtime',b/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
ref=Path(r['command'][r['command'].index('--reference')+1]);shutil.copytree(ref,b/'reference')
f=b/'runtime/teacher_reference_rewards_v2.py';text=f.read_text();assert "'foot_local':1.5" in text
text=text.replace("'foot_local':1.5","'foot_local':4.").replace("'v2_foot_local':1.5*","'v2_foot_local':4*").replace("'1.5*mean(exp(-sum((error/[0.040,0.040,0.020])^2)))","'4*mean(exp(-sum((error/[0.040,0.040,0.020])^2)))")
f.write_text(text)
f=b/'runtime/check_reference_v2.py';text=f.read_text().replace('float(sum(r.values())),9.5','float(sum(r.values())),12.');f.write_text(text)
f=b/'runtime/teacher_bank_core.py';text=f.read_text();needle='for policy in policies:policy.minimum_std=a.minimum_action_std;policy.maximum_std=.6';assert needle in text
text=text.replace(needle,needle+';policy.log_std.data.fill_(math.log(a.exploration_std))');f.write_text(text)
v=Path('/root/gpufree-data/envs/sable-isaacgym-py38');env=dict(os.environ,SABLE_GYM_ROOT=str(p),SABLE_GYM_ENV=str(v),PYTHONUNBUFFERED='1',PYTHONPATH=str(b/'runtime'),TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
env['PATH']=str(v/'bin')+':'+env.get('PATH','');env['LD_LIBRARY_PATH']=str(v/'lib')+':'+str(v/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
for name in ['check_reference_v2.py','teacher_turn_boundary.py']:
 result=subprocess.run([str(v/'bin/python'),str(b/'runtime'/name)],env=dict(env,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60)
 (b/(name+'.log')).write_text(result.stdout+'\n'+result.stderr);assert result.returncode==0,result.stderr
cmd=list(r['command'])
for flag in ['--resume-bank','--deadline','--bank','--envs-per-teacher']:
 if flag in cmd:
  i=cmd.index(flag);del cmd[i:i+2]
if '--allow-objective-change' in cmd:cmd.remove('--allow-objective-change')
cmd[1]=str(b/'runtime/teacher_bank_core.py');cmd[cmd.index('--reference')+1]=str(b/'reference')
with (b/'ppo_smoke.log').open('w') as log:
 result=subprocess.run(cmd+['--bank',str(b/'smoke'),'--envs-per-teacher','32','--smoke','--duration','120'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert result.returncode==0,'GPU smoke failed'
smoke=json.loads((b/'smoke/teacher_bank.json').read_text());assert smoke['status']=='smoke_complete' and smoke['initial_iteration']==0 and smoke['iteration']==1
end=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(seconds=2400)
cmd+=['--bank',str(b),'--envs-per-teacher','1024','--deadline',end.isoformat()]
with (b/'train.log').open('w') as log:child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with (b/'watchdog.log').open('w') as log:watch=subprocess.Popen([str(v/'bin/python'),str(b/'runtime/watch_teacher_bank.py'),'--bank',str(b),'--pid',str(child.pid),'--deadline',end.isoformat(),'--final-grace-s','120'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(bank=str(b),pid=child.pid,watchdog_pid=watch.pid,deadline=end.isoformat(),command=cmd,status='starting',fresh_weights=True,smoke_weights_reused=False,source_iteration=0,discarded_learning_bank=str(old),total_round_minutes=40,foot_weight=4.,foot_sigma_xyz_m=[.04,.04,.02],exploration_std=.15,minimum_std=.02)
for path in [b/'launch_receipt.json',s/'launch_receipt.json']:path.write_text(json.dumps(receipt,indent=2))
(old/'invalidated_by_user.json').write_text(json.dumps(dict(reason='User discarded this round; fresh teachers with foot weight restored to 4',replacement=str(b),created_unix=time.time(),checkpoints_preserved=True),indent=2))
manifest={str(f.relative_to(b/'runtime')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (b/'runtime').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
(b/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(receipt),flush=True)
