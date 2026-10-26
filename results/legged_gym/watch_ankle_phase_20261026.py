"""File/process watchdog; no PPO, model edits, or fabricated assistant feedback."""
import os,time,json,hashlib,fcntl,sys
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');RUN=ROOT/'runs/stand5_20261026_v34_ankle_phase';OUT=ROOT/'results/legged_gym'
def read(p,d=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return d
def write(p,d):
 q=p.with_name(p.name+'.watch.partial');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
last_eval=None;failures=0;end=time.time()+80*60
while time.time()<end:
 state=read(RUN/'status.json',{});ev=read(RUN/'latest_evaluation.json',{})
 if ev and ev.get('iteration')!=last_eval:
  last_eval=ev['iteration'];failures=failures+1 if ev['summary']['survival_16s']<.5 else 0
  if failures>=3:(RUN/'STOP').write_text('Automatic watch: three independent evaluations below50% survival; preserve checkpoint for review.\n')
 messages=[]
 with (OUT/'operator_feedback.jsonl').open() as f:
  fcntl.flock(f,fcntl.LOCK_SH)
  for line in f:
   if line.strip():messages.append(json.loads(line))
 answered={m.get('reply_to') for m in messages if m['role']=='assistant'}
 pending=[m for m in messages if m['role']=='user' and m['id'] not in answered]
 write(OUT/'ankle_phase_pending_feedback.json',{'scanned_unix':time.time(),'scope':'Automatic file scan only; not evidence the assistant read or acted on these messages.','pending':pending})
 watch={'checked_unix':time.time(),'run':RUN.name,'status':state.get('status'),'iteration':state.get('iteration'),'consecutive_survival_below_half':failures,'pending_operator_notes':len(pending),'deadline':state.get('deadline'),'automatic_resume':False,'mode':'automatic metrics/process watch; no reward/controller edits'}
 receipt=read(RUN/'backup_receipt.json')
 if receipt:
  completion=read(RUN/'completion.json',{});cp=read(RUN/'checkpoint.json',{});rt=read(RUN/'runtime_manifest.json',{})
  valid=sha(Path(receipt['archive']))==receipt['sha256'] and sha(Path(cp['path']))==cp['sha256'] and all(sha(RUN/'runtime'/n)==h for n,h in rt.get('files',{}).items())
  watch.update(finished=True,archive_verified=valid,training_exit_code=completion.get('training_exit_code'),backup=receipt)
  write(OUT/'completion_ankle_phase_20261026.json',watch)
  plan=read(OUT/'active_training_plan.json',{});plan.update(status='complete' if valid and completion.get('training_exit_code')==0 else 'needs_review',automatic_resume=False);write(OUT/'active_training_plan.json',plan)
  with (OUT/'operator_monitor_status.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);m=read(OUT/'operator_monitor_status.json',{});m.update(finished=True)
   m['batch']['status']='complete' if valid else 'needs_review';m['batch']['updated_unix']=time.time();m['batch']['arms'][0]['backup']=receipt;write(OUT/'operator_monitor_status.json',m)
  write(OUT/'ankle_phase_watch_status.json',watch);break
 write(OUT/'ankle_phase_watch_status.json',watch)
 time.sleep(30)
else:
 write(OUT/'ankle_phase_watch_status.json',{'finished':False,'status':'watch_timeout_needs_review','checked_unix':time.time(),'automatic_resume':False})
