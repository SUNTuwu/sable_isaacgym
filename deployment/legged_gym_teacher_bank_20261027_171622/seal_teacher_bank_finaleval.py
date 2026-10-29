"""Bounded non-learning final evaluation and verified archive of one teacher bank."""
import argparse,datetime,hashlib,json,os,signal,subprocess,tarfile,time
from pathlib import Path

def atomic(path,data):
 path=Path(path);tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2),encoding='utf8');tmp.replace(path)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
 return h.hexdigest()
def identity(pid):
 try:
  proc=Path('/proc/%d'%pid);stat=proc.joinpath('stat').read_text();tail=stat.rsplit(')',1)[1].split()
  if tail[0]=='Z':return None
  return {'start_ticks':tail[19],'command':proc.joinpath('cmdline').read_bytes().decode(errors='replace').replace('\0',' ')}
 except FileNotFoundError:return None
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--core',type=Path,required=True);p.add_argument('--evaluator',type=Path,required=True);a=p.parse_args()
 bank=a.bank.resolve();project=Path(os.environ['SABLE_GYM_ROOT']).resolve();assert bank.parent==project/'runs' and bank.name.startswith('teacher_bank_')
 expected=identity(a.pid);assert expected and str(bank) in expected['command'];state=json.loads((bank/'teacher_bank.json').read_text(encoding='utf8'));deadline=datetime.datetime.fromisoformat(state['deadline']).timestamp()
 atomic(bank/'finalization_supervisor.json',{'pid':os.getpid(),'training_pid':a.pid,'training_start_ticks':expected['start_ticks'],'learning_deadline':state['deadline'],'term_before_deadline_s':5,'small_evaluator':str(a.evaluator),'core':str(a.core),'max_gpu_post_deadline_s':180,'status':'waiting'})
 while identity(a.pid) is not None and time.time()<deadline-5:time.sleep(min(5,max(.1,deadline-5-time.time())))
 current=identity(a.pid)
 if current is not None:
  assert current['start_ticks']==expected['start_ticks'] and str(bank) in current['command']
  atomic(bank/'external_final_evaluation_request.json',{'created_unix':time.time(),'reason':'Stop learning and skip expensive fulltraining-batch final inference; use isolated32env evaluation','training_pid':a.pid})
  os.kill(a.pid,signal.SIGTERM)
 while identity(a.pid) is not None:time.sleep(1)
 state=json.loads((bank/'teacher_bank.json').read_text(encoding='utf8'));evaluation_status={'attempted':False}
 remaining=deadline+180-time.time()
 if remaining>=10 and not (bank/'superseded.json').exists():
  command=[str(Path(os.environ['SABLE_GYM_ENV'])/'bin/python'),str(a.evaluator),'--bank',str(bank),'--core',str(a.core),'--max-wall-s',str(min(150,max(1,int(remaining)-5)))]
  try:
   result=subprocess.run(command,capture_output=True,text=True,timeout=min(175,remaining))
   (bank/'final_small_evaluation.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
   evaluation_status={'attempted':True,'returncode':result.returncode,'command':command,'completed_unix':time.time()}
  except subprocess.TimeoutExpired as error:
   evaluation_status={'attempted':True,'failed':'wall_budget_timeout','completed_unix':time.time()}
   def txt(v):return v.decode(errors='replace') if isinstance(v,bytes) else (v or '')
   (bank/'final_small_evaluation.log').write_text(txt(error.stdout)+'\n'+txt(error.stderr),encoding='utf8')
 else:evaluation_status={'attempted':False,'reason':'postdeadline_budget_exhausted_or_superseded'}
 checks=[]
 for teacher in state['teachers']:
  folder=bank/teacher['id'];receipt=json.loads((folder/'checkpoint.json').read_text(encoding='utf8'));cp=Path(receipt['path']).resolve();assert folder in cp.parents and digest(cp)==receipt['sha256']
  e=json.loads((folder/'latest_evaluation.json').read_text(encoding='utf8')) if (folder/'latest_evaluation.json').exists() else None
  # Both embedded and standalone evaluators record the exact frozen checkpoint.
  actual_hash=e.get('checkpoint',{}).get('sha256') if e else None
  checks.append({'teacher_id':teacher['id'],'checkpoint':str(cp),'sha256':receipt['sha256'],'iteration':receipt['iteration'],'checkpoint_verified':True,'final_independent_evaluation_complete':bool(e and e.get('iteration')==receipt['iteration'] and actual_hash==receipt['sha256'])})
 assert len(set(x['iteration'] for x in checks))==1
 archive=project/'results/teacher_bank'/(bank.name+'.tar.gz');archive.parent.mkdir(exist_ok=True)
 completion={'status':state['status'],'worker_error':state.get('error'),'training_pid':a.pid,'process_no_longer_active':True,'exit_code_observation':'Not observed by detached supervisor','learning_deadline':state['deadline'],'created_unix':time.time(),'teacher_count':len(checks),'checkpoints':checks,'small_evaluation':evaluation_status,'final_evaluations_complete':all(c['final_independent_evaluation_complete'] for c in checks),'archive':str(archive),'scope':'Final learning checkpoint followed by bounded nonlearning32env evaluation. No optimizer or training restart.'}
 atomic(bank/'completion.json',completion)
 part=archive.with_suffix('.partial')
 with tarfile.open(part,'w:gz',compresslevel=3) as tar:tar.add(bank,arcname=bank.name)
 part.replace(archive)
 with tarfile.open(archive,'r:gz') as tar:
  assert tar.getmember(bank.name+'/completion.json') and tar.getmember(bank.name+'/runtime_manifest.json')
  for c in checks:
   member=bank.name+'/'+str(Path(c['checkpoint']).relative_to(bank));stream=tar.extractfile(member);h=hashlib.sha256()
   for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
   assert h.hexdigest()==c['sha256']
 receipt={'archive':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,'contents_verified':True,'checkpoint_count':len(checks),'created_unix':time.time()}
 atomic(bank/'archive_receipt.json',receipt);atomic(archive.with_suffix('.receipt.json'),receipt)
 atomic(bank/'finalization_supervisor.json',{'pid':os.getpid(),'status':'complete','completion':str(bank/'completion.json'),'archive_receipt':str(bank/'archive_receipt.json'),'created_unix':time.time()})
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
