"""Non-learning final archive monitor for a single explicitly named teacher bank."""
import argparse,hashlib,json,time,tarfile,os
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def atomic(path,data):
 path=Path(path);tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2),encoding='utf8');tmp.replace(path)
def alive(pid):
 try:return Path('/proc/%d/stat'%pid).read_text().split()[2]!='Z'
 except FileNotFoundError:return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--pid',type=int,required=True);a=p.parse_args();bank=a.bank.resolve()
 root=Path(os.environ['SABLE_GYM_ROOT']).resolve();assert bank.parent==root/'runs' and bank.name.startswith('teacher_bank_')
 while alive(a.pid):time.sleep(5)
 state=json.loads((bank/'teacher_bank.json').read_text(encoding='utf8'))
 checks=[]
 for teacher in state['teachers']:
  folder=bank/teacher['id'];receipt=json.loads((folder/'checkpoint.json').read_text(encoding='utf8'));cp=Path(receipt['path']).resolve()
  assert folder in cp.parents and cp.is_file()
  checkpoint_verified=digest(cp)==receipt['sha256'];assert checkpoint_verified
  evaluation=json.loads((folder/'latest_evaluation.json').read_text(encoding='utf8')) if (folder/'latest_evaluation.json').exists() else None
  final_eval=bool(evaluation and evaluation['iteration']==receipt['iteration'] and evaluation['checkpoint']['sha256']==receipt['sha256'])
  checks.append({'teacher_id':teacher['id'],'checkpoint':str(cp),'sha256':receipt['sha256'],'iteration':receipt['iteration'],'checkpoint_verified':True,'final_independent_evaluation_complete':final_eval})
 archive=root/'results/teacher_bank'/(bank.name+'.tar.gz');archive.parent.mkdir(exist_ok=True)
 completion={'status':state['status'],'error':state.get('error'),'sealed_unix':time.time(),'training_pid':a.pid,
  'process_no_longer_active':True,'exit_code_observation':'not_observed_by_detached_monitor','learning_deadline':state['deadline'],
  'teacher_count':len(checks),'checkpoints':checks,'final_evaluations_complete':all(x['final_independent_evaluation_complete'] for x in checks),
  'archive':str(archive),'scope':'No policy evaluation, optimization or GPU work is performed by this finalizer.'}
 atomic(bank/'completion.json',completion)
 temp=archive.with_suffix('.partial')
 with tarfile.open(temp,'w:gz',compresslevel=3) as tar:tar.add(bank,arcname=bank.name)
 temp.replace(archive)
 with tarfile.open(archive,'r:gz') as tar:
  assert tar.getmember(bank.name+'/completion.json')
  assert tar.getmember(bank.name+'/runtime_manifest.json')
  for c in checks:
   member=bank.name+'/'+str(Path(c['checkpoint']).relative_to(bank));h=hashlib.sha256();stream=tar.extractfile(member)
   for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
   assert h.hexdigest()==c['sha256']
 receipt={'archive':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,'contents_verified':True,'checkpoint_count':len(checks),'created_unix':time.time()}
 atomic(bank/'archive_receipt.json',receipt);atomic(archive.with_suffix('.receipt.json'),receipt)
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
