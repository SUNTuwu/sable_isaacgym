"""Read-only audit of stopped stage archives; no simulator import or training."""
import datetime,hashlib,json,os,tarfile,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
POST=ROOT/'postrun/adaptation_speedtrack20_20261026'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def audit(run,final=False):
    complete=read(run/'completion.json');backup=read(run/'backup_receipt.json');state=read(run/'status.json')
    assert complete['training_exit_code']==0
    assert complete['reason']==('deadline' if final else 'training_exited')
    assert state['status'] in ['stopped_at_deadline','stopped_requested']
    if final:
        assert complete['deadline']=='2026-10-26T02:00:00+08:00'
        deadline=datetime.datetime.fromisoformat(complete['deadline']).timestamp()
        assert time.time()>=deadline and deadline<=backup['checkpoint']['updated_unix']<=deadline+60
    for pid in [state.get('pid'),read(run/'supervisor.json').get('train_pid')]:
        cmdline=Path('/proc')/str(pid)/'cmdline'
        if cmdline.exists():assert str(run/'runtime/train.py').encode() not in cmdline.read_bytes(),'Training still active'
    archive=Path(backup['archive']);assert archive.stat().st_size==backup['bytes'] and sha(archive)==backup['sha256']
    full=0;prefix=[]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():continue
            path=ROOT/'runs'/member.name;data=tar.extractfile(member).read()
            if path.name=='supervisor.log':
                assert path.read_bytes().startswith(data);prefix.append(member.name)
            else:
                assert hashlib.sha256(data).hexdigest()==sha(path),member.name;full+=1
    manifest=read(run/'runtime_manifest.json')
    assert sha(run/'runtime_manifest.json')==backup['checkpoint']['runtime_sha256']==state['runtime_sha256']
    for name,expected in manifest['files'].items():assert sha(run/'runtime'/name)==expected,name
    checkpoint=Path(backup['checkpoint']['path']);assert sha(checkpoint)==backup['checkpoint']['sha256']
    assert complete['final_evaluation']['return_code']==0
    report=read(Path(complete['final_evaluation']['report']))
    assert report['checkpoint_sha256']==backup['checkpoint']['sha256']
    assert report['runtime_sha256']==backup['checkpoint']['runtime_sha256']
    assert report['num_envs']==24 and report['horizon_s']==16
    return {'run':run.name,'passed':True,'archive':backup,'exact_files_verified':full,'append_only_log_prefix_verified':prefix,
       'runtime_files_verified':len(manifest['files']),'status':state['status'],'updates':state['updates_this_run'],
       'total_environment_steps':state['total_steps'],'completion':complete,'evaluation_summary':report['summary'],
       'all20':report.get('all20_diagnostics'),'velocity_tracking':report.get('velocity_tracking')}

def main():
    plan=read(OUT/'adaptation_round_plan_20261026.json')
    assert time.time()>=datetime.datetime.fromisoformat(plan['deadline']).timestamp(),'Wait for the real deadline'
    stages=[audit(Path(path),path==plan['run']) for path in plan['stages']]
    POST.mkdir(parents=True,exist_ok=True)
    receipt={'passed':True,'created_unix':time.time(),'deadline':plan['deadline'],'stages':stages,
        'verifier_sha256':sha(Path(__file__)),'scope':'Server-local archive and live-file comparison. No Windows copy or hardware validation claimed.'}
    target=POST/'training_archives_verified.json';assert not target.exists(),'Do not overwrite a finished verification'
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({'passed':True,'report':str(target),'stages':[{'run':s['run'],'files':s['exact_files_verified'],'checkpoint':s['archive']['checkpoint']['iteration']} for s in stages]}))
if __name__=='__main__':main()
