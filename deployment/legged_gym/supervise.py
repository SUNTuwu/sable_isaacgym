"""Remote owner of one training process; deadline, crash record and final archive."""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
from run_io import atomic_json,sha256,read_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--deadline',default='2026-10-25T10:00:00+08:00');p.add_argument('--resume');p.add_argument('--critic-checkpoint');p.add_argument('--phase',choices=['balance','walk','gait','lift','guided'],default='walk')
    p.add_argument('--reference-strength',type=float,default=.25);p.add_argument('--mirror-policy',action='store_true');a=p.parse_args()
    run=a.run.resolve();deadline=datetime.datetime.fromisoformat(a.deadline).timestamp()
    assert str(run).startswith('/root/gpufree-data/')
    if time.time()>=deadline:raise SystemExit('Deadline passed; refusing launch')
    with (run/'train.log').open('a') as log:
        command=[sys.executable,'-u',str(run/'runtime/train.py'),'--out',str(run),'--deadline',a.deadline]
        if a.resume:command+=['--resume',a.resume]
        if a.critic_checkpoint:command+=['--critic-checkpoint',a.critic_checkpoint]
        command+=['--phase',a.phase]
        if a.phase=='guided':command+=['--reference-strength',str(a.reference_strength)]
        if a.mirror_policy:command+=['--mirror-policy']
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(run/'supervisor.json',{'pid':os.getpid(),'train_pid':child.pid,'deadline':a.deadline,'started_unix':time.time()})
    (run/'train.pid').write_text(str(child.pid))
    while child.poll() is None and time.time()<deadline:time.sleep(min(5,max(.1,deadline-time.time())))
    if child.poll() is None:
        (run/'STOP').write_text('10:00 Asia/Hong_Kong deadline\n')
        child.send_signal(signal.SIGTERM)
        try:child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            # Process group created by this supervisor only; never signal unrelated workers.
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    checkpoint=read_json(run/'checkpoint.json',{})
    receipt={'finished_unix':time.time(),'training_exit_code':child.returncode,'deadline':a.deadline,'checkpoint':checkpoint,
        'reason':'deadline' if time.time()>=deadline else 'training_exited','status':read_json(run/'status.json',{})}
    atomic_json(run/'completion.json',receipt)
    archive=run.parent/(run.name+'_backup.tar.gz');tmp=archive.with_suffix('.partial')
    with tarfile.open(str(tmp),'w:gz',compresslevel=1) as tar:tar.add(str(run),arcname=run.name)
    os.replace(str(tmp),str(archive))
    atomic_json(run/'backup_receipt.json',{'archive':str(archive),'sha256':sha256(archive),'bytes':archive.stat().st_size,'created_unix':time.time(),'checkpoint':checkpoint})
    print('BACKUP_COMPLETE',str(archive),flush=True)

if __name__=='__main__':main()
