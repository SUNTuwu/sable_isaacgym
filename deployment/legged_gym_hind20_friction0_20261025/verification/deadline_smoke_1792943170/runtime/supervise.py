"""Remote owner of one training process; deadline, crash record and final archive."""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
from run_io import atomic_json,sha256,read_json
from progress_report import publish

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--deadline',required=True);p.add_argument('--resume');p.add_argument('--critic-checkpoint');p.add_argument('--phase',choices=['balance','walk','gait','lift','guided'],default='walk')
    p.add_argument('--reference-strength',type=float,default=.25);p.add_argument('--mirror-policy',action='store_true')
    p.add_argument('--front-friction-scale',type=float,default=0.);p.add_argument('--num-envs',type=int,default=1024);a=p.parse_args()
    run=a.run.resolve();deadline=datetime.datetime.fromisoformat(a.deadline).timestamp()
    assert str(run).startswith('/root/gpufree-data/')
    if time.time()>=deadline:raise SystemExit('Deadline passed; refusing launch')
    with (run/'train.log').open('a') as log:
        command=[sys.executable,'-u',str(run/'runtime/train.py'),'--out',str(run),'--deadline',a.deadline]
        if a.resume:command+=['--resume',a.resume]
        if a.critic_checkpoint:command+=['--critic-checkpoint',a.critic_checkpoint]
        command+=['--phase',a.phase]
        command+=['--front-friction-scale',str(a.front_friction_scale),'--num-envs',str(a.num_envs)]
        if a.phase=='guided':command+=['--reference-strength',str(a.reference_strength)]
        if a.mirror_policy:command+=['--mirror-policy']
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(run/'supervisor.json',{'pid':os.getpid(),'train_pid':child.pid,'deadline':a.deadline,'started_unix':time.time()})
    (run/'train.pid').write_text(str(child.pid))
    next_report=0.
    while child.poll() is None and time.time()<deadline:
        if time.time()>=next_report:
            publish(run);next_report=time.time()+300.
        time.sleep(min(5,max(.1,deadline-time.time()),max(.1,next_report-time.time())))
    if child.poll() is None:
        (run/'STOP').write_text(a.deadline+' deadline\n')
        child.send_signal(signal.SIGTERM)
        try:child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            # Process group created by this supervisor only; never signal unrelated workers.
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    checkpoint=read_json(run/'checkpoint.json',{})
    # PPO has ended. Replay the actual final weight before the final archive.
    final_evaluation={'attempted':False}
    if child.returncode==0 and checkpoint.get('path') and (run/'config.json').exists():
        report=run/'evaluations'/('final_%06d.json'%checkpoint['iteration'])
        with (run/'evaluation.log').open('a') as log:
            try:
                replay=subprocess.run([sys.executable,str(run/'runtime/evaluate.py'),'--checkpoint',checkpoint['path'],'--output',str(report)],stdout=log,stderr=subprocess.STDOUT,timeout=360)
                final_evaluation={'attempted':True,'return_code':replay.returncode,'report':str(report)}
                if replay.returncode==0:
                    ev=read_json(report)
                    if ev:atomic_json(run/'latest_evaluation.json',ev)
            except subprocess.TimeoutExpired:final_evaluation={'attempted':True,'return_code':124}
    publish(run,final=True)
    receipt={'finished_unix':time.time(),'training_exit_code':child.returncode,'deadline':a.deadline,'checkpoint':checkpoint,
        'final_evaluation':final_evaluation,'reason':'deadline' if time.time()>=deadline else 'training_exited','status':read_json(run/'status.json',{})}
    atomic_json(run/'completion.json',receipt)
    archive=run.parent/(run.name+'_backup.tar.gz');tmp=archive.with_suffix('.partial')
    with tarfile.open(str(tmp),'w:gz',compresslevel=1) as tar:tar.add(str(run),arcname=run.name)
    os.replace(str(tmp),str(archive))
    atomic_json(run/'backup_receipt.json',{'archive':str(archive),'sha256':sha256(archive),'bytes':archive.stat().st_size,'created_unix':time.time(),'checkpoint':checkpoint})
    print('BACKUP_COMPLETE',str(archive),flush=True)

if __name__=='__main__':main()
