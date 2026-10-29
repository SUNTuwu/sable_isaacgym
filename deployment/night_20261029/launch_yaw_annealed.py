"""Preflight an isolated yaw runtime, then preserve model/Adam while changing exploration only.

Prepared for explicit execution on the cloud server. Importing this module does nothing.
"""
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import signal
import subprocess
import sys
import time

P=Path('/root/gpufree-data/projects/sable-isaacgym')
D=P/'deployment/night_20261029';YD=P/'deployment/yaw_sine_20261029'
V=Path('/root/gpufree-data/envs/sable-isaacgym-py38')
END=datetime.datetime(2026,10,29,8,30,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
EXPLORATION={'max_std_deg':1.5,'anneal_iterations':40}

def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
    temporary=path.with_name(path.name+'.yaw_anneal.%s.tmp'%os.getpid())
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temporary,path)
def publish_bytes(path,data):
    temporary=path.with_name(path.name+'.yaw_anneal.%s.tmp'%os.getpid())
    temporary.write_bytes(data);os.replace(temporary,path)

def scoped_run(path):
    result=Path(path).resolve()
    if (P/'runs').resolve() not in result.parents:raise RuntimeError('Run path outside project runs: '+str(result))
    return result

def identity(pid):
    try:
        root=Path('/proc')/str(pid);fields=(root/'stat').read_text().rsplit(')',1)[1].split()
        args=[x.decode(errors='replace') for x in (root/'cmdline').read_bytes().split(b'\0') if x]
        if fields[0]=='Z' or not args:return None
        return {'pid':int(pid),'start_ticks':fields[19],'args':args}
    except (OSError,ValueError,IndexError,TypeError):return None

def same_process(saved):
    current=identity(saved['pid']) if saved else None
    return bool(current and current['start_ticks']==saved['start_ticks'] and current['args']==saved['args'])

def checked_identity(pid,required_args,script_name=None,allow_exited=False):
    current=identity(pid)
    if current is None:
        if allow_exited:return None
        raise RuntimeError('Required process is not running: '+str(pid))
    if any(str(x) not in current['args'] for x in required_args):raise RuntimeError('PID command-line identity mismatch: '+str(pid))
    if script_name and not any(Path(x).name==script_name for x in current['args']):raise RuntimeError('PID script mismatch: '+str(pid))
    return current

def stop_verified(saved,timeout=90):
    if not saved or not same_process(saved):return
    os.kill(saved['pid'],signal.SIGTERM)
    limit=time.time()+timeout
    while same_process(saved) and time.time()<limit:time.sleep(.5)
    if same_process(saved):raise RuntimeError('Process did not stop gracefully: '+str(saved['pid']))

def exploration_control(control,smoke=False):
    result=copy.deepcopy(control);yaw=result.setdefault('yaw',{})
    if smoke:yaw.pop('reload',None)
    elif yaw.get('reload'):raise RuntimeError('Pending yaw reload must be acknowledged before preserving the current model')
    yaw['exploration']=dict(EXPLORATION)
    return result

def patched_watchdog(source):
    # Keep training/deadline/grace unchanged; only the final evaluator receives extra time.
    if source.count('timeout=600')==0 and source.count('timeout=900')==1:return source
    if source.count('timeout=600')!=1:raise RuntimeError('Expected exactly one 600-second final-evaluation timeout')
    changed=source.replace('timeout=600','timeout=900')
    compile(changed,'<yaw-watchdog-final-eval-900>','exec')
    return changed

def checked_monitor_source(source):
    tree=ast.parse(source)
    qualifying=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.If):continue
        calls=[x for statement in node.body for x in ast.walk(statement) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='yaw_exploration_decision']
        names={x.id for x in ast.walk(node.test) if isinstance(x,ast.Name)}
        if calls and 'flat' in names and 'negative' not in names:qualifying.append(node)
    if not qualifying:raise RuntimeError('Monitor candidate must prioritize exploration only on flat windows, not negative windows')
    return source

def verify_resume(receipt,source,expected_checkpoint=None):
    if receipt.get('model_state_equal_before_exploration_control') is not True or receipt.get('optimizer_state_equal_before_exploration_control') is not True:
        raise RuntimeError('Exact model/Adam resume verification is missing')
    model=Path(receipt['checkpoint']).resolve();source=source.resolve()
    if source not in model.parents or Path(receipt['source']).resolve()!=source:raise RuntimeError('Resume source identity mismatch')
    if not model.is_file() or receipt.get('source_checkpoint_sha256')!=sha(model):raise RuntimeError('Resume source checkpoint SHA mismatch')
    if expected_checkpoint and (model!=Path(expected_checkpoint['path']).resolve()
            or receipt.get('source_iteration')!=expected_checkpoint['iteration']
            or receipt.get('source_checkpoint_sha256')!=expected_checkpoint['sha256']):
        raise RuntimeError('New teacher did not load the final gracefully saved checkpoint')
    return receipt

def training_env(runtime):
    env=dict(os.environ,SABLE_GYM_ROOT=str(P),SABLE_GYM_ENV=str(V),PYTHONPATH=str(runtime),PYTHONUNBUFFERED='1',
             TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
    env.pop('SABLE_NIGHT_CONTROL_FILE',None)
    env['PATH']=str(V/'bin')+':'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(V/'lib')+':'+str(V/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
    return env

def publish_receipts(run,receipt):
    # The canonical pointer is published last, after its supporting receipt copies.
    for path in (run/'launch_receipt.json',D/'yaw_receipt.json',YD/'launch_receipt.json'):atomic(path,receipt)

def restore_monitor(previous,record):
    if same_process(previous):return
    recorded=read(D/'monitor_pid.json') if (D/'monitor_pid.json').exists() else {}
    possible=identity(recorded.get('pid'))
    if possible and str(D/'monitor.py') in possible['args']:return
    with (D/'monitor.py.log').open('a') as log:
        child=subprocess.Popen([sys.executable,str(D/'monitor.py')],cwd=str(P),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(.5)
    if child.poll() is not None:raise RuntimeError('CPU monitor failed to restart')
    record.update(pid=child.pid,profile='yaw_annealed_v1',module_sha256=sha(D/'monitor.py'),restarted_unix=time.time())
    atomic(D/'monitor_pid.json',record)
    # monitor_status.json / monitor_memory.json are deliberately retained, including last_review_unix.

def main():
    if time.time()>=END-180:raise RuntimeError('Too close to the fixed 08:30 deadline to safely switch')
    receipt=read(YD/'launch_receipt.json');old=scoped_run(receipt['run'])
    old_trainer=checked_identity(receipt['pid'],[str(old)],'yaw_sine_teacher.py')
    if read(old/'status.json').get('status')!='training':raise RuntimeError('Source yaw teacher is not training')
    checked_identity(receipt.get('watchdog_pid'),[str(old),str(receipt['pid'])],'watch.py',allow_exited=True)
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    run=scoped_run(P/'runs'/('yaw_sine_night_annealed_'+stamp));run.mkdir()
    runtime=run/'runtime';shutil.copytree(old/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__'))
    trainer_source=(D/'yaw_sine_teacher_annealed.py').read_bytes()
    monitor_source=checked_monitor_source((D/'monitor_yaw_annealed.py').read_text())
    (runtime/'yaw_sine_teacher.py').write_bytes(trainer_source)
    if sha(runtime/'evaluate_yaw.py')!=sha(old/'runtime/evaluate_yaw.py'):raise RuntimeError('Frozen evaluator was changed')
    watchdog_source=old/'runtime/watch.py' if (old/'runtime/watch.py').exists() else YD/'watch.py'
    (runtime/'watch.py').write_text(patched_watchdog(watchdog_source.read_text()))
    for path in (runtime/'yaw_sine_teacher.py',runtime/'watch.py',runtime/'evaluate_yaw.py'):py_compile.compile(str(path),doraise=True)
    smoke_control=run/'smoke_control.json';atomic(smoke_control,exploration_control(read(D/'control.json'),smoke=True))
    env=training_env(runtime);command=[str(V/'bin/python'),str(runtime/'yaw_sine_teacher.py')]
    state={'status':'preflighting','source_run':str(old),'attempted_run':str(run),'deadline_unix':END,'stage':'smoke'}
    atomic(run/'switch_status.json',state)
    monitor=None;monitor_record={};monitor_pause_requested=False;old_stop_requested=False;old_stopped=False;child=None;watch=None;old_watch=None;old_watch_stopped=False
    try:
        with (run/'smoke.log').open('w') as log:
            result=subprocess.run(command+['--run',str(run/'smoke'),'--envs','32','--smoke','--resume',str(old),'--deadline',str(END)],
                cwd=str(P),env=dict(env,SABLE_NIGHT_CONTROL_FILE=str(smoke_control)),stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if result.returncode or read(run/'smoke/status.json').get('status')!='smoke_complete':raise RuntimeError('Yaw preflight failed; old teacher remains running')
        verify_resume(read(run/'smoke/resume.json'),old)
        if read(YD/'launch_receipt.json').get('run')!=str(old) or not same_process(old_trainer):raise RuntimeError('Source changed during preflight')
        if (D/'yaw_sine_teacher_annealed.py').read_bytes()!=trainer_source or checked_monitor_source((D/'monitor_yaw_annealed.py').read_text())!=monitor_source:
            raise RuntimeError('Candidates changed during preflight')
        if time.time()>=END-90:raise RuntimeError('Preflight exhausted the safe switch window')
        monitor_record=read(D/'monitor_pid.json');monitor=checked_identity(monitor_record['pid'],[str(D/'monitor.py')])
        state['stage']='pause_monitor';monitor_pause_requested=True;stop_verified(monitor,30)
        control_before=read(D/'control.json');control_after=exploration_control(control_before)
        if control_after.get('teachers')!=control_before.get('teachers'):raise RuntimeError('Bank teacher configuration would change')
        atomic(run/'control_before.json',control_before)
        state['stage']='stop_old_watchdog'
        old_watch=checked_identity(receipt.get('watchdog_pid'),[str(old),str(receipt['pid'])],'watch.py',allow_exited=True)
        stop_verified(old_watch,30);old_watch_stopped=True
        state['stage']='save_old_yaw';stop_time=time.time();old_stop_requested=True;stop_verified(old_trainer,90);old_stopped=True
        final_status=read(old/'status.json');final_cp=read(old/'checkpoint.json');final_path=Path(final_cp['path']).resolve()
        if final_status.get('status') not in ('completed','stopped_requested') or final_cp.get('reason')!='final':raise RuntimeError('Old yaw did not publish a graceful final save')
        if old not in final_path.parents or final_path.stat().st_mtime<stop_time:raise RuntimeError('Final checkpoint predates the graceful stop')
        final_cp=dict(final_cp,sha256=sha(final_path));atomic(run/'source_final_checkpoint.json',final_cp)
        for target,data in ((D/'yaw_sine_teacher.py',trainer_source),(D/'monitor.py',monitor_source.encode())):
            shutil.copy2(target,target.with_name(target.name+'.before_yaw_anneal_'+stamp));publish_bytes(target,data)
        control_after['revision']=control_before.get('revision',0)+1
        atomic(D/'control.json',control_after);atomic(run/'control_after.json',control_after)
        state['stage']='start_new_yaw'
        train_command=command+['--run',str(run),'--envs','512','--resume',str(old),'--deadline',str(END)]
        with (run/'train.log').open('w') as log:
            child=subprocess.Popen(train_command,cwd=str(P),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        startup_end=min(time.time()+180,END)
        while time.time()<startup_end:
            if child.poll() is not None:raise RuntimeError('New yaw exited during initialization')
            if (run/'resume.json').exists() and (run/'status.json').exists():
                resume=verify_resume(read(run/'resume.json'),old,final_cp);status=read(run/'status.json')
                if status.get('status')=='training' and status.get('pid')==child.pid:break
            time.sleep(1)
        else:raise RuntimeError('New yaw failed to acknowledge exact model/Adam resume before deadline')
        state['stage']='start_new_watchdog'
        with (run/'watchdog.log').open('w') as log:
            watch=subprocess.Popen([sys.executable,str(runtime/'watch.py'),str(run),str(child.pid)],cwd=str(P),env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        new_receipt=dict(run=str(run),pid=child.pid,watchdog_pid=watch.pid,status='training',source_run=str(old),
            source_checkpoint=final_cp,source_iteration=final_cp['iteration'],resume_verification=resume,
            deadline_unix=END,command=train_command,exploration=dict(EXPLORATION),
            frozen_evaluator_sha256=sha(runtime/'evaluate_yaw.py'),frozen_watchdog_sha256=sha(runtime/'watch.py'),
            final_evaluation_timeout_s=900,bank_teachers_unchanged=True,transition_id=stamp)
        publish_receipts(run,new_receipt);state.update(status='completed',stage='published',receipt=new_receipt)
        atomic(run/'switch_status.json',state)
    except BaseException as exc:
        cleanup=[]
        for process,name in ((watch,'watch.py'),(child,'yaw_sine_teacher.py')):
            if process is not None and process.poll() is None:
                try:stop_verified(checked_identity(process.pid,[str(run)],name),60)
                except Exception as cleanup_error:cleanup.append(repr(cleanup_error))
        old_running=same_process(old_trainer)
        if old_running and old_watch_stopped:
            try:
                with (old/'watchdog_recovery.log').open('a') as log:
                    restored_watch=subprocess.Popen([sys.executable,str(watchdog_source),str(old),str(receipt['pid'])],cwd=str(P),
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                receipt=dict(receipt,watchdog_pid=restored_watch.pid)
                for path in (D/'yaw_receipt.json',YD/'launch_receipt.json'):atomic(path,receipt)
            except Exception as cleanup_error:cleanup.append('old_watchdog_restore: '+repr(cleanup_error))
        state.update(status='failed_old_training_kept' if old_running else 'failed_after_old_yaw_stopped',error=repr(exc),
            old_stop_requested=old_stop_requested,old_stopped=old_stopped,old_process_running=old_running,cleanup_errors=cleanup)
        atomic(run/'switch_status.json',state)
        if old_stop_requested and not old_running:
            failed=dict(receipt,status='failed_switch',attempted_run=str(run),failure=state,deadline_unix=END,
                recovery_source='old run final checkpoint; new attempt was not accepted')
            publish_receipts(run,failed)
        print(json.dumps(state),flush=True);raise
    finally:
        if monitor_pause_requested:
            try:restore_monitor(monitor,monitor_record)
            except Exception as restore_error:
                state.update(status='failed_monitor_restore',monitor_restore_error=repr(restore_error))
                atomic(run/'switch_status.json',state);print(json.dumps(state),flush=True);raise
    print(json.dumps(new_receipt),flush=True)

if __name__=='__main__':main()
