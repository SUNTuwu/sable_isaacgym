"""Preflight a frozen reward revision while existing teachers keep training."""
import argparse,datetime,hashlib,json,os,py_compile,shutil,signal,subprocess,time
from pathlib import Path

P=Path('/root/gpufree-data/projects/sable-isaacgym');D=P/'deployment/night_20261029'
V=Path('/root/gpufree-data/envs/sable-isaacgym-py38')
END=datetime.datetime(2026,10,29,8,30,tzinfo=datetime.timezone(datetime.timedelta(hours=8)))

def read(path):return json.loads(path.read_text())
def atomic(path,value):
    temp=path.with_name(path.name+'.netprogress.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def live(pid,token):
    path=Path('/proc',str(pid),'cmdline')
    return path.exists() and str(token).encode() in path.read_bytes()
def wait_exit(pid):
    path=Path('/proc',str(pid),'cmdline')
    for _ in range(160):
        if not path.exists() or not path.read_bytes():return
        time.sleep(.5)
    raise RuntimeError('Process did not exit: '+str(pid))
def patched_control(control,token,yaw_weight,rollback=True):
    baseline=read(D/'baseline_sources.json')
    for ident in ['04_TURN_LEFT','05_TURN_RIGHT']:
        cfg=control['teachers'].setdefault(ident,{})
        cfg.setdefault('multipliers',{})['v2_yaw']=yaw_weight
        cfg['reward_profile']='cycle_net_progress_gamma099_v1'
        cfg['reason']='Replace instantaneous yaw tracking with discounted per-cycle signed net-angle potential; retain30deg/s and physical acceptance.'
    for ident in ['02_STRAFE_LEFT','03_STRAFE_RIGHT']:
        cfg=control['teachers'].setdefault(ident,{})
        cfg['cross_track_sigma_m']=.10
        cfg['reason']='Replace weak body-frame cross-speed cost with signed integrated cross-path error in initial heading frame; normal reversible sway remains cheap.'
    for ident in ['04_TURN_LEFT','05_TURN_RIGHT','02_STRAFE_LEFT','03_STRAFE_RIGHT']:
        cfg=control['teachers'][ident]
        if rollback:cfg['reload']=dict(baseline[ident]['checkpoint'],token=token+'_'+ident)
        else:cfg.pop('reload',None)
    return control

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--turn-yaw-multiplier',type=float,default=.4)
    args=parser.parse_args();assert .25<=args.turn_yaw_multiplier<=1.2
    assert time.time()<END.timestamp()-120
    receipt_path=P/'deployment/reference_v2_20261028/launch_receipt.json'
    receipt=read(receipt_path);old=Path(receipt['bank']);state=read(old/'teacher_bank.json')
    assert state['status']=='training' and live(state['pid'],old)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bank=P/'runs'/('teacher_bank_fullref_referencev2_netprogress_'+stamp);bank.mkdir()
    shutil.copytree(old/'runtime',bank/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
    overlays={'audited_core.py':'teacher_bank_core.py','audited_rewards_turnprogress.py':'teacher_reference_rewards_v2.py',
              'teacher_turn_cycles_reviewed.py':'teacher_turn_cycles.py','night_tuning_pace_path.py':'night_tuning.py'}
    hashes={}
    for source,target in overlays.items():
        shutil.copy2(D/source,bank/'runtime'/target);py_compile.compile(str(bank/'runtime'/target),doraise=True)
        hashes[target]=hashlib.sha256((bank/'runtime'/target).read_bytes()).hexdigest()
    reference=Path(receipt['command'][receipt['command'].index('--reference')+1]);shutil.copytree(reference,bank/'reference')
    env=dict(os.environ,SABLE_GYM_ROOT=str(P),SABLE_GYM_ENV=str(V),PYTHONPATH=str(bank/'runtime'),PYTHONUNBUFFERED='1',TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
    env['PATH']=str(V/'bin')+':'+env.get('PATH','');env['LD_LIBRARY_PATH']=str(V/'lib')+':'+str(V/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
    draft=bank/'smoke_control.json';atomic(draft,patched_control(read(D/'control.json'),stamp,args.turn_yaw_multiplier,False))
    command=list(receipt['command']);command[1]=str(bank/'runtime/teacher_bank_core.py')
    for flag,value in [('--bank',str(bank)),('--reference',str(bank/'reference')),('--deadline',END.isoformat())]:command[command.index(flag)+1]=value
    if '--resume-bank' in command:command[command.index('--resume-bank')+1]=str(old)
    else:command+=['--resume-bank',str(old)]
    smoke=list(command);smoke[smoke.index('--bank')+1]=str(bank/'smoke');smoke[smoke.index('--envs-per-teacher')+1]='32';smoke+=['--smoke']
    with (bank/'ppo_smoke.log').open('w') as log:
        result=subprocess.run(smoke,env=dict(env,SABLE_NIGHT_CONTROL_FILE=str(draft)),stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0 and read(bank/'smoke/teacher_bank.json')['status']=='smoke_complete','Preflight failed; original trainer untouched'
    assert read(receipt_path)['bank']==str(old) and live(state['pid'],old),'Source run changed during preflight'
    monitor=read(D/'monitor_pid.json')['pid'];assert live(monitor,D/'monitor.py')
    os.kill(monitor,signal.SIGTERM);wait_exit(monitor)
    try:
        control=read(D/'control.json');atomic(bank/'control_before.json',control)
        control=patched_control(control,'netprogress_'+stamp,args.turn_yaw_multiplier);control['revision']+=1
        atomic(D/'control.json',control);atomic(bank/'control_after.json',control)
        os.kill(state['pid'],signal.SIGTERM);wait_exit(state['pid'])
        final=read(old/'teacher_bank.json');assert final['status']=='stopped_requested'
        # Keep future recovery launches on exactly this audited reward implementation.
        for source,target in [('audited_rewards_turnprogress.py','audited_rewards.py'),('night_tuning_pace_path.py','night_tuning.py')]:
            backup=D/(target+'.before_netprogress_'+stamp);shutil.copy2(D/target,backup);shutil.copy2(D/source,D/target)
        with (bank/'train.log').open('w') as log:child=subprocess.Popen(command,cwd=str(P),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with (bank/'watchdog.log').open('w') as log:watch=subprocess.Popen([str(V/'bin/python'),str(bank/'runtime/watch_teacher_bank.py'),'--bank',str(bank),'--pid',str(child.pid),'--deadline',END.isoformat(),'--final-grace-s','120'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt.update(bank=str(bank),pid=child.pid,watchdog_pid=watch.pid,command=command,source_bank=str(old),source_iteration=final['iteration'],deadline=END.isoformat(),status='starting',modified_teachers=['04_TURN_LEFT','05_TURN_RIGHT','02_STRAFE_LEFT','03_STRAFE_RIGHT'],reward_revision='cycle_net_progress_gamma099_and_pace_integrated_cross_path',frozen_module_sha256=hashes)
        atomic(receipt_path,receipt);atomic(bank/'launch_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    finally:
        with (D/'monitor.py.log').open('a') as log:monitor_child=subprocess.Popen(['python3',str(D/'monitor.py')],cwd=str(P),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(D/'monitor_pid.json',dict(pid=monitor_child.pid,profile='stable_objective_monotone_trend_v3'))

if __name__=='__main__':main()
