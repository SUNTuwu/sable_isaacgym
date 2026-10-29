"""Deploy a frozen turn/lateral reward revision, check, then resume one round."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def atomic(path, data):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    project=Path(os.environ['SABLE_GYM_ROOT']).resolve()
    python=str(Path(os.environ['SABLE_GYM_ENV'])/'bin/python')
    parent=project/'runs/teacher_bank_fullref_directlift_20261028_020724'
    state=json.loads((parent/'teacher_bank.json').read_text())
    assert state['status']=='training'
    inherited_deadline=state['deadline']
    assert datetime.datetime.fromisoformat(inherited_deadline).timestamp()>time.time()+120
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    bank=project/'runs'/('teacher_bank_fullref_directlift_xy_'+stamp)
    bank.mkdir();runtime=bank/'runtime'
    shutil.copytree(parent/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__'))
    here=Path(__file__).resolve().parent
    for name in ['teacher_bank_core.py','teacher_task_tracking.py','evaluate_teacher_bank_small.py',
                 'check_teacher_task_tracking.py','launch_teacher_directlift_xy.py','teacher_motion_rewards.py',
                 'teacher_step_quality.py','check_teacher_step_quality.py','teacher_yaw_integral.py','check_teacher_yaw_integral.py','teacher_gait_balance.py','check_teacher_gait_balance.py','teacher_translation_guard.py','check_teacher_translation_guard.py','teacher_lift_limits.py','teacher_event_symmetry.py','check_teacher_lift_symmetry.py','turn_lift_limits.json']:
        shutil.copy2(here/name,runtime/name)
    hashes={str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.rglob('*') if p.is_file()}
    atomic(bank/'runtime_manifest.json',dict(created_unix=time.time(),files=hashes,
           inherited_runtime=str(parent/'runtime'),entrypoint=str(runtime/'teacher_bank_core.py')))
    env=dict(os.environ,PYTHONUNBUFFERED='1');env['PYTHONPATH']=str(runtime)+os.pathsep+env.get('PYTHONPATH','')
    check=subprocess.run([python,str(runtime/'check_teacher_yaw_integral.py')],env=env,capture_output=True,text=True,timeout=40)
    (bank/'task_tracking_cpu_contract.log').write_text(check.stdout+'\n'+check.stderr)
    assert check.returncode==0,check.stderr
    reference=project/'motion_library/pace30_switch_20261027/training_reference'
    assert (reference/'manifest.json').is_file()
    common=[python,str(runtime/'teacher_bank_core.py'),'--reference',str(reference),'--resume-bank',str(parent),
        '--control-mode','reference_plus_residual','--reference-strength','1','--time-scale','1',
        '--offset-limit-deg','20','--imitation-scale','.2','--balance-first','--minimum-action-std','.15',
        '--exploration-std','.15','--adaptive-curriculum','--motion-reward-module','--task-tracking-profile','turn90_lateral_tracking_v1','--step-quality-profile','strict_steps_v1','--translation-guard','--turn-speed-scale','.5','--turn-reference-time-scale','3','--turn-offset-rate-deg-s','50','--turn-lift-config',str(runtime/'turn_lift_limits.json')]
    with (bank/'ppo_smoke.log').open('w') as log:
        smoke=subprocess.run(common+['--bank',str(bank/'smoke'),'--envs-per-teacher','128','--smoke','--duration','120'],
                             env=env,stdout=log,stderr=subprocess.STDOUT,timeout=150)
    assert smoke.returncode==0,'PPO smoke failed; see ppo_smoke.log'
    smoke_state=json.loads((bank/'smoke/teacher_bank.json').read_text())
    assert smoke_state['status']=='smoke_complete' and smoke_state['iteration']==smoke_state['initial_iteration']+1
    # Freeze/save the exact parent only after smoke passes. Its sealer archives
    # without an overlapping GPU evaluation; the new run keeps the old deadline.
    prior=json.loads((parent/'teacher_bank.json').read_text());pid=int(prior['pid'])
    proc=Path('/proc')/str(pid);cmdline=proc.joinpath('cmdline').read_bytes().decode(errors='replace')
    assert str(parent/'runtime/teacher_bank_core.py') in cmdline and str(parent) in cmdline
    atomic(parent/'superseded.json',{'next_bank':str(bank),'created_unix':time.time(),'reason':'Raw contact XY symmetry fix; same deadline, checkpoint and Adam continuation'})
    atomic(parent/'external_final_evaluation_request.json',{'created_unix':time.time(),'reason':'Continue current checkpoints in new immutable runtime; skip duplicate final GPU evaluation'})
    os.kill(pid,signal.SIGTERM)
    stop_end=time.time()+65
    while proc.exists():
        stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':break
        assert time.time()<stop_end,'Parent did not finish saving; refuse concurrent formal training'
        time.sleep(.5)
    final=json.loads((parent/'teacher_bank.json').read_text());assert final['status']=='stopped_requested'
    checks=[json.loads((parent/t['id']/'checkpoint.json').read_text()) for t in final['teachers']]
    assert len({x['iteration'] for x in checks})==1
    assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in checks)
    atomic(bank/'parent_transition.json',{'source_bank':str(parent),'source_iteration':checks[0]['iteration'],'checkpoint_hashes_verified':True,'old_learning_pid':pid,'deadline_unchanged':inherited_deadline})
    deadline=datetime.datetime.fromisoformat(inherited_deadline)
    cmd=common+['--bank',str(bank),'--envs-per-teacher','2048','--minibatches','2','--deadline',deadline.isoformat()]
    with (bank/'train.log').open('w') as log:
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(here/'launch_receipt.json',dict(bank=str(bank),pid=process.pid,deadline=deadline.isoformat(),status='starting'))
    # Install deadline containment immediately; waiting for trainer initialization
    # is only needed by the final-evaluation supervisor, which reads bank metadata.
    watch=project/'deployment/teacher_bank_support_20261027/watch_teacher_bank_v2.py'
    with (bank/'watchdog.log').open('w') as log:
        watchdog=subprocess.Popen(['python3',str(watch),'--bank',str(bank),'--pid',str(process.pid),
                                   '--deadline',deadline.isoformat()],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    wait_end=time.time()+120
    while not (bank/'teacher_bank.json').exists():
        assert process.poll() is None,'Trainer failed during initialization'
        assert time.time()<wait_end,'Trainer initialization timeout; watchdog remains active'
        time.sleep(1)
    seal=project/'deployment/legged_gym_teacher_bank_20261027_171622/seal_teacher_bank_finaleval.py'
    with (bank/'sealer.log').open('w') as log:
        sealer=subprocess.Popen(['python3',str(seal),'--bank',str(bank),'--pid',str(process.pid),
                '--core',str(runtime/'teacher_bank_core.py'),'--evaluator',str(runtime/'evaluate_teacher_bank_small.py')],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(bank=str(bank),pid=process.pid,deadline=deadline.isoformat(),status='training',
                 watchdog_pid=watchdog.pid,sealer_pid=sealer.pid,source_bank=str(parent),source_iteration=json.loads((parent/'teacher_bank.json').read_text())['iteration'],
                 task_tracking_profile='turn90_lateral_tracking_v1',step_quality_profile='strict_steps_v1',ppo_smoke_updates=1,
                 formal_resumed_smoke_weights=False,command=cmd)
    atomic(bank/'launch_receipt.json',receipt);atomic(here/'launch_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
