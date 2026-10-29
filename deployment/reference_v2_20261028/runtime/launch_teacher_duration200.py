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
    parent=project/'runs/teacher_bank_fullref_strict_steps_bounded_20261027_201812'
    state=json.loads((parent/'teacher_bank.json').read_text())
    assert state['iteration']==527 and state['status'].startswith('stopped')
    assert json.loads((parent/'completion.json').read_text())['final_evaluations_complete']
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    bank=project/'runs'/('teacher_bank_fullref_duration200_yawsq_v4_'+stamp)
    bank.mkdir();runtime=bank/'runtime'
    shutil.copytree(parent/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__'))
    here=Path(__file__).resolve().parent
    for name in ['teacher_bank_core.py','teacher_task_tracking.py','evaluate_teacher_bank_small.py',
                 'check_teacher_task_tracking.py','launch_teacher_duration200.py','teacher_motion_rewards.py',
                 'teacher_step_quality.py','check_teacher_step_quality.py']:
        shutil.copy2(here/name,runtime/name)
    hashes={str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.rglob('*') if p.is_file()}
    atomic(bank/'runtime_manifest.json',dict(created_unix=time.time(),files=hashes,
           inherited_runtime=str(parent/'runtime'),entrypoint=str(runtime/'teacher_bank_core.py')))
    env=dict(os.environ,PYTHONUNBUFFERED='1');env['PYTHONPATH']=str(runtime)+os.pathsep+env.get('PYTHONPATH','')
    check=subprocess.run([python,str(runtime/'check_teacher_step_quality.py')],env=env,capture_output=True,text=True,timeout=40)
    (bank/'task_tracking_cpu_contract.log').write_text(check.stdout+'\n'+check.stderr)
    assert check.returncode==0,check.stderr
    reference=project/'motion_library/blender_motion_library_20261027_165843/processed_v1'
    assert (reference/'manifest.json').is_file()
    common=[python,str(runtime/'teacher_bank_core.py'),'--reference',str(reference),'--resume-bank',str(parent),
        '--control-mode','reference_plus_residual','--reference-strength','1','--time-scale','1',
        '--offset-limit-deg','20','--imitation-scale','.2','--balance-first','--minimum-action-std','.15',
        '--exploration-std','.15','--adaptive-curriculum','--motion-reward-module','--task-tracking-profile','turn90_lateral_tracking_v1','--step-quality-profile','strict_steps_v1']
    with (bank/'ppo_smoke.log').open('w') as log:
        smoke=subprocess.run(common+['--bank',str(bank/'smoke'),'--envs-per-teacher','128','--smoke','--duration','120'],
                             env=env,stdout=log,stderr=subprocess.STDOUT,timeout=150)
    assert smoke.returncode==0,'PPO smoke failed; see ppo_smoke.log'
    smoke_state=json.loads((bank/'smoke/teacher_bank.json').read_text())
    assert smoke_state['status']=='smoke_complete' and smoke_state['iteration']==528
    # Formal training resumes 527, not the deliberately separate smoke update.
    deadline=datetime.datetime.fromisoformat('2026-10-27T21:30:36.570789+08:00')
    assert deadline.timestamp()-time.time()>90,'Requested half-hour budget exhausted; do not extend'
    old=project/'runs/teacher_bank_fullref_duration200_yawsq_20261027_210024'
    old_state=json.loads((old/'teacher_bank.json').read_text());old_pid=old_state['pid']
    command_path=Path('/proc')/str(old_pid)/'cmdline'
    if command_path.exists():
        assert str(old) in command_path.read_bytes().decode().replace('\0',' ')
        atomic(old/'superseded.json',dict(reason='Restore v44 real-flight gate to exclude contact chatter; rebalance survival under added squared yaw cost',replacement_bank=str(bank),created_unix=time.time()))
        os.kill(old_pid,signal.SIGTERM)
        stop_limit=time.time()+60
        while command_path.exists() and command_path.read_bytes():
            assert time.time()<stop_limit,'Old worker did not save and exit'
            time.sleep(.5)
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
                 watchdog_pid=watchdog.pid,sealer_pid=sealer.pid,source_bank=str(parent),source_iteration=527,
                 task_tracking_profile='turn90_lateral_tracking_v1',step_quality_profile='strict_steps_v1',ppo_smoke_updates=1,
                 formal_resumed_smoke_weights=False,command=cmd)
    atomic(bank/'launch_receipt.json',receipt);atomic(here/'launch_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
