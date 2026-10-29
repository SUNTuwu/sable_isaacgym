"""Bounded periodic inference for one exact live bank; never changes training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path,data):
    tmp=path.with_name(path.name+'.monitor.tmp')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        return (stat[19],p.joinpath('cmdline').read_bytes())
    except FileNotFoundError:return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',required=True,type=Path)
    args=parser.parse_args();bank=args.bank.resolve()
    root=Path(os.environ['SABLE_GYM_ROOT']).resolve();assert bank.parent==root/'runs'
    s=json.loads((bank/'teacher_bank.json').read_text());pid=s['pid'];expected=identity(pid)
    assert expected and str(bank).encode() in expected[1]
    deadline=datetime.datetime.fromisoformat(s['deadline']).timestamp()
    python=str(Path(os.environ['SABLE_GYM_ENV'])/'bin/python')
    evaluator=bank/'runtime/evaluate_teacher_bank_small.py';core=bank/'runtime/teacher_bank_core.py'
    env=dict(os.environ,PYTHONPATH=str(core.parent)+os.pathsep+os.environ.get('PYTHONPATH',''))
    path=bank/'periodic_pace_monitor.json';next_time=time.time()+180;last_iteration=None
    history=[]
    while time.time()<deadline-255 and identity(pid)==expected:
        if time.time()<next_time:
            time.sleep(min(5,next_time-time.time()));continue
        next_time=time.time()+180
        state=json.loads((bank/'teacher_bank.json').read_text())
        if state['status']!='training':break
        receipt=json.loads((bank/'02_STRAFE_LEFT/checkpoint.json').read_text())
        if receipt['iteration']==last_iteration:continue
        lock=bank/'small_evaluation.lock'
        if lock.exists():
            old=json.loads(lock.read_text())
            if identity(int(old['pid'])) is not None:continue
        last_iteration=receipt['iteration']
        atomic(path,dict(status='evaluating',pid=os.getpid(),training_pid=pid,
          deadline=s['deadline'],history=history,interval_s=180,never_updates_policy=True))
        cmd=[python,str(evaluator),'--bank',str(bank),'--core',str(core),'--max-wall-s','240']
        with (bank/'periodic_pace_evaluation.log').open('a') as log:
            try:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=245)
                entry={'unix':time.time(),'returncode':result.returncode}
            except subprocess.TimeoutExpired:
                entry={'unix':time.time(),'failed':'bounded_evaluator_timeout'}
        reports=[]
        for ident in ('02_STRAFE_LEFT','03_STRAFE_RIGHT'):
            p=bank/ident/'latest_evaluation.json'
            if p.exists():
                report=json.loads(p.read_text())
                reports.append({k:report.get(k) for k in ('teacher_id','iteration','survival_fraction',
                    'xy_rmse_m_s','yaw_rmse_rad_s','pace_dense_diagnostics','pace_acceptance')})
        entry['pace_reports']=reports;history.append(entry)
        atomic(path,dict(status='monitoring',pid=os.getpid(),training_pid=pid,
          deadline=s['deadline'],history=history,interval_s=180,never_updates_policy=True))
    atomic(path,dict(status='finished',pid=os.getpid(),history=history,
      deadline=s['deadline'],final_evaluation_owner='existing finalization supervisor'))


if __name__=='__main__':main()
