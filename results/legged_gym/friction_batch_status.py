"""Concurrent batch progress with an honest, separate assistant-read timestamp."""
import fcntl,json,time,os
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
OUT=ROOT/'results/legged_gym'

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def atomic(path,value):
    tmp=path.with_name(path.name+'.%d.partial'%os.getpid());tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def publish(plan,agent_read=False):
    arms=[]
    for arm in plan['arms']:
        run=Path(arm['run']);state=read(run/'status.json',{});ev=read(run/'latest_evaluation.json') or read(run/'baseline_evaluation.json')
        baseline=read(run/'baseline_evaluation.json')
        diagnostic=ev.get('priority_diagnostics',{}).get('median_across_environments',{}) if ev else {}
        peaks=ev.get('swing_peak_diagnostics',{}) if ev else {}
        arms.append({'label':arm['label'],'run':run.name,'front_mu':arm['front_scale'],
          'status':state.get('status','preflight'),'iteration':state.get('iteration'),
          'updates':state.get('updates_this_run',0),'metrics':state.get('metrics',{}),
          'evaluation_iteration':ev.get('iteration') if ev else None,
          'evaluation_is_baseline':not (run/'latest_evaluation.json').exists(),
          'summary':ev.get('summary',{}) if ev else {},'priority':diagnostic,
          'peak_above10':peaks.get('fraction_above10mm_per_foot'),
          'baseline_summary':baseline.get('summary',{}) if baseline else {},
          'backup':read(run/'backup_receipt.json')})
    data={'id':plan['id'],'updated_unix':time.time(),'started_unix':plan.get('started_unix'),
          'deadline':plan['deadline'],'status':plan.get('status','training'),
          'num_envs_per_arm':plan['num_envs_per_arm'],'parent_iteration':5314,'arms':arms,
          'scope':'Two parallel runs, same parent/seed/rewards/optimizer. Different front friction; compare independent gait, not return alone. No validated friction-power measurement.'}
    atomic(OUT/'friction_batch_progress.json',data)
    with (OUT/'operator_monitor_status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        status=read(OUT/'operator_monitor_status.json',{})
        status['batch']=data
        if agent_read:status.update(last_checked_unix=time.time(),finished=False,active_until=plan['deadline'])
        atomic(OUT/'operator_monitor_status.json',status)
    return data

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--agent-read',action='store_true');a=p.parse_args()
    plan=read(OUT/'friction_batch_plan.json')
    print(json.dumps(publish(plan,a.agent_read),ensure_ascii=False))
