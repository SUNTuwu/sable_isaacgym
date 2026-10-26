"""Read-only training inspection plus an honest operator-message read heartbeat.

Run by the active assistant, never as an unattended heartbeat substitute.
"""
import json,sys,time,datetime
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
sys.path.insert(0,str(ROOT/'deployment/legged_gym_knee_termination_20261025'))
from operator_feedback import pending
from run_io import read_json,atomic_json,append_json
run=Path(read_json(ROOT/'runs/current.json')['run_dir'])
s=read_json(run/'status.json',{});m=s.get('metrics',{})
old=read_json(ROOT/'results/legged_gym/last_monitor_read.json',{})
record={'time':time.time(),'local_time':time.strftime('%H:%M:%S'),'run':run.name,'status':s.get('status'),
        'iteration':s.get('iteration'),'updates':s.get('updates_this_run'),'state_age_s':time.time()-s.get('updated_unix',0),
        'training':{k:m.get(k) for k in ['mean_reward','mean_duration_s','episodes_recent','fall_fraction','knee_ground_failure_fraction','noise_std','learning_rate']},
        'pending_feedback':pending()}
ev=read_json(run/'latest_evaluation.json',{})
record['evaluation_iteration']=ev.get('iteration')
if ev and (old.get('evaluation_iteration')!=ev.get('iteration') or old.get('run')!=run.name):
    record['new_evaluation']={'iteration':ev['iteration'],'summary':ev['summary'],
        'priority':ev.get('priority_diagnostics',{}).get('median_across_environments'),
        'strict_steps':[sum(s['steps_per_foot'][i] for s in ev['samples']) for i in range(4)]}
if record.get('new_evaluation') and ev.get('swing_peak_diagnostics'):
    record['new_evaluation']['swing_peaks']={k:v for k,v in ev['swing_peak_diagnostics'].items() if k!='samples'}
review=read_json(run/'posture_review.json')
if review:
    record['posture_review_iteration']=review['iteration']
    if old.get('posture_review_iteration')!=review['iteration']:record['posture_review']=review
    decision=read_json(run/'posture_review_decision.json')
    record['review_decision']=decision.get('action') if decision else 'pending'
if s.get('posture_review_after_unix') and not review:
    cutoff=datetime.datetime.fromisoformat(s['deadline']).timestamp()
    if s['posture_review_after_unix']>=cutoff-600:record['posture_review_plan']='covered_by_final_replay; scheduled15min review is inside last10min window'
    else:record['seconds_until_posture_review']=s['posture_review_after_unix']-time.time()
completion=read_json(run/'completion.json')
if completion:record['completion']={k:completion[k] for k in ['training_exit_code','reason','final_evaluation']}
atomic_json(ROOT/'results/legged_gym/last_monitor_read.json',record)
append_json(ROOT/'results/legged_gym/agent_monitor_reads.jsonl',record)
print(json.dumps(record,ensure_ascii=False))
