"""Executed by the active assistant; never an unattended fake-read heartbeat."""
import json,sys,time
from pathlib import Path
from friction_batch_status import read,atomic,publish,OUT,ROOT
sys.path.insert(0,str(ROOT/'deployment/legged_gym_knee_termination_20261025'))
from operator_feedback import messages
plan=read(OUT/'friction_batch_plan.json');progress=publish(plan,agent_read=True)
rows=messages();answered={r.get('reply_to') for r in rows if r['role']=='assistant'}
pending=[r for r in rows if r['role']=='user' and r['id'] not in answered]
previous=read(OUT/'friction_batch_last_read.json',{})
record={'checked_unix':time.time(),'time':time.strftime('%H:%M:%S'),'status':plan['status'],'deadline':plan['deadline'],'pending':pending,'arms':[]}
for a in progress['arms']:
    r={k:a[k] for k in ['label','status','iteration','updates','evaluation_iteration','evaluation_is_baseline']}
    r['training']={k:a['metrics'].get(k) for k in ['mean_reward','mean_duration_s','fall_fraction','knee_ground_failure_fraction','fps']}
    previous_arm=next((b for b in previous.get('arms',[]) if b['label']==a['label']),{})
    if (previous_arm.get('evaluation_iteration'),previous_arm.get('evaluation_is_baseline'))!=(r['evaluation_iteration'],r['evaluation_is_baseline']):
        r.update(new_evaluation=a['summary'],priority=a['priority'],peak_above10=a['peak_above10'])
    record['arms'].append(r)
atomic(OUT/'friction_batch_last_read.json',record)
with (OUT/'friction_batch_agent_reads.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
print(json.dumps(record,ensure_ascii=False))
