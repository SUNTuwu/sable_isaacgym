"""Called by the active assistant: inspect evidence, publish stage, read notes."""
import fcntl,json,sys,time,os
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
def read(p,d=None):return json.loads(p.read_text()) if p.exists() else d
def atomic(p,d):
    temp=p.with_name(p.name+'.%d.partial'%os.getpid());temp.write_text(json.dumps(d,ensure_ascii=False,indent=2));temp.replace(p)
plan=read(OUT/'adaptation_round_plan_20261026.json');run=Path(plan['run'])
state=read(run/'status.json',{});ev=read(run/'latest_evaluation.json') or read(run/'baseline_evaluation.json',{})
priority=ev.get('priority_diagnostics',{}).get('median_across_environments',{})
arm={'label':'low_front','run':run.name,'front_mu':plan['front_floor_mu'],'status':state.get('status'),
     'iteration':state.get('iteration'),'updates':state.get('updates_this_run',0),'metrics':state.get('metrics',{}),
     'evaluation_iteration':ev.get('iteration'),'evaluation_is_baseline':not (run/'latest_evaluation.json').exists(),
     'summary':ev.get('summary',{}),'priority':priority,'peak_above10':ev.get('swing_peak_diagnostics',{}).get('fraction_above10mm_per_foot'),
     'hind20':ev.get('hind20_diagnostics'),'all20':ev.get('all20_diagnostics'),
     'hip_posture_penalty_weight':plan.get('hip_posture_penalty_weight'),'hip_preferred_max_deg':plan.get('hip_preferred_max_deg'),
     'backup':read(run/'backup_receipt.json')}
batch={'id':'friction_adaptation_20261026','headline':plan.get('headline','当前主线：后肢推进 · 前足摩擦逐步降至0'),
    'description':plan.get('description','目标：前足对地摩擦0、四足有效迈步峰值>20mm及速度跟随。当前是过渡课程，实际摩擦和高度见数据；功率仅作诊断，测力未通过校准。膝部触地终止。'),
    'status':'complete' if (run/'backup_receipt.json').exists() else 'training','deadline':plan['deadline'],
    'started_unix':state.get('started_unix'),'updated_unix':time.time(),'arms':[arm],
    'speed_step':read(OUT/'speed_step_preview_20261026.json')}
with (OUT/'operator_monitor_status.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);m=read(OUT/'operator_monitor_status.json',{})
    m.update(batch=batch,last_checked_unix=time.time(),finished=False,active_until=plan['deadline'])
    atomic(OUT/'operator_monitor_status.json',m)
sys.path.insert(0,str(ROOT/'deployment/legged_gym_knee_termination_20261025'))
from operator_feedback import messages
rows=messages();answered={r.get('reply_to') for r in rows if r['role']=='assistant'}
pending=[r for r in rows if r['role']=='user' and r['id'] not in answered]
previous=read(OUT/'adaptation_last_read_20261026.json',{})
record={'time':time.strftime('%H:%M:%S'),'checked_unix':time.time(),'run':run.name,
    'status':state.get('status'),'iteration':state.get('iteration'),'updates':state.get('updates_this_run'),
    'evaluation_iteration':ev.get('iteration'),'baseline':arm['evaluation_is_baseline'],
    'training':{k:state.get('metrics',{}).get(k) for k in ['mean_reward','mean_duration_s','fall_fraction','knee_ground_failure_fraction','fps']},
    'pending':pending}
if (previous.get('run'),previous.get('evaluation_iteration'),previous.get('baseline'))!=(run.name,ev.get('iteration'),arm['evaluation_is_baseline']):
    record.update(evaluation=arm['summary'],priority=priority,peaks_above10=arm['peak_above10'],hind20=arm['hind20'],all20=ev.get('all20_diagnostics'),velocity_tracking=ev.get('velocity_tracking'))
atomic(OUT/'adaptation_last_read_20261026.json',record)
with (OUT/'adaptation_agent_reads_20261026.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
display=dict(record)
for key in ['hind20','all20']:
    if isinstance(display.get(key),dict):display[key]={k:v for k,v in display[key].items() if k not in ['steps_per_environment','definition']}
if 'priority' in display:display['priority']={k:v for k,v in display['priority'].items() if k in ['hip_angle_median_deg','hip_preferred_fraction','hip_near_upper_limit_fraction','elbow_rom_p95_p05_deg','knee_surface_clearance_p05_m']}
if 'velocity_tracking' in display and display['velocity_tracking']:
    display['velocity_tracking']=[{'command':g['command_forward_m_s'],'actual':g['mean_actual_velocity_xyz_m_s'][0],'rmse':g['rmse_velocity_xyz_m_s'][0]} for g in display['velocity_tracking']['groups']]
print(json.dumps(display,ensure_ascii=False))
