"""Five-minute evidence snapshots, without simulator imports or optimization."""
import time
from run_io import atomic_json, append_json, read_json


def publish(run, final=False):
    state = read_json(run / 'status.json', {})
    ev = read_json(run / 'latest_evaluation.json', {})
    rows = []
    path = run / 'metrics.jsonl'
    if path.exists():
        import json
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row['time'] >= time.time() - 300:
                    rows.append(row)
            except (KeyError, ValueError):
                pass
    rewards = [r['mean_reward'] for r in rows if r.get('mean_reward') is not None]
    report = {'created_unix': time.time(), 'interval_s': 300, 'final': final,
              'run': run.name, 'status': state.get('status', 'preflight'),
              'deadline': state.get('deadline'), 'iteration': state.get('iteration'),
              'updates_this_run': state.get('updates_this_run', 0),
              'updates_in_last_5min': len(rows),
              'reward_mean_last_5min': sum(rewards) / len(rewards) if rewards else None,
              'priorities': ['跪姿/膝与躯干接触', '肘关节活动幅度', '四足离地高度与持续时间', '严格步数与步幅'],
              'evaluation_iteration': ev.get('iteration'), 'evaluation_created_unix': ev.get('created_unix'),
              'evaluation_age_s': time.time() - ev['created_unix'] if ev.get('created_unix') else None,
              'summary': ev.get('summary'), 'priority_diagnostics': ev.get('priority_diagnostics', {}).get('median_across_environments'),
              'strict_steps_total_per_foot': [sum(s['steps_per_foot'][j] for s in ev.get('samples', [])) for j in range(4)] if ev else None,
              'accepted_locomotion': False,
              'note': '训练回报与独立评估分别记录；膝/躯干净接触力是诊断代理量，不能单独证明跪姿已消除。'}
    atomic_json(run / 'progress_5min.json', report)
    append_json(run / 'progress_5min.jsonl', report)
    return report

# Publish automatic metrics independently of assistant note-reading timestamps.
_original_publish=publish
def publish(run,final=False):
    report=_original_publish(run,final)
    import fcntl,json,os
    from pathlib import Path
    root=Path('/root/gpufree-data/projects/sable-isaacgym');out=root/'results/legged_gym'
    if read_json(root/'runs/current.json',{}).get('run_dir')!=str(run):return report
    intent=read_json(run/'stage_intent.json',{});state=read_json(run/'status.json',{})
    ev=read_json(run/'latest_evaluation.json') or read_json(run/'baseline_evaluation.json',{})
    arm={'label':'low_front','run':run.name,'front_mu':.02,'status':state.get('status'),'iteration':state.get('iteration'),'updates':state.get('updates_this_run',0),'metrics':state.get('metrics',{}),'evaluation_iteration':ev.get('iteration'),'evaluation_is_baseline':not (run/'latest_evaluation.json').exists(),'summary':ev.get('summary',{}),'priority':ev.get('priority_diagnostics',{}).get('median_across_environments',{}),'knees':ev.get('knee_posture_diagnostics'),'all20':ev.get('all20_diagnostics'),'hind_stance':ev.get('hind_stance_diagnostics'),'ankles':ev.get('ankle_diagnostics'),'hip_preferred_max_deg':intent.get('hip_preferred_max_deg',-19),'backup':read_json(run/'backup_receipt.json')}
    with (out/'operator_monitor_status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);m=read_json(out/'operator_monitor_status.json',{})
        m['automatic_metrics']=True;m['batch']={'id':'knee81_stride100_20261026','headline':intent.get('headline'),'description':intent.get('description'),'status':'final_evaluating' if final else 'training','deadline':state.get('deadline'),'updated_unix':time.time(),'arms':[arm]}
        atomic_json(out/'operator_monitor_status.json',m)
    return report
