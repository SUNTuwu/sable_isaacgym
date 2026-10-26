"""Publish verified diagnostic evidence to the dashboard monitor snapshot."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
p=argparse.ArgumentParser();p.add_argument('report',type=Path);a=p.parse_args();report=a.report.resolve();d=json.loads(report.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert d['command_profile']=='steps' and d['horizon_s']==24 and d['num_envs']==24
checkpoint=Path(d['checkpoint']);assert sha(checkpoint)==d['checkpoint_sha256']
preview={'created_unix':time.time(),'run':checkpoint.parent.parent.name,'iteration':d['iteration'],
 'checkpoint_sha256':d['checkpoint_sha256'],'report':str(report),'report_sha256':sha(report),
 'horizon_s':24,'heading_control':d.get('heading_control'),'priority_diagnostics':d.get('priority_diagnostics'),'stride_progress_diagnostics':d.get('stride_progress_diagnostics'),'summary':d['summary'],'velocity_tracking':d['velocity_tracking'],
 'trace':[{k:r[k] for k in ['t','command_m_s','actual_velocity_m_s']} for r in d['trace']],
 'scope':'Single-environment curve. Group speed metrics use24original episodes while alive; the twozero-command periods are aggregated. The legacy summary key survival_16s here refers to completing the requested24s horizon.'}
p=OUT/'speed_step_preview_20261026.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(preview,ensure_ascii=False,indent=2));tmp.replace(p)
print(json.dumps({k:v for k,v in preview.items() if k not in ['trace','priority_diagnostics','stride_progress_diagnostics']},ensure_ascii=False))
