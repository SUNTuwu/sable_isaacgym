"""Summarize an actual independent replay, without simulator imports."""
import argparse,json,statistics,hashlib,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
e=json.loads(a.report.read_text());d=e['priority_diagnostics'];rows=d['samples'];out={'created_unix':time.time(),'evaluation':str(a.report),'evaluation_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),'iteration':e['iteration'],'summary':e['summary'],'hip_reference_deg':d['hip_reference_deg'],'tolerance_deg':d['hip_tolerance_deg'],'median_across_environments':d['median_across_environments'],'command_groups':[],'scope':'Raw observed posture diagnostics; no automatic kneeling label or optimization'}
for cmd in e['command_set_m_s']:
 ids=[i for i,s in enumerate(e['samples']) if abs(s['command_m_s']-cmd)<1e-6];valid=[i for i in ids if rows[i].get('samples_after_1s',0)]
 group={'command_m_s':cmd,'num_envs':len(ids),'survived':sum(not e['samples'][i]['fell'] for i in ids),'knee_ground_failures':sum(e['samples'][i].get('knee_ground_failure',False) for i in ids),'valid_post_settling_envs':len(valid)}
 for key in ['hip_angle_median_deg','hip_stand_abs_deviation_median_deg','hip_within_stand_tolerance_fraction','knee_surface_clearance_p05_m','elbow_rom_p95_p05_deg']:
  if valid:group[key]={'min':[min(rows[i][key][j] for i in valid) for j in range(2)],'median':[statistics.median(rows[i][key][j] for i in valid) for j in range(2)],'max':[max(rows[i][key][j] for i in valid) for j in range(2)]}
 group['strict_steps_per_foot']=[sum(e['samples'][i]['steps_per_foot'][j] for i in ids) for j in range(4)];out['command_groups'].append(group)
if a.output:a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
