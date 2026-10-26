"""Summarize actual per-command speed/stride evidence; file processing only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('report',type=Path);a=p.parse_args()
r=json.loads(a.report.read_text());groups=[]
for command in r['command_set_m_s']:
    ids=[i for i,s in enumerate(r['samples']) if s['command_m_s']==command]
    samples=[r['samples'][i] for i in ids]
    peak=[r['swing_peak_diagnostics']['samples'][i] for i in ids]
    counts=np.array([s['counts'] for s in peak]).sum(0);above=np.array([s['above10mm'] for s in peak]).sum(0)
    strides=[r['forward_stride_diagnostics']['per_env_median_m'][i] for i in ids]
    hips=[r['priority_diagnostics']['samples'][i]['hip_angle_median_deg'] for i in ids]
    median_dx=float(np.median([s['displacement_m'] for s in samples]))
    groups.append({'command_m_s':command,'num_envs':len(ids),'survived':sum(not s['fell'] for s in samples),
        'four_foot_forward':sum(not s['fell'] and min(s['steps_per_foot'])>=2 and s['displacement_m']>.15 for s in samples),
        'median_forward_m':median_dx,'median_net_speed_over_16s_m_s':median_dx/r['horizon_s'],
        'hip_median_deg':np.median(hips,axis=0).tolist(),
        'completed_swings_per_foot':counts.tolist(),'swings_above10mm_per_foot':above.tolist(),
        'fraction_swings_above10mm_per_foot':(above/np.maximum(counts,1)).tolist(),
        'median_forward_stride_m_per_foot':[float(np.median([s[j] for s in strides if s[j] is not None])) if any(s[j] is not None for s in strides) else None for j in range(4)]})
result={'report':str(a.report.resolve()),'report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),
        'checkpoint_sha256':r['checkpoint_sha256'],'groups':groups,
        'scope':'Eight nominal-friction deterministic samples per command, no optimization. Command groups have different sampled starts; not a paired causal test. Net speed uses full intended16s horizon including failures.'}
out=a.report.with_name(a.report.stem+'_summary.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
