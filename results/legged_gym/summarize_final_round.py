"""Same-course initial/learned checkpoint comparison; no simulator or RL."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
run=a.run.resolve();out=Path(__file__).resolve().parent
completion=json.loads((run/'completion.json').read_text())
final_path=Path(completion['final_evaluation']['report'])
paths=[run/'baseline_evaluation.json']+sorted((run/'evaluations').glob('*.json'))
rows=[]
for path in paths:
    r=json.loads(path.read_text())
    if r.get('schema')!='sable-independent-evaluation/1':continue
    checkpoint=Path(r['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
    groups=[]
    for command in r['command_set_m_s']:
        ids=[i for i,s in enumerate(r['samples']) if s['command_m_s']==command]
        samples=[r['samples'][i] for i in ids]
        peaks=[r['swing_peak_diagnostics']['samples'][i] for i in ids]
        counts=np.array([s['counts'] for s in peaks]).sum(0);above=np.array([s['above10mm'] for s in peaks]).sum(0)
        travel=[r['priority_diagnostics']['samples'][i].get('strict_step_travel_median_m',[None]*4) for i in ids]
        groups.append({'command_m_s':command,'num_envs':len(ids),'survived':sum(not s['fell'] for s in samples),
            'median_net_speed_m_s':float(np.median([s['displacement_m'] for s in samples])/r['horizon_s']),
            'fraction_swings_above10mm_per_foot':(above/np.maximum(counts,1)).tolist(),
            'median_3d_foot_center_step_travel_m':[float(np.median([v[j] for v in travel if v[j] is not None])) if any(v[j] is not None for v in travel) else None for j in range(4)]})
    rows.append({'iteration':r['iteration'],'kind':'initial transfer baseline' if path.name=='baseline_evaluation.json' else 'learned checkpoint',
        'is_final':path.resolve()==final_path.resolve(),'report':str(path),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'checkpoint_sha256':r['checkpoint_sha256'],'summary':r['summary'],'priority':r['priority_diagnostics']['median_across_environments'],
        'swing_peaks':{k:v for k,v in r['swing_peak_diagnostics'].items() if k!='samples'},'command_groups':groups})
assert all(r['summary']['survival_16s']==1 and r['summary']['moving_all_four_fraction']==1 and r['summary']['knee_ground_failure_fraction']==0 for r in rows)
rows.sort(key=lambda r:r['iteration']);final=next(r for r in rows if r['is_final'])
result={'created_unix':time.time(),'run':run.name,'rows':rows,'final':final,
        'scope':'Same frozen v22 course,24 environments16s,0.03/0.06/0.09m/s. Descriptive correlated checkpoint comparisons, not independent significance testing. Peak>10mm is a separate metric from the2mm/60ms/5mm strict step criterion. Step travel is3D foot-center takeoff-to-landing distance, not exact contact-point slip.'}
target=out/'round_acceptance_comparison.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
fig,axes=plt.subplots(2,2,figsize=(11,7));x=[r['iteration'] for r in rows];colors=['#c98938','#4f81b3','#4f9c72','#a367b2']
for foot,name in enumerate(['LF','RF','LH','RH']):axes[0,0].plot(x,[r['swing_peaks']['fraction_above10mm_per_foot'][foot]*100 for r in rows],'.-',label=name,color=colors[foot])
axes[0,0].set_title('Completed swing peaks >10mm (%)');axes[0,0].set_ylim(0,103);axes[0,0].legend()
axes[0,1].axhspan(-19,-9,color='#4f9c72',alpha=.12,label='Standard +/-5deg')
axes[0,1].axhline(-14,color='grey',linestyle='--')
for side,name in enumerate(['Left','Right']):axes[0,1].plot(x,[r['priority']['hip_angle_median_deg'][side] for r in rows],'.-',label=name,color=colors[side])
axes[0,1].set_title('Hip angle median (deg)');axes[0,1].legend()
for i,command in enumerate([.03,.06,.09]):
    axes[1,0].plot(x,[r['command_groups'][i]['median_net_speed_m_s'] for r in rows],'.-',label='Command %.3f'%command,color=colors[i])
    axes[1,0].axhline(command,color=colors[i],linestyle=':',alpha=.5)
axes[1,0].set_title('Net forward speed over16s (m/s)');axes[1,0].legend()
for i,name in enumerate(['Front pair','Hind pair']):axes[1,1].plot(x,[r['swing_peaks']['paired_peak_rmse_m_front_hind'][i]*1000 for r in rows],'.-',label=name,color=colors[i])
axes[1,1].set_title('Opposite completed-swing peak difference RMS (mm)');axes[1,1].legend()
for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('Checkpoint iteration');ax.axvline(x[0],color='grey',linestyle=':',alpha=.4)
fig.suptitle('v22 actual same-course evidence; first point is transfer baseline, not new learning')
fig.text(.02,.01,'Each checkpoint:24/24 survival and four-foot forward gait; knee-ground termination0/24. Nominal simulation only.',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95]);plot=out/'round_acceptance_comparison.png';fig.savefig(plot,dpi=150);plt.close(fig)
receipt={'created_unix':time.time(),'data_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'plot_sha256':hashlib.sha256(plot.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'round_acceptance_comparison_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'final_iteration':final['iteration'],'groups':final['command_groups'],'receipt':receipt}))
