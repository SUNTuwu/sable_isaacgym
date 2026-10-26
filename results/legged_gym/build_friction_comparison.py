"""Read-only plots and evidence index for the two-arm friction experiment."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
OUT=ROOT/'results/legged_gym'
POST=ROOT/'postrun/front_friction_20261025'
POST.mkdir(parents=True,exist_ok=True)

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(p,label,kind):
    d=read(p);s=d['summary'];n=d['num_envs']
    return dict(policy=label,kind=kind,path=str(p),sha256=sha(p),iteration=d['iteration'],
        checkpoint_sha256=d['checkpoint_sha256'],material_friction=d.get('material_friction'),
        num_envs=n,horizon_s=d['horizon_s'],summary=s,
        command_groups=[dict(command_m_s=v,n=sum(x['command_m_s']==v for x in d['samples']),
          survived=sum(x['command_m_s']==v and not x['fell'] for x in d['samples']),
          median_forward_m=float(np.median([x['displacement_m'] for x in d['samples'] if x['command_m_s']==v])))
          for v in d['command_set_m_s']],
        total_strict_steps_per_foot=np.sum([x['steps_per_foot'] for x in d['samples']],axis=0).tolist(),
        priority=d.get('priority_diagnostics',{}).get('median_across_environments',{}),
        swing_peak={k:v for k,v in d.get('swing_peak_diagnostics',{}).items() if k!='samples'},
        foot_order=d['foot_order'])

plan=read(OUT/'friction_batch_plan.json');rows=[];series={}
for arm in plan['arms']:
    run=Path(arm['run']);label=arm['label'];points=[]
    baseline=run/'baseline_evaluation.json'
    rows.append(summarize(baseline,label,'parent_baseline'));points.append(rows[-1])
    candidates=list((run/'evaluations').glob('*.json'))
    seen=set()
    for p in sorted(candidates):
        d=read(p)
        if 'summary' not in d or 'iteration' not in d or d.get('num_envs')!=24:continue
        key=(d['iteration'],d['checkpoint_sha256'])
        if key in seen:continue
        seen.add(key);points.append(summarize(p,label,'independent_evaluation'))
    latest=run/'latest_evaluation.json'
    if latest.exists():
        d=read(latest);key=(d['iteration'],d['checkpoint_sha256'])
        if key not in seen:points.append(summarize(latest,label,'independent_evaluation'))
    points.sort(key=lambda x:x['iteration']);series[label]=points
    rows.extend(points[1:])
    cross=POST/(label+'_cross_friction.json')
    if cross.exists():rows.append(summarize(cross,label,'cross_friction'))

fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
colors={'low_front':'#087f8c','control':'#c7772e'}
names={'low_front':'Mainline: front mu 0.05','control':'Control: front mu 1.0'}
for label,points in series.items():
    x=[r['iteration']-5314 for r in points];color=colors[label]
    for ax,key,factor in [(axes[0,0],'median_displacement_m',1),(axes[0,1],'survival_16s',100),(axes[1,0],'moving_all_four_fraction',100)]:
        ax.plot(x,[r['summary'][key]*factor for r in points],'o-',color=color,label=names[label])
    axes[1,1].plot(x,[np.mean(r['priority']['hip_stand_abs_deviation_median_deg']) for r in points],'o-',color=color,label=names[label])
for ax,title,ylabel in zip(axes.flat,['Forward displacement in16s','Survived full16s','Four-foot forward criterion','Hip deviation from standard stand'],['Median (m)','Environments (%)','Environments (%)','Mean of L/R environment medians (deg)']):
    ax.set_title(title);ax.set_xlabel('New PPO updates (0 = parent baseline)');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
axes[0,0].legend();axes[0,1].set_ylim(0,105);axes[1,0].set_ylim(0,105)
axes[1,1].axhline(5,color='gray',ls='--',lw=1,label='5deg free band')
fig.suptitle('Front-friction experiment | 24 environments per evaluation | one seed per arm\nDifferent physical conditions; not a validated foot-force measurement',fontsize=12)
plot=POST/'friction_comparison.png';fig.savefig(plot,dpi=150);plt.close(fig)
report=dict(created_unix=time.time(),batch=plan['id'],status=plan['status'],deadline=plan['deadline'],
  primary_arm='low_front',source_sha256=sha(Path(__file__)),plot_sha256=sha(plot),evaluations=rows,
  scope='Nominal deterministic24-environment simulation, one training seed per arm. Different friction domains; count and wall budget are both retained. No calibrated per-foot propulsion-force claim.',
  strict_step='Capsule bottom>=2mm for>=60ms, then landed after>=5mm foot-center movement; initial0.5s excluded.',
  swing_peak='Separate measure: each complete planned swing peak above10mm, no60ms requirement.')
(POST/'friction_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'status':plan['status'],'rows':len(rows),'plot':str(plot),'report':str(POST/'friction_comparison.json')}))
