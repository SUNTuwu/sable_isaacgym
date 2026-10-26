"""Summarize independent evening evaluations; file processing only, no RL."""
from pathlib import Path
import json,hashlib,time,datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
tz=datetime.timezone(datetime.timedelta(hours=8))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stages=[];all_rows=[]
for run in sorted((ROOT/'runs').glob('stand5_20261025_v*')):
    if not run.is_dir() or '_transfer_' in run.name:continue
    version=int(run.name.split('_v')[1].split('_')[0])
    if version<14:continue
    if not (run/'config.json').exists():continue
    cfg=read(run/'config.json');state=read(run/'status.json');rows=[]
    for report in sorted((run/'evaluations').glob('*.json')):
        ev=read(report)
        if ev.get('schema')!='sable-independent-evaluation/1' or ev.get('zero_actions'):continue
        cp=Path(ev['checkpoint']);verified=cp.exists() and sha(cp)==ev['checkpoint_sha256']
        assert verified,str(cp)
        row={'run':run.name,'version':version,'iteration':ev['iteration'],'created_unix':ev['created_unix'],'report':str(report),'report_sha256':sha(report),'checkpoint':str(cp),'checkpoint_sha256':ev['checkpoint_sha256'],'checkpoint_verified':verified,'num_envs':ev['num_envs'],'horizon_s':ev['horizon_s'],'summary':ev['summary'],'priority':ev['priority_diagnostics']['median_across_environments'],'swing_peaks':{k:v for k,v in ev.get('swing_peak_diagnostics',{}).items() if k!='samples'},'strict_steps':[sum(s['steps_per_foot'][i] for s in ev['samples']) for i in range(4)]}
        rows.append(row)
    rows.sort(key=lambda x:x['created_unix']);all_rows.extend(rows)
    baseline=None
    bp=run/'baseline_evaluation.json'
    if bp.exists():
        b=read(bp);bc=Path(b['checkpoint'])
        if not bc.exists():bc=run/'preflight/transfer_smoke/checkpoints'/bc.name
        assert bc.exists() and sha(bc)==b['checkpoint_sha256'],str(bc)
        baseline={'report':str(bp),'report_sha256':sha(bp),'checkpoint':str(bc),'checkpoint_sha256':b['checkpoint_sha256'],'checkpoint_verified':True,'created_unix':b['created_unix'],'summary':b['summary'],'priority':b['priority_diagnostics']['median_across_environments'],'swing_peaks':{k:v for k,v in b.get('swing_peak_diagnostics',{}).items() if k!='samples'},'strict_steps':[sum(s['steps_per_foot'][i] for s in b['samples']) for i in range(4)],'kind':'initial transfer baseline; no new-stage learning'}
    stages.append({'run':run.name,'version':version,'status':state['status'],'started_unix':state['started_unix'],'deadline':state['deadline'],'tolerance_deg':cfg['env'].get('posture',{}).get('hip_stand_tolerance_deg'),'hip_penalty_weight':cfg['env']['rewards']['scales'].get('hip_stand_exp',0),'reference_strength':cfg['env']['reference']['strength'],'swing_height_target_m':cfg['env']['gait']['swing_height_m']*cfg['env']['reference']['strength'],'swing_peak_balance_weight':cfg['env']['rewards']['scales'].get('swing_peak_balance',0),'clearance_penalty_weight':cfg['env']['rewards']['scales']['gait_clearance'],'initial_baseline':baseline,'evaluations':rows,'backup_receipt':read(run/'backup_receipt.json') if (run/'backup_receipt.json').exists() else None,'archive_verification':read(run/'final_verification.json') if (run/'final_verification.json').exists() else None})
all_rows.sort(key=lambda x:x['created_unix'])
catalog={'created_unix':time.time(),'scope':'Current +24deg hip-limit evening runs, independent24-environment16s evaluations only. New reward/deadband stages are separated; differences are descriptive, not controlled causal tests. Server-local evidence only. Checkpoint files freshly SHA256-verified; archive verification references the existing per-run verification receipts.','stages':stages}
out=OUT/'evening_checkpoint_catalog.json';out.write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
fig,axes=plt.subplots(3,2,figsize=(14,10),sharex=True);axes=axes.ravel()
colors=['#c98938','#4f81b3','#4f9c72','#a367b2']
for stage in stages:
    rows=stage['evaluations']
    if not rows:continue
    times=[datetime.datetime.fromtimestamp(r['created_unix'],tz) for r in rows]
    for side in [0,1]:
        axes[0].plot(times,[r['priority'].get('hip_angle_median_deg',[float('nan')]*2)[side] for r in rows],'.-',color=colors[side],label=['Left','Right'][side] if stage==stages[0] else None)
        axes[1].plot(times,[r['priority'].get('knee_surface_clearance_p05_m',[float('nan')]*2)[side]*1000 for r in rows],'.-',color=colors[side])
        axes[2].plot(times,[r['priority'].get('elbow_rom_p95_p05_deg',[float('nan')]*2)[side] for r in rows],'.-',color=colors[side])
    for foot in range(4):
        axes[3].plot(times,[r['priority']['foot_clearance_p95_m'][foot]*1000 for r in rows],'.-',color=colors[foot],label=['LF','RF','LH','RH'][foot] if stage==stages[0] else None)
        axes[5].plot(times,[r['strict_steps'][foot] for r in rows],'.-',color=colors[foot])
    axes[4].plot(times,[r['summary']['survival_16s']*100 for r in rows],'.-',color='#4f81b3',label='16s survival' if stage==stages[0] else None)
    axes[4].plot(times,[r['summary']['moving_all_four_fraction']*100 for r in rows],'.-',color='#a367b2',label='Four-foot forward gait' if stage==stages[0] else None)
    b=stage['initial_baseline']
    if b:
        bt=datetime.datetime.fromtimestamp(b['created_unix'],tz)
        for panel,key,scale in [(0,'hip_angle_median_deg',1),(1,'knee_surface_clearance_p05_m',1000),(2,'elbow_rom_p95_p05_deg',1),(3,'foot_clearance_p95_m',1000)]:
            for j,value in enumerate(b['priority'].get(key,[])):
                if value is not None:axes[panel].scatter([bt],[value*scale],marker='s',facecolors='none',edgecolors=colors[j],s=27)
        for j,value in enumerate(b['strict_steps']):axes[5].scatter([bt],[value],marker='s',facecolors='none',edgecolors=colors[j],s=27)
        for key,color in [('survival_16s','#4f81b3'),('moving_all_four_fraction','#a367b2')]:axes[4].scatter([bt],[b['summary'][key]*100],marker='s',facecolors='none',edgecolors=color,s=27)
    x=datetime.datetime.fromtimestamp(stage['started_unix'],tz)
    for ax in axes:ax.axvline(x,color='grey',alpha=.3,linestyle=':')
    axes[0].annotate('v'+str(stage['version']),(x,.98),xycoords=('data','axes fraction'),va='top',fontsize=9)
axes[0].axhline(-14,color='black',linestyle='--',alpha=.5,label='Authored standard -14 deg')
axes[1].axhline(.5,color='red',linestyle=':',alpha=.5)
axes[3].axhline(2,color='grey',linestyle='--',alpha=.5)
axes[4].set_ylim(-3,103)
titles=['Hip angle: per-environment median (deg)','Knee surface clearance: per-env p05 (mm)','Elbow range: per-env p95-p05 (deg)','Foot clearance: per-environment p95 (mm)','Independent survival / four-foot success (%)','Strict steps summed over 24 environments']
for ax,title in zip(axes,titles):
    ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=tz))
for i in [0,3,4]:axes[i].legend(fontsize=8,loc='best')
for ax in axes[4:]:ax.set_xlabel('2026-10-25, UTC+08')
fig.suptitle('Evening evidence: separate lines for separate training stages; hollow squares = initial transfer baselines\nv15: +/-10 deg, weight0.5; v16: +/-5 deg, weight0.5; v17+: +/-5 deg, weight2.5\nv14-18: 50% reference; v19+: 75% reference; v20+: clearance weight -32; v21: target13.5mm + paired swing-peak cost',fontsize=10)
fig.text(.02,.015,'Angle / height / ROM: median across per-environment statistics, excluding initial 1s. Baselines are not new-stage learning.\nStrict steps: >=2mm / >=60ms / >=5mm; stage comparisons are descriptive, not controlled causal tests or hardware validation.',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.96]);image=OUT/'evening_progress.png';fig.savefig(image,dpi=150);plt.close(fig)
receipt={'created_unix':time.time(),'catalog':str(out),'catalog_sha256':sha(out),'plot':str(image),'plot_sha256':sha(image),'source_sha256':sha(Path(__file__)),'evaluations':len(all_rows),'stages':len(stages),'scope':'File-only summary; no simulator or RL imported.'};(OUT/'evening_progress_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
