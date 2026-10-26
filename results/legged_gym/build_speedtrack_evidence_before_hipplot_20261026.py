"""Build a checkpoint evidence index and static chart from saved JSON only."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(path,kind):
    d=read(path);checkpoint=Path(d['checkpoint']);assert sha(checkpoint)==d['checkpoint_sha256']
    group=[];all20=d.get('all20_diagnostics',{}).get('steps_per_environment')
    for v in d.get('velocity_tracking',{}).get('groups',[]):
        speed=v['command_forward_m_s'];ids=[i for i,s in enumerate(d['samples']) if s['command_m_s'] is not None and abs(s['command_m_s']-speed)<1e-6]
        group.append(dict(v,survived=sum(not d['samples'][i]['fell'] for i in ids),num_environments=len(ids),
            all20_success=sum(min(all20[i])>=2 and not d['samples'][i]['fell'] and d['samples'][i]['displacement_m']>.15 for i in ids) if all20 else None))
    return {'kind':kind,'path':str(path),'report_sha256':sha(path),'checkpoint':str(checkpoint),'checkpoint_sha256':d['checkpoint_sha256'],
      'iteration':d['iteration'],'created_unix':d['created_unix'],'horizon_s':d['horizon_s'],
      'summary':d['summary'],'all20':{k:v for k,v in d.get('all20_diagnostics',{}).items() if k!='steps_per_environment'},
      'swing_peak':{k:v for k,v in d.get('swing_peak_diagnostics',{}).items() if k!='samples'},'velocity_groups':group}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--suffix',default='');args=parser.parse_args()
    plan=read(OUT/'adaptation_round_plan_20261026.json');run=args.run.resolve() if args.run else Path(plan['run']);rows=[]
    suffix='_'+args.suffix if args.suffix else ''
    baseline=run/'baseline_evaluation.json'
    if baseline.exists():rows.append(summarize(baseline,'transfer_baseline'))
    for path in sorted((run/'evaluations').glob('*.json')):
        d=read(path)
        if d.get('horizon_s')==16 and d.get('command_profile','constant')=='constant' and d.get('all20_diagnostics'):
            rows.append(summarize(path,'final' if path.name.startswith('final_') else 'trained'))
    rows.sort(key=lambda x:(x['iteration'],x['kind']))
    payload={'created_unix':time.time(),'run':run.name,'deadline':plan['deadline'],'reports':rows,
       'scope':'Same configuration within this stage only. Deterministic same-seed24-environment tests; descriptive evidence, not independent statistical significance. Transfer baseline is not newly learned.',
       'generator_sha256':sha(Path(__file__))}
    target=OUT/('speedtrack_evidence_20261026'+suffix+'.json');tmp=target.with_suffix('.partial');tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2));tmp.replace(target)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');x=[r['iteration'] for r in rows]
    axes[0,0].plot(x,[r['summary']['survival_16s']*100 for r in rows],'o-',label='16s survival')
    axes[0,0].plot(x,[r['all20']['four_feet_2steps_forward_survive_fraction']*100 for r in rows],'s-',label='All4 valid20mm steps + forward')
    axes[0,0].set(ylabel='Environments (%)',ylim=(-3,105),title='Survival and all-feet20mm success');axes[0,0].legend(fontsize=8)
    colors=['#3979ac','#d89d2d','#458958']
    for speed,color in zip([.03,.06,.09],colors):
        pairs=[(r['iteration'],v) for r in rows for v in r['velocity_groups'] if abs(v['command_forward_m_s']-speed)<1e-6]
        axes[0,1].plot([a for a,b in pairs],[b['mean_actual_velocity_xyz_m_s'][0] for a,b in pairs],'o-',color=color,label=f'command{speed:.2f}m/s')
        axes[0,1].axhline(speed,color=color,ls=':',alpha=.6)
        axes[1,0].plot([a for a,b in pairs],[b['rmse_velocity_xyz_m_s'][0] for a,b in pairs],'o-',color=color,label=f'{speed:.2f}m/s')
    axes[0,1].set(title='Measured forward speed (dotted: command)',ylabel='m/s');axes[0,1].legend(fontsize=8)
    axes[1,0].set(title='Forward velocity RMSE',ylabel='m/s');axes[1,0].legend(fontsize=8)
    for foot,name in enumerate(['LF','RF','LH','RH']):axes[1,1].plot(x,[r['all20']['completed_swing_peak_above20mm_fraction_per_foot'][foot]*100 for r in rows],'o-',label=name)
    axes[1,1].set(title='Completed swing peaks strictly above20mm',ylabel='Swings (%)',ylim=(-3,105));axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('Checkpoint iteration')
    fig.suptitle(run.name+' / speed tracking / all-feet20mm | first point: transfer baseline\nFlat simulation, front friction0.02. Same seed; no hardware validation.',fontsize=12)
    chart=OUT/('speedtrack_progress_20261026'+suffix+'.png');fig.savefig(chart,dpi=150);plt.close(fig)
    receipt={'created_unix':time.time(),'json_sha256':sha(target),'png_sha256':sha(chart),'reports':len(rows),'simulation_executed':False}
    (OUT/('speedtrack_progress_20261026'+suffix+'_receipt.json')).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))
if __name__=='__main__':main()
