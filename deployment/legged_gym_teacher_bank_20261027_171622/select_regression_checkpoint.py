"""Select an S2 continuation parent by independent regression, never reward."""
import math
from pathlib import Path
from run_io import read_json,sha256


def select_checkpoint(run, baseline):
    candidates=[]
    limits=baseline['regression_limits']
    for path in sorted((Path(run)/'evaluations').glob('independent_*.json')):
        report=read_json(path)
        if report.get('suite')!='S2' or report.get('horizon_s',0)<16 or report.get('reference_execution_calls',1):continue
        rows=report.get('scenarios',[])
        if {r['name']for r in rows}!={'forward09','forward105','forward12','stop','start_stop','speed_switch'}:continue
        cp=Path(report['checkpoint'])
        if cp.parent.resolve()!=Path(run).resolve()/'checkpoints' or not cp.exists():continue
        if sha256(cp)!=report['checkpoint_sha256']:raise ValueError('Evaluated checkpoint changed')
        survival_fail=sum(r['survival_fraction']<.9 for r in rows)
        metric_fail=0;excess=0.
        for row in rows:
            metrics=[(row['xy_rmse_m_s'],limits['xy_rmse_max_m_s'])]
            if row['name'].startswith('forward'):metrics.append((row['forward_rmse_m_s'],limits['forward_rmse_max_m_s']))
            for value,maximum in metrics:
                if value is None or not math.isfinite(value):metric_fail+=1;excess+=100.
                else:metric_fail+=int(value>maximum);excess+=max(value/maximum-1.,0.)
        candidates.append({'checkpoint':str(cp),'sha256':report['checkpoint_sha256'],'iteration':report['iteration'],
            'evaluation':str(path),'evaluation_sha256':sha256(path),
            'score':[survival_fail,metric_fail,excess,-report['iteration']]})
    if not candidates:raise RuntimeError('No independent S2 regression reports to select from')
    winner=min(candidates,key=lambda row:row['score'])
    cp={'path':winner['checkpoint'],'sha256':winner['sha256'],'iteration':winner['iteration']}
    return cp,{'selected':winner,'candidates':candidates,
        'rule':'Minimize number of scenario survival failures, metric failures, normalized excess; latest breaks ties',
        'scope':'Development parent selection only; fixed monitor seed is not multi-seed acceptance',
        'acceptance_passed':False}
