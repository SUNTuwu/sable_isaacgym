"""A curriculum progression gate, explicitly separate from full S3 acceptance."""
import argparse
import math
from pathlib import Path
from acceptance import assess,number
from evaluation_scenarios import scenarios
from run_io import read_json,atomic_json,sha256

OLD={'forward09','forward105','forward12','stop','start_stop','speed_switch'}
GOALS={'reverse':[-.04,0.,0.],'left':[0.,.04,0.],'right':[0.,-.04,0.],
       'turn_left':[0.,0.,.35],'turn_right':[0.,0.,-.35]}


def assess_frontier(report,endpoint,baseline,alpha):
    failures=[];limits=baseline['new_capability_limits']
    if report.get('horizon_s',0)<16 or report.get('environments_per_group',0)<4:
        failures.append('frontier_requires16s_and4env_per_family')
    if report.get('reference_execution_calls',1)!=0:failures.append('frontier_reference_dependency')
    if report.get('checkpoint_sha256')!=endpoint.get('checkpoint_sha256'):failures.append('checkpoint_mismatch')
    if endpoint.get('teacher_sha256')!=baseline['teacher_sha256']:failures.append('teacher_mismatch')
    if endpoint.get('reference_execution_calls',1) or endpoint.get('teacher_control'):failures.append('endpoint_reference_dependency')
    if endpoint.get('horizon_s',0)<16 or endpoint.get('num_envs',0)<56 or endpoint.get('suite')!='S3':
        failures.append('complete_endpoint_monitor_required')
    if {x['name']for x in endpoint.get('scenarios',[])}!={n for n,_ in scenarios('S3')}:
        failures.append('endpoint_scenarios_missing')
    for failure in assess('S3',[endpoint],baseline)['failures']:
        suffix=failure.split(':',1)[-1]
        if any(suffix.startswith(n+':') or suffix.startswith(n+'_segment')for n in OLD):
            failures.append('old_domain:'+failure)
    selected=[]
    for family,goal in GOALS.items():
        rows=[x for x in report.get('groups',[])if x.get('family')==family and abs(x.get('alpha',-1)-alpha)<1e-8]
        if len(rows)!=1:failures.append(family+':missing_or_duplicate');continue
        row=rows[0];selected.append(row);prefix=family+':'
        expected=[(1-alpha)*.105+alpha*goal[0],alpha*goal[1],alpha*goal[2]]
        command=row.get('actual_command')or[];actual=row.get('mean_actual')or[]
        if len(command)!=3 or any(not number(x) or abs(x-y)>1e-6 for x,y in zip(command,expected)):
            failures.append(prefix+'wrong_intermediate_command')
        if row.get('survival_fraction',0)<.9:failures.append(prefix+'survival')
        for field in ['xy_rmse_m_s','yaw_rmse_rad_s']:
            if not number(row.get(field)) or row[field]>limits[field]:failures.append(prefix+field)
        if len(actual)!=3 or not all(number(x)for x in actual):failures.append(prefix+'missing_motion');continue
        moving=math.hypot(*expected[:2])>.01 or abs(expected[2])>.08
        if moving:
            if row.get('minimum_steps_per_foot',0)<7 or row.get('four_foot_stepping_fraction',0)<.9:
                failures.append(prefix+'actual_four_foot_steps')
            frequency=row.get('actual_frequency_from_mean_interval_hz')or[]
            counts=row.get('interval_count_per_foot')or[]
            if len(frequency)!=4 or len(counts)!=4 or any(not number(f)or abs(f-1.)>.15 or c<3 for f,c in zip(frequency,counts)):
                failures.append(prefix+'actual_cadence')
        elif math.hypot(*actual[:2])>.02 or abs(actual[2])>.1:failures.append(prefix+'did_not_stop')
        for axis in range(3):
            if abs(expected[axis])>=(.02 if axis<2 else .1) and actual[axis]/expected[axis]<limits['mean_command_gain_min']:
                failures.append(prefix+'axis%d_insufficient_response'%axis)
    return dict(schema='s3_frontier_progression_v1',stage='S3_frontier',training_alpha=alpha,
        passed=not failures,failures=failures,checkpoint_sha256=report.get('checkpoint_sha256'),
        maximum_next_alpha=min(1.,alpha+.125),full_S3_accepted=False,selected_rows=selected,
        scope='One-seed intermediate curriculum progression only. All five direction paths, actual four-foot events/cadence, motion response and unchanged old-domain gates. Full S3 endpoints still require the original three-seed acceptance.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    p.add_argument('--endpoint-report',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--alpha',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=assess_frontier(read_json(a.report),read_json(a.endpoint_report),read_json(a.baseline),a.alpha)
    result['sources']={name:dict(path=str(path.resolve()),sha256=sha256(path))for name,path in
        [('frontier',a.report),('endpoint',a.endpoint_report),('baseline',a.baseline)]}
    atomic_json(a.output,result);print(result)


if __name__=='__main__':main()
