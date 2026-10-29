"""Pure candidate rules, not imported by the running monitor.

An accepted=False stationary policy is a plateau, not automatically a rollback.
These broad extreme-task gates supplement the unchanged survival<.75 rule.
The caller must first choose a provenance-matched report (existing monitor SHA
check), and must retain its original recovery, cooldown and baseline selection.
"""
import math


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def first_episode_task_metrics(report):
    """Allow latched first-episode diagnostics, or complete no-fall raw reports.

    Legacy cp28 had one fall and no first-episode mask; its lift maximum cannot
    be used as a trustworthy initial-episode chatter reference.
    """
    if not isinstance(report,dict):return None
    corrected=(report.get('evaluation_metric_scope','').startswith('Initial episode only')
               and report.get('finite_tracking_data') is True
               and report.get('source_checkpoint_iteration')==report.get('checkpoint_iteration'))
    unreset=report.get('survival_fraction')==1. and finite(report.get('valid_tracking_samples')) and report['valid_tracking_samples']>0
    if not (corrected or unreset):return None
    amplitude=report.get('amplitude_deg');budget=report.get('liftoff_budget')
    rmse=report.get('angle_rmse_deg');peak=report.get('peak_error_deg');counts=report.get('max_liftoffs_per_foot')
    if not (all(finite(x) for x in (amplitude,budget,rmse,peak)) and amplitude>0 and budget>=1 and rmse>=0 and peak>=0):return None
    if not isinstance(counts,(list,tuple)) or len(counts)!=4 or not all(finite(x) and x>=0 for x in counts):return None
    return dict(amplitude_deg=amplitude,liftoff_budget=budget,rmse_deg=rmse,peak_deg=peak,max_lifts=max(counts))


def yaw_task_severity(report,reference=None):
    """Return evidence only. Never changes a policy, control file or baseline."""
    result=dict(severe=False,comparable=False,reasons=[],reference_used=False)
    if not isinstance(report,dict) or not report:return result
    survival=report.get('survival_fraction')
    if finite(survival) and survival<.75:
        return dict(result,severe=True,comparable=True,reasons=['survival_below_original_0.75_gate'])
    data=first_episode_task_metrics(report)
    if data is None:return dict(result,reason='no trustworthy initial-episode task metrics')
    a=data['amplitude_deg'];budget=data['liftoff_budget']
    limits=dict(rmse_deg=a,peak_deg=3*a,max_lifts=10*budget)
    anchor=first_episode_task_metrics(reference)
    # Only a no-fall stable task reference can relax these broad gates. This is
    # not a new accepted checkpoint or a replacement for the protected fallback.
    if (anchor is not None and reference.get('survival_fraction')==1.
            and anchor['rmse_deg']<=.8*anchor['amplitude_deg']
            and anchor['peak_deg']<=1.5*anchor['amplitude_deg']
            and anchor['max_lifts']<=3*anchor['liftoff_budget']):
        limits['rmse_deg']=max(limits['rmse_deg'],1.4*anchor['rmse_deg']*a/anchor['amplitude_deg'])
        limits['peak_deg']=max(limits['peak_deg'],2*anchor['peak_deg']*a/anchor['amplitude_deg'])
        limits['max_lifts']=max(limits['max_lifts'],3*anchor['max_lifts']*budget/anchor['liftoff_budget']+1)
        result['reference_used']=True
    reasons=[]
    if data['rmse_deg']>=limits['rmse_deg']:reasons.append('tracking_rmse_extreme')
    if data['peak_deg']>=limits['peak_deg']:reasons.append('angle_error_peak_extreme')
    if data['max_lifts']>=limits['max_lifts']:reasons.append('repeated_liftoffs_extreme')
    return dict(result,severe=bool(reasons),comparable=True,reasons=reasons,thresholds=limits,metrics=data)


def review_distinct_checkpoint(state,run,report,reference=None):
    """Candidate two-report guard. Caller handles an eventual reload itself.

    Repeated/stale checkpoints do not count. A normal or incomparable report
    interrupts the streak. Run/stage/budget changes start a new comparison scope.
    """
    state=dict(state or {})
    cp=report.get('checkpoint_iteration')
    if not isinstance(cp,int) or isinstance(cp,bool):
        return state,dict(rollback_evidence=False,reason='missing checkpoint identity')
    scope=(str(run),report.get('amplitude_deg'),report.get('liftoff_budget'))
    previous_scope=tuple(state.get('scope',()))
    if scope!=previous_scope:state={'scope':list(scope),'last_checkpoint':None,'count':0}
    last=state.get('last_checkpoint')
    if last is not None and cp<=last:
        return state,dict(rollback_evidence=False,reason='duplicate or stale checkpoint')
    verdict=yaw_task_severity(report,reference)
    before=state.get('count',0)
    count=before+1 if verdict['severe'] and verdict['comparable'] else 0
    state.update(last_checkpoint=cp,count=count,last_verdict=verdict)
    return state,dict(verdict,rollback_evidence=before<2 and count>=2,
                      consecutive_severe_checkpoints=count,
                      reason='two distinct consecutive severe reports' if before<2 and count>=2 else 'collect or clear evidence')
