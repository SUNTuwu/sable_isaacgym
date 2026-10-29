"""Pure candidate selector; it neither changes basic() nor writes live settings.

Use only for a translation that failed the existing basic gate after the usual
stable-objective review window. Return multipliers for the specific failed
quantity, not a blanket velocity increase:

* Mean progress outside 60-140% or excessive mean along-speed error: velocity.
* Excessive mean cross-speed or measured cross-track excursion: cross_motion.
* Excessive final integrated heading: heading_integral, never raw yaw_sigma.
* Mean motion and path already pass, but raw XY/yaw RMS fails and contact phase
  agreement is below .85 or verified reference cadence is exceeded: contact_timing
  by at most 1.1. Do not change velocity or heading for this case.

Cadence is optional. No period or one-lift assumption is made. A caller may add
reference_motion={verified_against_checkpoint:True, checkpoint_sha256:..., 
reference_sha256:..., period_s:..., liftoffs_per_cycle:[four actual counts]}.
The checkpoint SHA must match this evaluation. Only full-survival evaluations
with explicit horizon_s and num_envs are used; ceil(horizon/period) provides a
conservative per-foot reference event bound, including noninteger periods.

No acceleration change is inferred from stochastic training reward averages or
10 Hz replay derivatives. High-rate deterministic evidence is needed for that.
The caller must also remove the old unconditional translation yaw_sigma update;
otherwise choosing contact_timing would still silently change the yaw objective.
"""
import math


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def vector(value,size):
    return isinstance(value,(list,tuple)) and len(value)==size and all(finite(x) for x in value)


def sha256(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdefABCDEF' for c in value)


def verified_reference_lift_excess(e):
    """None means unknown, rather than an invented one-lift reference."""
    reference=e.get('reference_motion')
    if not isinstance(reference,dict) or reference.get('verified_against_checkpoint') is not True:return None
    checkpoint=e.get('checkpoint')
    if not isinstance(checkpoint,dict):return None
    if not sha256(checkpoint.get('sha256')) or reference.get('checkpoint_sha256')!=checkpoint['sha256']:return None
    if not sha256(reference.get('reference_sha256')):return None
    period=reference.get('period_s');horizon=e.get('horizon_s');envs=e.get('num_envs')
    expected=reference.get('liftoffs_per_cycle');actual=e.get('actual_liftoffs_per_foot')
    if not (finite(period) and period>0 and finite(horizon) and horizon>0):return None
    if not isinstance(envs,int) or isinstance(envs,bool) or envs<=0 or e.get('survival_fraction')!=1:return None
    if not vector(expected,4) or not vector(actual,4):return None
    if any(x<0 or x!=int(x) for x in expected) or any(x<0 for x in actual):return None
    cycles=math.ceil(horizon/period-1e-9)
    bounds=[envs*cycles*x for x in expected]
    return any(observed>bound for observed,bound in zip(actual,bounds))


def translation_failure_adjustments(e,factor):
    """Return existing reward-name multipliers, without mutating the report."""
    if not isinstance(e,dict) or not finite(factor) or factor<=1:return {}
    actual=e.get('mean_actual');command=e.get('mean_command')
    if not vector(actual,3) or not vector(command,3) or abs(command[2])>1e-6:return {}
    survival=e.get('survival_fraction')
    if not finite(survival) or survival<.95:return {}  # Existing survival branch handles this.
    # Mixing heading-frame means with an anchored path command recreates the old
    # circling bug; absent frame provenance is not permission to guess.
    if not str(e.get('velocity_frame','')).startswith('episode-start heading'):return {}
    speed=math.hypot(*command[:2])
    if speed<1e-6:return {}
    dx,dy=command[0]/speed,command[1]/speed
    along=actual[0]*dx+actual[1]*dy
    cross=actual[0]*dy-actual[1]*dx
    tolerance=max(.025,.6*speed)  # Identical to the existing basic XY-RMS bound.
    factor=min(factor,1.2);adjustments={}
    # A small command can overshoot badly while remaining inside the absolute
    # 25 mm/s RMS floor. Route that failed gate to velocity, not contact timing.
    # This is a diagnostic branch only; the existing basic gate stays unchanged.
    if along/speed<.6 or along/speed>1.4 or abs(along-speed)>tolerance:adjustments['v2_velocity']=factor
    cross_track=e.get('maximum_cross_track_m');heading=e.get('max_abs_final_heading_deg')
    if abs(cross)>tolerance or (finite(cross_track) and cross_track>.10):adjustments['v2_cross_motion']=factor
    if finite(heading) and heading>15:adjustments['v2_heading_integral']=factor
    if adjustments:return adjustments
    # Missing path evidence cannot be interpreted as a passing path.
    if not finite(cross_track) or not finite(heading):return {}
    raw_xy=e.get('xy_rmse_m_s');raw_yaw=e.get('yaw_rmse_rad_s')
    raw_failure=(finite(raw_xy) and raw_xy>tolerance) or (finite(raw_yaw) and raw_yaw>.25)
    if not raw_failure:return {}
    agreement=e.get('contact_phase_agreement')
    if (finite(agreement) and agreement<.85) or verified_reference_lift_excess(e) is True:
        return {'v2_contact_timing':min(factor,1.1)}
    return {}
