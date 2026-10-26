"""Remote single simulator check for gait clock, capsule heights and 50D observations."""
import argparse,json,math
from pathlib import Path
from isaacgym import gymapi
import torch,numpy as np
from preflight import create_env
from sable_config import SableCfg
from run_io import atomic_json,sha256
p=argparse.ArgumentParser();p.add_argument('--period-s',type=float,default=.5);p.add_argument('--guided',action='store_true');p.add_argument('--reference-strength',type=float,default=.25);a=p.parse_args()
cfg=SableCfg();cfg.gait.enabled=True;cfg.gait.period_s=a.period_s;cfg.env.num_observations=50;cfg.commands.heading_command=True
if a.guided:
    cfg.reference.enabled=True;cfg.reference.strength=a.reference_strength;cfg.env.num_observations=52;cfg.gait.height_profile='sine_squared'
    cfg.gait.contact_phase_independent_of_strength=True;cfg.gait.clearance_relative_to_target=True
env=create_env(4,noise=False,cfg=cfg);env.reset_noise=0.;env.fixed_command=.06 if a.guided else .08;env.reset()
env.gait_phase_offset[:]=0.;env.episode_length_buf[:]=0
peaks=torch.zeros(4,device=env.device);max_swing=0;clock_error=0.;clear_errors=[];finite=True
contact_peaks=torch.zeros_like(peaks);max_contact_swing=0
for _ in range(150):
    obs,_,_,_,_=env.step(torch.zeros(4,12,device=env.device))
    finite &= bool(torch.isfinite(obs).all()) and obs.shape==(4,cfg.env.num_observations)
    clock_error=max(clock_error,float((obs[:,48:50].square().sum(1)-1).abs().max()))
    peaks=torch.maximum(peaks,env.gait_target_clearance.max(0).values)
    contact_peaks=torch.maximum(contact_peaks,env.gait_swing_weight.max(0).values)
    max_contact_swing=max(max_contact_swing,int((env.gait_swing_weight>1e-6).sum(1).max()))
    max_swing=max(max_swing,int((env.gait_target_clearance>1e-6).sum(1).max()))
    mask=env.contact_forces[:,env.feet_indices,2]>2
    clear_errors.extend(env.foot_clearance[mask].abs().cpu().tolist())
    finite &= bool(torch.isfinite(env._reward_gait_contact()).all()&torch.isfinite(env._reward_gait_clearance()).all())
geometry_error=float(np.percentile(clear_errors,95))
expected_peak=.006*(cfg.reference.strength if a.guided else 1.)
passed=finite and max_swing<=1 and bool((peaks>expected_peak*.91).all()) and bool((peaks<=expected_peak+1e-6).all()) and clock_error<1e-6 and geometry_error<.003
passed=passed and max_contact_swing<=1 and bool((contact_peaks>.99).all()) and bool((contact_peaks<=1).all())
folder=Path(__file__).parent
report={'passed':passed,'guided':a.guided,'reference_strength':cfg.reference.strength,'period_s':cfg.gait.period_s,'obs_shape':list(obs.shape),'max_simultaneous_swing_targets':max_swing,'per_foot_peak_target_m':peaks.cpu().tolist(),
    'contact_phase_independent_of_strength':cfg.gait.contact_phase_independent_of_strength,'clearance_relative_to_target':cfg.gait.clearance_relative_to_target,
    'per_foot_peak_swing_contact_weight':contact_peaks.cpu().tolist(),'max_simultaneous_contact_swing':max_contact_swing,
    'clock_unit_circle_max_error':clock_error,'contact_clearance_p95_m':geometry_error,
    'source_sha256':{n:sha256(folder/n) for n in ['sable_env.py','sable_config.py','gait_support.py','guided_reference.py','assets/gait_reference.json']}}
atomic_json(folder/'verification/gait_signals.json',report);print('GAIT_SIGNALS',json.dumps(report),flush=True);env.gym.destroy_sim(env.sim)
if not passed:raise SystemExit(2)
