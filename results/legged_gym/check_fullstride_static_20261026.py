"""CPU-only reference, reward event-rate and speed-metric checks."""
import hashlib,json,math,sys,types
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
TEST=ROOT/'results/legged_gym/fullstride_probe_20261026';RT=TEST/'runtime';sys.path.insert(0,str(RT))
from isaacgym import gymapi
import numpy as np
import torch
from guided_reference import CapsuleKinematics,PAIRS
from sable_config import SableCfg
from sable_env import SableRobot
from swing_peak_balance import SwingPeakBalance
from velocity_metrics import VelocityMetrics
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=SableCfg();data=json.loads((RT/'assets/gait_reference.json').read_text());kin=CapsuleKinematics(RT/'assets/sable.xml',json.loads((RT/'assets/physical_manifest.json').read_text()))
assert cfg.rewards.scales.hip_stand_exp==0.
minimum=math.inf;rows=[];table=np.array(data['deltas_rad'])
for si,speed in enumerate(data['speeds_m_s']):
    peaks=np.zeros(4)
    for pi in range(table.shape[1]):
        q=dict(kin.q0)
        for f,pair in enumerate(PAIRS):
            for j,name in enumerate(pair):
                q[name]+=float(table[si,pi,f,j])
                minimum=min(minimum,q[name]-kin.limits[name][0],kin.limits[name][1]-q[name])
        peaks=np.maximum(peaks,kin.fk(q)[:,2]-kin.base[:,2])
    if speed:assert (peaks>.020).all()
    rows.append({'speed':speed,'applied_peak_m':peaks.tolist()})
assert minimum>0
def cost(dt,peak):
    tracker=SwingPeakBalance(1,'cpu',.8,dt);cfg.gait.period_s=.8
    fake=types.SimpleNamespace(cfg=cfg,dt=dt,peak_balance=tracker)
    offset=torch.tensor([[.75,.25,0.,.5]]);total=0.
    for k in range(round(6.4/dt)):
        t=k*dt;phase=torch.remainder(t/.8+offset,1.);u=((phase-.75)/.25).clamp(0,1)
        tracker.update(peak*torch.sin(math.pi*u).square(),phase>=.75,torch.tensor([t>=.8]),torch.full((1,4),.023))
        v=float(SableRobot._reward_swing_peak_shortfall(fake))
        if t>=3.2:total+=v*dt
    return total/3.2
adequate=cost(.02,.023);short=cost(.02,.011);fine=cost(.01,.011)
assert adequate==0 and abs(short-.25)<.01 and abs(short-fine)<.01
m=VelocityMetrics(2,.02);commands=np.array([[.03,0,0],[.09,0,0]])
velocity=np.array([[.05,.01,0],[.2,0,0]]);angular=np.zeros((2,3))
m.observe(.5,np.array([True,True]),commands,velocity,angular)
m.observe(1.1,np.array([True,False]),commands,velocity,angular)
v=m.result()['groups'];assert len(v)==1 and v[0]['samples']==1
assert abs(v[0]['rmse_velocity_xyz_m_s'][0]-.02)<1e-9 and abs(v[0]['forward_bias_m_s']-.02)<1e-9
report={'passed':True,'reference_rows':rows,'minimum_limit_margin_deg':math.degrees(minimum),
    'hip_posture_cost_disabled':True,'physical_hip_limits_rad':kin.limits['L_hip_pitch'].tolist(),
    'adequate_peak_cost':adequate,'half_target_peak_cost_per_second':short,'dt_consistent':True,
    'speed_metrics_initial_and_inactive_samples_excluded':True,'speed_metrics_known_error_passed':True,
    'script_sha256':sha(Path(__file__)),'runtime_files':{str(p.relative_to(RT)):sha(p) for p in RT.rglob('*') if p.is_file() and '__pycache__' not in str(p)}}
(TEST/'static_reward_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['reference_rows','runtime_files']}))
