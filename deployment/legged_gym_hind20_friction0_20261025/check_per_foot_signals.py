"""CPU checks of front/hind targets, normalization and peak-pair symmetry."""
from pathlib import Path
import hashlib,json,types
from isaacgym import gymapi
import torch
from sable_config import SableCfg,MODEL
from gait_support import GaitSignals
from sable_env import SableRobot
from swing_peak_balance import SwingPeakBalance

cfg=SableCfg();cfg.gait.enabled=True;cfg.gait.period_s=.8
cfg.gait.height_profile='sine_squared';cfg.gait.clearance_relative_to_target=True
cfg.reference.enabled=True;cfg.reference.strength=.75
env=types.SimpleNamespace(cfg=cfg,device='cpu',num_envs=1,
    feet_indices=torch.arange(4),body_names=MODEL['foot_bodies'],dt=.02,
    episode_length_buf=torch.tensor([100.]),gait_phase_offset=torch.zeros(1),
    commands=torch.tensor([[.03,0,0,0]]),rigid_states=torch.zeros(1,4,13))
env.rigid_states[:,:,6]=1.;env.gait_signals=GaitSignals(env)
env.motion_reference=types.SimpleNamespace(speed=torch.tensor([.03]),ramp=lambda age:torch.full_like(age,.75))
peaks=torch.zeros(4)
for k in range(80):
    env.episode_length_buf[:]=100+k;env.gait_signals.update(env)
    peaks=torch.maximum(peaks,env.gait_target_clearance[0])
expected=torch.tensor([.0135,.0135,.023,.023])
assert torch.allclose(peaks,expected,atol=1e-7),(peaks,expected)
before=env.gait_signals.heights.clone();env.gait_target_clearance=torch.zeros(1,4)
env.foot_clearance=expected[None,:]
for _ in range(1000):assert abs(float(SableRobot._reward_gait_clearance(env))-1)<1e-6
assert torch.equal(before,env.gait_signals.heights)
tracker=SwingPeakBalance(1,'cpu',.8,.02)
for _ in range(2):
    tracker.update(expected[None,:],torch.ones(1,4,dtype=torch.bool),torch.tensor([True]),expected[None,:])
    value=tracker.update(torch.zeros(1,4),torch.zeros(1,4,dtype=torch.bool),torch.tensor([True]),expected[None,:])
    assert float(value)==0
here=Path(__file__).resolve().parent
result={'passed':True,'actual_target_peaks_m':peaks.tolist(),
    'repeated_reward_preserves_height_tensor':True,'unit_relative_height_error_cost':1.,
    'per_foot_equal_pair_cost_zero':True,
    'source_files':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in
        ['check_per_foot_signals.py','sable_env.py','sable_config.py','guided_reference.py',
         'gait_support.py','swing_peak_balance.py','evaluate.py','train.py','supervise.py']},
    'scope':'CPU tensor behavior only; dynamic preflight pending'}
(here/'verification/per_foot_height_signals.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
