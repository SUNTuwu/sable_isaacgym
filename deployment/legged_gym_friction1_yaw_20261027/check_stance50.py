from isaacgym import gymapi
import json,hashlib,types,torch
from pathlib import Path
from sable_config import SableCfg
from gait_support import GaitSignals
from ankle_phase import PhaseAnkleMotion
D=Path(__file__).parent
cfg=SableCfg();cfg.gait.height_profile='sine_squared'
n=200;env=types.SimpleNamespace(cfg=cfg,device='cpu',num_envs=n,dt=.005,feet_indices=torch.arange(4),body_names=['L_wrist_fixed','R_wrist_fixed','L_ankle_pitch','R_ankle_pitch'],episode_length_buf=torch.arange(n),gait_phase_offset=torch.zeros(n),rigid_states=torch.zeros(n,4,13),commands=torch.zeros(n,4))
env.rigid_states[:,:,6]=1.;env.commands[:,0]=.1
g=GaitSignals(env);g.update(env)
phase=torch.remainder(g.phase(env)[:,None]+g.offsets,1.)
swing=phase>=cfg.gait.stance_fraction
assert cfg.gait.period_s==1. and cfg.gait.stance_fraction==.5
assert torch.equal(swing.sum(0),torch.full((4,),100)) and torch.equal(swing.sum(1),torch.full((n,),2))
assert torch.allclose(env.gait_target_clearance.max(0).values,torch.tensor([.023]*4),atol=1e-7)
assert (env.gait_target_clearance[~swing]==0).all()
ankle=PhaseAnkleMotion(1,'cpu',.02,period=cfg.gait.period_s,stance_fraction=cfg.gait.stance_fraction,landing_deg=cfg.ankle_phase.landing_angle_deg,takeoff_deg=cfg.ankle_phase.takeoff_angle_deg)
assert ankle.stance_duration==ankle.swing_duration==.5
ref=json.loads((D/'assets/gait_reference.json').read_text());t=torch.tensor(ref['deltas_rad']);assert ref['period_s']==1. and ref['stance_fraction']==.5 and ref['max_ik_error_m']<2e-6 and ref['minimum_limit_margin_deg']>0
assert torch.allclose(t[:,0],t[:,-1],atol=1e-5)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert ref['builder_sha256']==sha(D/'guided_reference.py') and ref['source_asset_sha256']==sha(Path(cfg.asset.file)) and ref['source_manifest_sha256']==sha(D/'assets/physical_manifest.json')
report={'passed':True,'period_s':1.,'frequency_hz_per_foot':1.,'stance_fraction':.5,'stance_duration_s':.5,'swing_duration_s':.5,'phase_offsets':g.offsets.tolist(),'simultaneous_swing_feet':2,'height_target_m_per_foot':env.gait_target_clearance.max(0).values.tolist(),'reference_max_ik_error_m':ref['max_ik_error_m'],'reference_minimum_limit_margin_deg':ref['minimum_limit_margin_deg'],'horizontal_span_m_at_max_reference_speed':.09*.5,'note':'Reference stance travel retains commanded ground speed, so horizontal span scales with the reduced stance duration.80mm reward threshold remains an independent learning objective.','source_sha256':{name:sha(D/name) for name in ['sable_config.py','train.py','guided_reference.py','gait_support.py','ankle_phase.py','assets/gait_reference.json']}}
(D/'verification/stance50_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
