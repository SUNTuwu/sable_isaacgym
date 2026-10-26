"""Audit the tested 50->52 input transfer and fixed-reference experiment."""
from isaacgym import gymapi
import torch,json,time
from pathlib import Path
from run_io import atomic_json,sha256,read_json
code=Path(__file__).resolve().parent;root=code.parent.parent;run=root/'runs/guided_transfer_smoke_v2'
parent=root/'runs/stand5_20261025_v6_crawl/checkpoints/model_001000.pt'
old=torch.load(str(parent),map_location='cpu')
initial=torch.load(str(run/'checkpoints/model_001000.pt'),map_location='cpu')
final=torch.load(str(run/'checkpoints/model_001001.pt'),map_location='cpu')
checks={}
for k,v in old['model_state_dict'].items():
    new=initial['model_state_dict'][k]
    if k=='std':continue
    if k in ['actor.0.weight','critic.0.weight']:
        checks[k+'_preserved']=torch.equal(v,new[:,:50]) and new.shape[1]==52 and bool((new[:,50:]==0).all())
    else:checks[k+'_preserved']=torch.equal(v,new)
checks.update(std_reset_004=torch.allclose(initial['model_state_dict']['std'],torch.full_like(initial['model_state_dict']['std'],.04)),
    optimizer_initially_empty=not initial['optimizer_state_dict']['state'],
    trained_std_above_floor=bool((final['model_state_dict']['std']>=.04).all()),one_update_completed=final['iter']==1001)
cfg=read_json(run/'config.json')
checks['guided_config']=cfg['phase']=='guided' and cfg['env']['reference']['enabled'] and cfg['env']['reference']['strength']==.25 and cfg['env']['gait']['height_profile']=='sine_squared' and cfg['env']['rewards']['tracking_sigma']==.000625
replay=read_json(code/'verification/guided_quarter_parent.json');zero=read_json(code/'verification/guided_quarter_reference_only.json')
checks['replay_commands']=replay['command_set_m_s']==[.03,.045,.06] and replay['horizon_s']==16 and replay['num_envs']==24
checks['controller_attributed']=replay['kinematic_reference_enabled'] and zero['zero_policy_residual'] and zero['kinematic_reference_enabled']
checks['parent_survival_gate']=replay['summary']['survival_16s']>=.9
checks['both_replays_finite']=all(torch.isfinite(torch.tensor(v)).all() for report in [replay,zero] for v in report['summary'].values())
manifest=read_json(run/'runtime_manifest.json')
sources=['train.py','sable_env.py','sable_config.py','gait_support.py','guided_reference.py','assets/gait_reference.json']
checks['tested_current_sources']=all(manifest['files'][n]==sha256(code/n) for n in sources)
assert all(checks.values()),checks
report={'passed':True,'checks':checks,'parent_sha256':sha256(parent),'source_sha256':{n:sha256(code/n) for n in sources+['evaluate.py']},
    'reference_only_summary':zero['summary'],'parent_with_reference_summary':replay['summary'],
    'created_unix':time.time(),'scope':'One remote PPO update, exact transfer audit, and two 16s deterministic baselines; not gait acceptance'}
atomic_json(code/'verification/guided_transfer.json',report);print('GUIDED_TRANSFER',json.dumps(report))
