"""Verify the actual remote slow-lift transfer, one update, replay and deadline proof."""
from isaacgym import gymapi
import torch,json,time,shutil
from pathlib import Path
from run_io import atomic_json,sha256,read_json
code=Path(__file__).resolve().parent;root=code.parent.parent;run=root/'runs/lift_transfer_smoke_v2'
parent=root/'runs/stand5_20261025_v6_crawl/checkpoints/model_001000.pt'
old=torch.load(str(parent),map_location='cpu')
initial=torch.load(str(run/'checkpoints/model_001000.pt'),map_location='cpu')
final=torch.load(str(run/'checkpoints/model_001001.pt'),map_location='cpu')
checks={'weights_preserved':all(torch.equal(v,initial['model_state_dict'][k]) for k,v in old['model_state_dict'].items() if k!='std'),
    'std_reset_006':torch.allclose(initial['model_state_dict']['std'],torch.full_like(initial['model_state_dict']['std'],.06)),
    'optimizer_initially_empty':not initial['optimizer_state_dict']['state'],
    'trained_std_above_floor':bool((final['model_state_dict']['std']>=.04).all()),'one_update_completed':final['iter']==1001}
cfg=read_json(run/'config.json');checks['lift_config']=cfg['phase']=='lift' and cfg['env']['gait']['period_s']==.8 and cfg['train']['runner']['min_policy_std']==.04
replay=read_json(code/'verification/lift_replay_smoke_v2.json')
checks['replay_commands']=replay['command_set_m_s']==[.025,.04,.06] and replay['horizon_s']==2
manifest=read_json(run/'runtime_manifest.json');checks['tested_current_train']=manifest['files']['train.py']==sha256(code/'train.py')
assert all(checks.values()),checks
report={'passed':True,'checks':checks,'parent_sha256':sha256(parent),'train_sha256':sha256(code/'train.py'),
    'evaluator_sha256':sha256(code/'evaluate.py'),'created_unix':time.time(),'scope':'One remote PPO update and 2s structural replay, not gait validation'}
atomic_json(code/'verification/lift_transfer.json',report);print('LIFT_TRANSFER',json.dumps(report))
test=code/'verification/deadline_smoke_1792881860';receipt=read_json(test/'backup_receipt.json')
assert sha256(receipt['archive'])==receipt['sha256']
assert sha256(test/'runtime/supervise.py')==sha256(code/'supervise.py')
dest=code/'verification/deadline_stop_backup_verified.tar.gz';shutil.copy2(receipt['archive'],dest)
proof={'passed':True,'tested_same_supervisor':True,'supervisor_sha256':sha256(code/'supervise.py'),
    'test_archive':dest.name,'test_archive_sha256':sha256(dest),'created_unix':time.time()}
atomic_json(code/'verification/deadline_validation.json',proof);print('DEADLINE_PROOF',json.dumps(proof))
