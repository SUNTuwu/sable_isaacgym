from isaacgym import gymapi
import torch,json,hashlib
from pathlib import Path
D=Path(__file__).parent;ROOT=D.parents[1];T=ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026';P=ROOT/'runs/stand5_20261026_v28_ankle_inline/checkpoints/model_006621.pt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=torch.load(str(P),map_location='cpu');a=torch.load(str(T/'checkpoints/model_006621.pt'),map_location='cpu');cp=json.loads((T/'checkpoint.json').read_text());b=torch.load(cp['path'],map_location='cpu')
assert all(torch.equal(v,a['model_state_dict'][k]) for k,v in p['model_state_dict'].items() if k!='std')
assert torch.allclose(a['model_state_dict']['std'][:-2],torch.full((12,),.04)) and torch.allclose(a['model_state_dict']['std'][-2:],torch.full((2,),.08))
assert not a['optimizer_state_dict']['state'] and b['optimizer_state_dict']['state']
for prefix in ['actor.','critic.']:assert any(not torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith(prefix))
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values()) and (b['model_state_dict']['std']>=.04).all()
cfg=json.loads((T/'config.json').read_text());assert cfg['env']['posture']['hip_preferred_max_deg']==-19
sc=cfg['env']['rewards']['scales'];assert sc['hind_foot_angle']==-4 and sc['ankle_stance_motion']==2 and sc['ankle_stance_reversal']==-.5
assert cfg['env']['env']['num_observations']==58 and cfg['mirror_policy']
status=json.loads((T/'status.json').read_text());assert status['status']=='stopped_at_deadline'
done=json.loads((T/'completion.json').read_text());assert done['training_exit_code']==0 and done['reason']=='deadline'
backup=json.loads((T/'backup_receipt.json').read_text());assert sha(Path(backup['archive']))==backup['sha256'] and sha(Path(cp['path']))==cp['sha256']
assert done['final_evaluation']['return_code']==0
result={'passed':True,'exact_actor_critic_transfer':True,'fresh_optimizer':True,'actual_ppo_actor_critic_update':True,'updates':b['iter']-a['iter'],'finite_parameters':True,'ankle_std_initial':.08,'other_std_initial':.04,'correct_reward_config':True,'deadline_exit0':True,'final_evaluation_exit0':True,'archive_sha256_verified':True,'parent_sha256':sha(P),'frozen_runtime_manifest_sha256':sha(T/'runtime_manifest.json')}
(D/'verification/ppo_deadline_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
