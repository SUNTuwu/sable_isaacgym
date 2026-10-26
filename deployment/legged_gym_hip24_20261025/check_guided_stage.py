"""Verify a new guided stage against its actual one-update test and replay."""
import argparse,time,json
from pathlib import Path
from isaacgym import gymapi
import torch
from run_io import read_json,atomic_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
p.add_argument('--strength',type=float,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--reference-only',type=Path,required=True);p.add_argument('--mirror-policy',action='store_true');a=p.parse_args()
code=Path(__file__).resolve().parent;old=torch.load(str(a.parent),map_location='cpu');it=int(old['iter'])
initial=torch.load(str(a.run/'checkpoints'/('model_%06d.pt'%it)),map_location='cpu')
final=torch.load(str(a.run/'checkpoints'/('model_%06d.pt'%(it+1))),map_location='cpu');checks={}
for k,v in old['model_state_dict'].items():
    new=initial['model_state_dict'][k]
    if k=='std':continue
    if k in ['actor.0.weight','critic.0.weight']:
        checks[k+'_preserved']=torch.equal(v,new[:,:v.shape[1]]) and new.shape[1]==52 and bool((new[:,v.shape[1]:]==0).all())
    else:checks[k+'_preserved']=torch.equal(v,new)
checks.update(std_reset_004=torch.allclose(initial['model_state_dict']['std'],torch.full_like(initial['model_state_dict']['std'],.04)),
    optimizer_initially_empty=not initial['optimizer_state_dict']['state'],optimizer_updated=bool(final['optimizer_state_dict']['state']),
    one_update_completed=final['iter']==it+1,trained_std_above_floor=bool((final['model_state_dict']['std']>=.04).all()))
checks['actor_updated']=any(not torch.equal(v,final['model_state_dict'][k]) for k,v in initial['model_state_dict'].items() if k.startswith('actor.'))
checks['critic_updated']=any(not torch.equal(v,final['model_state_dict'][k]) for k,v in initial['model_state_dict'].items() if k.startswith('critic.'))
cfg=read_json(a.run/'config.json');baseline=read_json(a.baseline);zero=read_json(a.reference_only)
checks['guided_config']=cfg['phase']=='guided' and cfg['env']['reference']['enabled'] and cfg['env']['reference']['strength']==a.strength
checks['contact_phase_decoupled']=cfg['env']['gait']['contact_phase_independent_of_strength']
checks['clearance_scale_matches_course']=cfg['env']['gait']['clearance_relative_to_target']
checks['mirror_policy_config']=bool(cfg.get('mirror_policy',False))==a.mirror_policy
checks['replay_scope']=all(r['kinematic_reference_enabled'] and r['reference_strength']==a.strength and r['horizon_s']==16 and r['num_envs']==24 and r['command_set_m_s']==[.03,.045,.06] for r in [baseline,zero])
checks['reference_control']=zero['zero_policy_residual'] and not baseline['zero_policy_residual']
checks['replay_policy_class']=all(bool(r.get('mirror_policy',False))==a.mirror_policy for r in [baseline,zero])
checks['parent_survival_gate']=baseline['summary']['survival_16s']>=.9
checks['finite_replays']=all(torch.isfinite(torch.tensor(v)).all() for r in [baseline,zero] for v in r['summary'].values())
manifest=read_json(a.run/'runtime_manifest.json')
sources=['train.py','sable_env.py','sable_config.py','gait_support.py','guided_reference.py','preflight.py','assets/gait_reference.json','assets/sable.xml','assets/physical_manifest.json','evaluate.py','priority_metrics.py']
projection={}
if a.mirror_policy:
    from equivariant_policy import SymmetricActorCritic
    sources+=['policy_symmetry.py','equivariant_policy.py']
    checks['runner_policy_class']=cfg['train']['runner']['policy_class_name']=='SymmetricActorCritic'
    actor=SymmetricActorCritic(52,52,12,**cfg['train']['policy']).to('cuda:0');actor.load_state_dict(initial['model_state_dict'])
    torch.manual_seed(947);obs=torch.randn(128,52,device='cuda:0');m=actor.mirror(obs)
    with torch.no_grad():
        mean=actor.act_inference(obs);reflected=actor.act_inference(m.observations(obs))
        projection['actor_equivariance_max_abs']=float((reflected-m.actions(mean)).abs().max())
        projection['critic_invariance_max_abs']=float((actor.evaluate(obs)-actor.evaluate(m.observations(obs))).abs().max())
        actor.update_distribution(obs)
        projection['std_symmetry_max_abs']=float((actor.action_std-actor.action_std[:,m.ai]).abs().max())
    checks['projected_actor_and_critic']=all(v<1e-6 for v in projection.values())
    proof=read_json(code/'verification/symmetry_geometry.json')
    checks['static_mirror_geometry']=proof['passed'] and proof['mirror_source_sha256']==sha256(code/'policy_symmetry.py') and proof['asset_sha256']==sha256(code/'assets/sable.xml') and proof['manifest_sha256']==sha256(code/'assets/physical_manifest.json')
checks['tested_current_sources']=all(manifest['files'][n]==sha256(code/n) for n in sources)
checks['actual_evaluator']=all(r['evaluator_sha256']==sha256(code/'evaluate.py') for r in [baseline,zero])
checks={k:bool(v) for k,v in checks.items()}
report={'passed':all(checks.values()),'checks':checks,'reference_strength':a.strength,'mirror_policy':a.mirror_policy,'projection_metrics':projection,'parent_sha256':sha256(a.parent),
    'source_sha256':{n:sha256(code/n) for n in sources},'reference_only_summary':zero['summary'],'parent_with_reference_summary':baseline['summary'],
    'created_unix':time.time(),'scope':'Exact weight transfer, one actual PPO update and 16s baselines; not gait acceptance'}
atomic_json(code/'verification/guided_transfer.json',report);print('GUIDED_STAGE',json.dumps(report))
if not report['passed']:raise SystemExit(2)
