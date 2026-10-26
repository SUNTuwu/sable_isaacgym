from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time
from isaacgym import gymapi
import torch
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_stride80_all_20261027';V=D/'verification';OLD=ROOT/'runs/stand5_20261027_v43_event_angles';RUN=ROOT/'runs/stand5_20261027_v44_stride80_all';SMOKE=ROOT/'runs/stride80_all_ppo_smoke_20261027';PARENT=Path(json.loads((SMOKE/'parent_selected.json').read_text())['path'])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
 q=p.with_suffix('.launch.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
assert not RUN.exists()
for _ in range(100):
 if (SMOKE/'status.json').exists() and read(SMOKE/'status.json')['status']=='iterations_complete':break
 time.sleep(.2)
assert read(SMOKE/'status.json')['status']=='iterations_complete'
checks=read(V/'new_rewards_checks.json');assert checks['passed']
for n,h in checks['source_sha256'].items():assert sha(D/n)==h,n
p=torch.load(str(PARENT),map_location='cpu');a=torch.load(str(SMOKE/'checkpoints'/('model_%06d.pt'%p['iter'])),map_location='cpu');b=torch.load(str(SMOKE/'checkpoints'/('model_%06d.pt'%(p['iter']+1))),map_location='cpu')
assert all(torch.equal(v,a['model_state_dict'][k]) for k,v in p['model_state_dict'].items() if k!='std')
assert not a['optimizer_state_dict']['state'] and b['optimizer_state_dict']['state']
for prefix in ['actor.','critic.']:assert any(not torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith(prefix))
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values())
cfg=read(SMOKE/'config.json');sc=cfg['env']['rewards']['scales']
assert cfg['env']['commands']['ranges']['lin_vel_x']==[.09,.12] and cfg['env']['stride']['hind_zero_m']==.08
old_scales=read(OLD/'config.json')['env']['rewards']['scales'];expected={k:v for k,v in old_scales.items() if k!='hind_stride_shortfall'};expected.update(stride_shortfall=-2.,stride_variance=-2.)
assert sc==expected and len(sc)==27 and 'hip_stand_exp' not in sc and 'knee_stand_exp' not in sc
assert cfg['env']['stride']['front_zero_m']==cfg['env']['stride']['hind_zero_m']==.08 and cfg['env']['stride']['reward_cap_m']==.1
checks2=read(V/'stride_all_checks.json');assert checks2['passed']
for n,h in checks2['source_sha256'].items():assert sha(D/n)==h,n
assert cfg['env']['waist_vertical_acceleration']['normalization_m_s2']==5.
assert read(ROOT/'results/legged_gym/event_angles_watch_status.json').get('finished') is True
assert read(ROOT/'results/legged_gym/event_angles_final_verification_20261027.json')['passed']
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256'] and sha(PARENT)==read(OLD/'latest_evaluation.json')['checkpoint_sha256']
for name in ['train.py','supervise.py','guided_reference.py','gait_support.py']:
 assert (D/name).read_bytes()==(OLD/'runtime'/name).read_bytes()
for f in (D/'assets').rglob('*'):
 if f.is_file():assert sha(f)==sha(OLD/'runtime/assets'/f.relative_to(D/'assets')),str(f)
for n,h in read(SMOKE/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
baseline=read(OLD/'latest_evaluation.json');probe=read(SMOKE/'evaluations/stride_schema_8s.json')
assert baseline['iteration']==p['iter'] and baseline['horizon_s']==16. and baseline['num_envs']==24
assert probe['iteration']==p['iter'] and probe['horizon_s']==8. and probe['num_envs']==6
assert probe['checkpoint_sha256']==sha(SMOKE/'checkpoints'/('model_%06d.pt'%p['iter']))
assert probe['ankle_event_diagnostics']['mode']=='step_events_only' and not probe['knee_posture_diagnostics']['penalty_enabled']
assert probe['active_reward_scales']==sc and probe['waist_vertical_acceleration_diagnostics']['weight']==-1
events=probe['ankle_event_diagnostics']['events'];assert len(events)>10
assert probe['stride_events']['unified_all_feet'] and probe['stride_events']['front_zero_m']==probe['stride_events']['hind_zero_m']==.08
assert probe['stride_variance_diagnostics']['weight']==-2 and probe['stride_variance_diagnostics']['event_count']>10
for v in probe['stride_variance_diagnostics']['events']:
 values=v['completed_stride_m'];mean=sum(values)/4;expected_var=sum((x-mean)**2 for x in values)/4
 assert abs(v['variance_m2']-expected_var)<1e-7
assert cfg['env']['gait']['stance_fraction']==.5 and cfg['env']['gait']['period_s']==1.
assert cfg['env']['material_friction']['front_scale']==1. and cfg['env']['commands']['heading_command']
assert cfg['env']['ankle_phase']['landing_angle_deg']==30 and cfg['env']['ankle_phase']['takeoff_angle_deg']==110
stance_checks=read(V/'stance50_checks.json');assert stance_checks['passed']
for n,h in stance_checks['source_sha256'].items():assert sha(D/n)==h,n
parent_survival_gate_passed=baseline['summary']['survival_16s']>=.9
validation={'passed':True,'real_ppo_updates':1,'exact_network_transfer_except_std':True,'actor_critic_updated':True,'fresh_optimizer':True,'finite_parameters':True,'physical_assets_controller_unchanged':True,'parent_final_24env_baseline_reused_without_reward_claim':True,'schema_probe_seconds':8,'schema_probe_environments':6,'angle_event_count':len(events),'all_four_stride80_cap100_verified':True,'stride_variance_verified_from_report_samples':True,'knee_posture_removed':True,'waist_height_acceleration_weight':-1,'waist_normalization_m_s2':5,'unit_checks_passed':True,'initial_performance_gate_passed':parent_survival_gate_passed,'deadline_proof_reused_stop_save_path_unchanged':str(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'),'parent_sha256':sha(PARENT)}
atomic(V/'launch_validation.json',validation)
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for f in D.glob('*.py'):shutil.copy2(f,rt/f.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(OLD/'latest_evaluation.json',RUN/'preflight/parent_prior_reward_evaluation.json');shutil.copy2(SMOKE/'evaluations/stride_schema_8s.json',RUN/'preflight/stride_schema_8s.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
for n in ['completion.json','backup_receipt.json']:shutil.copy2(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'/n,RUN/'preflight'/('deadline_'+n))
prefs=read(ROOT/'results/legged_gym/training_preferences.json');assert prefs['default_duration_s']==1200 and prefs['progress_interval_s']==prefs['evaluation_interval_s']==prefs['video_interval_s']==180
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
intent=dict(read(OLD/'stage_intent.json'))
intent.update(created_unix=time.time(),deadline=deadline,requested_duration_s=prefs['default_duration_s'],parent=str(PARENT),parent_sha256=sha(PARENT),headline='当前主线：四足80mm步长 · 步长方差惩罚',description='从v43最终'+str(p['iter'])+'轮继续20分钟。前足与后足统一：80mm为步长正负分界，低于80mm负指数，高于80mm正指数，100mm奖励封顶。新增四足真实步长方差惩罚−2×Var(四足最近真实落脚步长)/(80mm)²；四足数据齐全且均在最近两个周期内，只在真实落脚事件计罚。实际步长不截到100mm。保留脚板30°/110°防抖事件误差、腰锚点竖直加速度惩罚、膝/髋优选区关闭、1Hz/50%、前后摩擦1及偏航纠偏。每3分钟同步。',front_stride_zero_m=.08,hind_stride_zero_m=.08,stride_reward_cap_m=.10,stride_shortfall_weight=-2.,stride_variance_weight=-2.,stride_variance_normalization_m=.08,stride_variance_max_age_s=2.,initial_performance_gate_passed=parent_survival_gate_passed,initial_survival_fraction=baseline['summary']['survival_16s'],stage_mode='validated',baseline_scope='Parent final24env16s replay reused for unchanged physics/controller;8s6env probe checks new all-foot stride and variance outputs, not a16s acceptance result.',user_authorization='用户要求前足和后足统一80mm迈步奖惩、100mm封顶，并取前后脚步长方差惩罚大方差；改完直接训练。',automatic_resume=False)
atomic(RUN/'stage_intent.json',intent)
cmd=[sys.executable,'-u',str(rt/'supervise.py'),'--run',str(RUN),'--deadline',deadline,'--resume',str(PARENT),'--phase','guided','--reference-strength','.75','--mirror-policy','--front-friction-scale','1','--num-envs','1024']
with (RUN/'supervisor.log').open('a') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(300):
 if (RUN/'status.json').exists():break
 assert proc.poll() is None;time.sleep(.2)
assert (RUN/'status.json').exists();atomic(ROOT/'runs/current.json',{'run_dir':str(RUN)})
atomic(ROOT/'results/legged_gym/active_training_plan.json',dict(intent,run=RUN.name,run_dir=str(RUN),status='training',supervisor_pid=proc.pid,progress_interval_s=180,web_port=11111))
sys.path.insert(0,str(rt));from progress_report import publish
publish(RUN);print(json.dumps({'started':str(RUN),'deadline':deadline,'supervisor_pid':proc.pid}),flush=True)
