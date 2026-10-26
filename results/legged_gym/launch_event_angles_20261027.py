from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time
from isaacgym import gymapi
import torch
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_event_angles_20261027';V=D/'verification';OLD=ROOT/'runs/stand5_20261027_v42_friction1_yaw';RUN=ROOT/'runs/stand5_20261027_v43_event_angles';SMOKE=ROOT/'runs/event_angles_ppo_smoke_20261027';PARENT=Path(json.loads((SMOKE/'parent_selected.json').read_text())['path'])
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
removed=['ankle_landing_low','ankle_phase_angle','ankle_phase_event','ankle_phase_lag','ankle_phase_reverse','knee_stand_exp']
old_scales=read(OLD/'config.json')['env']['rewards']['scales'];expected={k:v for k,v in old_scales.items() if k not in removed};expected.update(ankle_event_angle=-2.,waist_vertical_acceleration=-1.)
assert sc==expected and len(sc)==26 and 'hip_stand_exp' not in sc
assert cfg['env']['waist_vertical_acceleration']['normalization_m_s2']==5.
assert read(ROOT/'results/legged_gym/friction1_yaw_watch_status.json').get('finished') is True
assert read(ROOT/'results/legged_gym/friction1_yaw_final_verification_20261027.json')['passed']
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256'] and sha(PARENT)==read(OLD/'latest_evaluation.json')['checkpoint_sha256']
for name in ['train.py','supervise.py','guided_reference.py','gait_support.py']:
 assert (D/name).read_bytes()==(OLD/'runtime'/name).read_bytes()
for f in (D/'assets').rglob('*'):
 if f.is_file():assert sha(f)==sha(OLD/'runtime/assets'/f.relative_to(D/'assets')),str(f)
for n,h in read(SMOKE/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
baseline=read(OLD/'latest_evaluation.json');probe=read(SMOKE/'evaluations/event_schema_8s.json')
assert baseline['iteration']==p['iter'] and baseline['horizon_s']==16. and baseline['num_envs']==24
assert probe['iteration']==p['iter'] and probe['horizon_s']==8. and probe['num_envs']==6
assert probe['checkpoint_sha256']==sha(SMOKE/'checkpoints'/('model_%06d.pt'%p['iter']))
assert probe['ankle_event_diagnostics']['mode']=='step_events_only' and not probe['knee_posture_diagnostics']['penalty_enabled']
assert probe['active_reward_scales']==sc and probe['waist_vertical_acceleration_diagnostics']['weight']==-1
events=probe['ankle_event_diagnostics']['events'];seen={};assert len(events)>10
for event in sorted(events,key=lambda x:(x['confirmed_s'],x['time_s'])):
 key=(event['environment'],event['foot'],event['pair_id']);kind=event['kind'];assert (key,kind) not in seen
 if kind=='takeoff':assert (key,'landing') in seen and event['time_s']>=seen[(key,'landing')]['time_s'] and event['confirmed_s']>event['time_s']
 seen[(key,kind)]=event
assert cfg['env']['gait']['stance_fraction']==.5 and cfg['env']['gait']['period_s']==1.
assert cfg['env']['material_friction']['front_scale']==1. and cfg['env']['commands']['heading_command']
assert cfg['env']['ankle_phase']['landing_angle_deg']==30 and cfg['env']['ankle_phase']['takeoff_angle_deg']==110
stance_checks=read(V/'stance50_checks.json');assert stance_checks['passed']
for n,h in stance_checks['source_sha256'].items():assert sha(D/n)==h,n
parent_survival_gate_passed=baseline['summary']['survival_16s']>=.9
validation={'passed':True,'real_ppo_updates':1,'exact_network_transfer_except_std':True,'actor_critic_updated':True,'fresh_optimizer':True,'finite_parameters':True,'physical_assets_controller_unchanged':True,'parent_final_24env_baseline_reused_without_reward_claim':True,'schema_probe_seconds':8,'schema_probe_environments':6,'angle_event_count':len(events),'angle_pairs_unique_ordered':True,'knee_posture_removed':True,'waist_height_acceleration_weight':-1,'waist_normalization_m_s2':5,'unit_checks_passed':True,'initial_performance_gate_passed':parent_survival_gate_passed,'deadline_proof_reused_stop_save_path_unchanged':str(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'),'parent_sha256':sha(PARENT)}
atomic(V/'launch_validation.json',validation)
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for f in D.glob('*.py'):shutil.copy2(f,rt/f.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(OLD/'latest_evaluation.json',RUN/'preflight/parent_prior_reward_evaluation.json');shutil.copy2(SMOKE/'evaluations/event_schema_8s.json',RUN/'preflight/event_schema_8s.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
for n in ['completion.json','backup_receipt.json']:shutil.copy2(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'/n,RUN/'preflight'/('deadline_'+n))
prefs=read(ROOT/'results/legged_gym/training_preferences.json');assert prefs['default_duration_s']==1200 and prefs['progress_interval_s']==prefs['evaluation_interval_s']==prefs['video_interval_s']==180
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
intent=dict(read(OLD/'stage_intent.json'))
for key in list(intent):
 if any(x in key for x in ['knee','ankle_phase','ankle_landing_low','ankle_tracking']):intent.pop(key)
intent.update(created_unix=time.time(),deadline=deadline,requested_duration_s=prefs['default_duration_s'],parent=str(PARENT),parent_sha256=sha(PARENT),headline='当前主线：落脚/抬脚角事件 · 腰部稳定',description='从v42最终9012轮继续20分钟。取消膝关节优选区域；脚板连续相位角跟随、方向及低落地角惩罚全部撤销。仅在真正抬步后首次落脚(30°)、此后首次确认抬脚(110°)计算指数角误差，无免罚带；接触振动不重复计数。新增腰关节锚点竖直加速度平方惩罚−(a_z/5m/s²)²，重置后前两采样不计。保留1Hz、支撑/摆动各50%、前后足摩擦1、偏航纠偏、后足80mm步长阈值。网页每3分钟同步。',knee_posture_penalty_weight=0.,knee_preferred_region=None,ankle_angle_reward_mode='step_events_only',ankle_event_angle_weight=-2.,ankle_event_error_scale_deg=30.,ankle_event_error_deadband_deg=0.,ankle_landing_target_deg=30.,ankle_takeoff_target_deg=110.,waist_vertical_acceleration_weight=-1.,waist_vertical_acceleration_normalization_m_s2=5.,unchanged_physics=True,unchanged_physical_assets=True,initial_performance_gate_passed=parent_survival_gate_passed,initial_survival_fraction=baseline['summary']['survival_16s'],stage_mode='validated',baseline_scope='Parent final24env16s replay reused because physical assets/controller unchanged;8s6env probe validates new event and acceleration outputs, not a16s acceptance result.',user_authorization='用户要求删连续脚板角目标跟随、只按真实落脚/抬脚事件指数罚30/110误差、删除膝优选区域，并同时添加腰关节锚点高度加速度惩罚；改完直接开训。',automatic_resume=False)
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
