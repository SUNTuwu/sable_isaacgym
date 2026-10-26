from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time
from isaacgym import gymapi
import torch
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_stance50_20261027';V=D/'verification';OLD=ROOT/'runs/stand5_20261027_v40_ankle_lag';RUN=ROOT/'runs/stand5_20261027_v41_stance50';SMOKE=ROOT/'runs/stance50_ppo_smoke_20261027';PARENT=Path(json.loads((SMOKE/'parent_selected.json').read_text())['path'])
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
cfg=read(SMOKE/'config.json');assert cfg['env']['commands']['ranges']['lin_vel_x']==[.09,.12] and cfg['env']['rewards']['scales']['ankle_phase_lag']==-8 and 'ankle_phase_motion' not in cfg['env']['rewards']['scales'] and 'hip_stand_exp' not in cfg['env']['rewards']['scales'] and cfg['env']['rewards']['scales']['hind_landing_width']==-4
assert cfg['env']['posture']['knee_stand_reference_offset_deg']==-10 and cfg['env']['stride']['reward_cap_m']==.1 and cfg['env']['stride']['reward_exponent']==4
assert cfg['env']['stride']['hind_zero_m']==.08 and cfg['env']['ankle_phase']['landing_min_deg']==30 and cfg['env']['rewards']['scales']['ankle_landing_low']==-2 and cfg['env']['rewards']['scales']['hind_stride_shortfall']==-2
assert cfg['env']['posture']['knee_stand_tolerance_deg']==10 and cfg['env']['rewards']['scales']['knee_stand_exp']==-2.5
assert read(ROOT/'results/legged_gym/ankle_lag_watch_status.json').get('finished') is True
assert read(ROOT/'results/legged_gym/ankle_lag_final_verification_20261027.json')['passed']
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256'] and sha(PARENT)==read(OLD/'latest_evaluation.json')['checkpoint_sha256']
assert sha(D/'supervise.py')==sha(OLD/'runtime/supervise.py')
assert (D/'train.py').read_text()==(OLD/'runtime/train.py').read_text().replace('Give each foot 250 ms to swing','Give each foot 500 ms to swing')

for f in (D/'assets').rglob('*'):
 if f.is_file() and f.name!='gait_reference.json':assert sha(f)==sha(OLD/'runtime/assets'/f.relative_to(D/'assets')),str(f)
for n,h in read(SMOKE/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
baseline=read(SMOKE/'evaluations/initial_stance50_24env.json');assert baseline['iteration']==p['iter'] and baseline['horizon_s']==16. and baseline['num_envs']==24
assert baseline['checkpoint_sha256']==sha(SMOKE/'checkpoints'/('model_%06d.pt'%p['iter']))
assert baseline['cadence_diagnostics']['configured_period_s']==1. and baseline['cadence_diagnostics']['stance_fraction']==.5 and baseline['cadence_diagnostics']['swing_duration_s']==.5
assert baseline['ankle_tracking_shortfall']['mode']=='negative_only' and baseline['stride_events']['hind_zero_m']==.08 and len(baseline['active_reward_scales'])==29
assert cfg['env']['gait']['stance_fraction']==.5 and cfg['env']['rewards']['scales']==read(OLD/'config.json')['env']['rewards']['scales']
stance_checks=read(V/'stance50_checks.json');assert stance_checks['passed']
for n,h in stance_checks['source_sha256'].items():assert sha(D/n)==h,n
parent_survival_gate_passed=baseline['summary']['survival_16s']>=.9
assert cfg['env']['gait']['period_s']==1. and cfg['env']['ankle_phase']['landing_angle_deg']==30 and cfg['env']['ankle_phase']['takeoff_angle_deg']==110
assert cfg['env']['rewards']['scales']['proximal_acceleration_peak']==-4
assert not any(k in cfg['env']['rewards']['scales'] for k in ['foot_spacing','ankle_stance_motion','ankle_stance_reversal','hind_foot_angle','stand_still','dof_vel','feet_stumble'])
validation={'passed':True,'real_ppo_updates':1,'exact_network_transfer_except_std':True,'actor_critic_updated':True,'fresh_optimizer':True,'finite_parameters':True,'negative_only_ankle_shortfall_checks':True,'signed_hind_stride_threshold_and_negative_clipping_checks':True,'physical_assets_unchanged':True,'changed_controller_new_24_environment_16second_baseline':True,'baseline_survival_fraction':baseline['summary']['survival_16s'],'initial_performance_gate_passed':parent_survival_gate_passed,'width60_and_acceleration_rewards_unchanged':True,'evaluator_and_stance50_signals_checked':True,'positive_stride_height_gate_independent':True,'baseline_new_reward_not_trained':True,'deadline_proof_reused_stop_save_path_unchanged_report_and_eval_cadence_only':str(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'),'parent_sha256':sha(PARENT)}
atomic(V/'launch_validation.json',validation)
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for f in D.glob('*.py'):shutil.copy2(f,rt/f.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(OLD/'latest_evaluation.json',RUN/'preflight/parent_prior_reward_evaluation.json');shutil.copy2(SMOKE/'evaluations/initial_stance50_24env.json',RUN/'preflight/initial_stance50_24env.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
for n in ['completion.json','backup_receipt.json']:shutil.copy2(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'/n,RUN/'preflight'/('deadline_'+n))
prefs=read(ROOT/'results/legged_gym/training_preferences.json');assert prefs['default_duration_s']==1200 and prefs['progress_interval_s']==prefs['evaluation_interval_s']==prefs['video_interval_s']==180
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
intent=dict(read(OLD/'stage_intent.json'));intent.update(created_unix=time.time(),deadline=deadline,requested_duration_s=prefs['default_duration_s'],parent=str(PARENT),parent_sha256=sha(PARENT),headline='当前主线：1Hz · 支撑50% · 摆动50%',description='从v40最终'+str(p['iter'])+'轮继续20分钟。每脚1Hz，支撑/摆动各0.5秒，参考轨迹与踝角相位同步，四脚相位偏移顺序保留。接触/离地目标现在允许两脚同时摆动。脚板30°落地/110°抬脚、踝转角跟不上惩罚、后足80mm分界/100mm封顶、髋姿态免罚等设置保留。网页每3分钟同步进度和录像。',gait_period_s=1.,gait_stance_fraction=.5,stance_duration_s=.5,swing_duration_s=.5,simultaneous_reference_swing_feet=2,initial_performance_gate_passed=parent_survival_gate_passed,initial_survival_fraction=baseline['summary']['survival_16s'],stage_mode='validated' if parent_survival_gate_passed else 'experimental_user_requested_stance50',baseline_scope='Actual parent weights under the new50% stance controller; not a learned result. Any initial survival shortfall is retained, with the explicitly requested curriculum allowed after structural/numerical checks.',user_authorization='用户要求步频1Hz、支撑50%；按既有直接开训授权启动新轮20分钟，每3分钟同步。',deadline_basis='Twenty minutes from formal launch.',automatic_resume=False)
atomic(RUN/'stage_intent.json',intent)
cmd=[sys.executable,'-u',str(rt/'supervise.py'),'--run',str(RUN),'--deadline',deadline,'--resume',str(PARENT),'--phase','guided','--reference-strength','.75','--mirror-policy','--front-friction-scale','.02','--num-envs','1024']
with (RUN/'supervisor.log').open('a') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(300):
 if (RUN/'status.json').exists():break
 assert proc.poll() is None;time.sleep(.2)
assert (RUN/'status.json').exists();atomic(ROOT/'runs/current.json',{'run_dir':str(RUN)})
atomic(ROOT/'results/legged_gym/active_training_plan.json',dict(intent,run=RUN.name,run_dir=str(RUN),status='training',supervisor_pid=proc.pid,progress_interval_s=180,web_port=11111))
sys.path.insert(0,str(rt));from progress_report import publish
publish(RUN);print(json.dumps({'started':str(RUN),'deadline':deadline,'supervisor_pid':proc.pid}),flush=True)
