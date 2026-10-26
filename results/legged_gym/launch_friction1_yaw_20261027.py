from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time
from isaacgym import gymapi
import torch
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_friction1_yaw_20261027';V=D/'verification';OLD=ROOT/'runs/stand5_20261027_v41_stance50';RUN=ROOT/'runs/stand5_20261027_v42_friction1_yaw';SMOKE=ROOT/'runs/friction1_yaw_ppo_smoke_20261027';PARENT=Path(json.loads((SMOKE/'parent_selected.json').read_text())['path'])
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
assert read(ROOT/'results/legged_gym/stance50_watch_status.json').get('finished') is True
assert read(ROOT/'results/legged_gym/stance50_final_verification_20261027.json')['passed']
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256'] and sha(PARENT)==read(OLD/'latest_evaluation.json')['checkpoint_sha256']
assert (D/'supervise.py').read_text()==(OLD/'runtime/supervise.py').read_text().replace("p.add_argument('--front-friction-scale',type=float,default=.02)","p.add_argument('--front-friction-scale',type=float,default=1.)")
assert (D/'train.py').read_text()==(OLD/'runtime/train.py').read_text().replace("p.add_argument('--front-friction-scale',type=float,default=.02)","p.add_argument('--front-friction-scale',type=float,default=1.)")

for f in (D/'assets').rglob('*'):
 if f.is_file():assert sha(f)==sha(OLD/'runtime/assets'/f.relative_to(D/'assets')),str(f)
for n,h in read(SMOKE/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
baseline=read(SMOKE/'evaluations/initial_friction1_24env.json');assert baseline['iteration']==p['iter'] and baseline['horizon_s']==16. and baseline['num_envs']==24
assert baseline['checkpoint_sha256']==sha(SMOKE/'checkpoints'/('model_%06d.pt'%p['iter']))
assert baseline['cadence_diagnostics']['configured_period_s']==1. and baseline['cadence_diagnostics']['stance_fraction']==.5 and baseline['cadence_diagnostics']['swing_duration_s']==.5
assert baseline['ankle_tracking_shortfall']['mode']=='negative_only' and baseline['stride_events']['hind_zero_m']==.08 and len(baseline['active_reward_scales'])==30
assert cfg['env']['gait']['stance_fraction']==.5
old_scales=read(OLD/'config.json')['env']['rewards']['scales'];assert cfg['env']['rewards']['scales']==dict(old_scales,yaw_rate_error=-2.)
assert cfg['env']['material_friction']['front_scale']==1. and baseline['material_friction']['nominal_front_floor_mu']==1.
assert baseline['yaw_rate_penalty']['weight']==-2. and baseline['yaw_rate_penalty']['after_positive_clipping'] and cfg['env']['commands']['heading_command']
stance_checks=read(V/'stance50_checks.json');assert stance_checks['passed']
for n,h in stance_checks['source_sha256'].items():assert sha(D/n)==h,n
for log_name in ['train.log','baseline.log']:
 lines=(SMOKE/log_name).read_text().splitlines();imported=json.loads(next(x.split(' ',1)[1] for x in lines if x.startswith('IMPORTED_ASSET ')))
 materials=imported['foot_shape_friction'];values=[v for row in materials.values() for v in row]
 assert len(values)==4 and max(values)-min(values)<1e-6
 assert all(abs(v-2.)<1e-6 for v in values) if log_name=='baseline.log' else all(1.8<=v<=2.2 for v in values)
parent_survival_gate_passed=baseline['summary']['survival_16s']>=.9
assert cfg['env']['gait']['period_s']==1. and cfg['env']['ankle_phase']['landing_angle_deg']==30 and cfg['env']['ankle_phase']['takeoff_angle_deg']==110
assert cfg['env']['rewards']['scales']['proximal_acceleration_peak']==-4
assert not any(k in cfg['env']['rewards']['scales'] for k in ['foot_spacing','ankle_stance_motion','ankle_stance_reversal','hind_foot_angle','stand_still','dof_vel','feet_stumble'])
validation={'passed':True,'real_ppo_updates':1,'exact_network_transfer_except_std':True,'actor_critic_updated':True,'fresh_optimizer':True,'finite_parameters':True,'negative_only_ankle_shortfall_checks':True,'signed_hind_stride_threshold_and_negative_clipping_checks':True,'physical_assets_unchanged':True,'changed_friction_new_24_environment_16second_baseline':True,'front_friction_restored_to1':True,'yaw_error_negative_squared_cost':True,'baseline_survival_fraction':baseline['summary']['survival_16s'],'initial_performance_gate_passed':parent_survival_gate_passed,'width60_and_acceleration_rewards_unchanged':True,'evaluator_and_unchanged_stance50_signals_checked':True,'positive_stride_height_gate_independent':True,'baseline_new_reward_not_trained':True,'deadline_proof_reused_stop_save_path_unchanged_report_and_eval_cadence_only':str(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'),'parent_sha256':sha(PARENT)}
atomic(V/'launch_validation.json',validation)
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for f in D.glob('*.py'):shutil.copy2(f,rt/f.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(OLD/'latest_evaluation.json',RUN/'preflight/parent_prior_reward_evaluation.json');shutil.copy2(SMOKE/'evaluations/initial_friction1_24env.json',RUN/'preflight/initial_friction1_24env.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
for n in ['completion.json','backup_receipt.json']:shutil.copy2(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'/n,RUN/'preflight'/('deadline_'+n))
prefs=read(ROOT/'results/legged_gym/training_preferences.json');assert prefs['default_duration_s']==1200 and prefs['progress_interval_s']==prefs['evaluation_interval_s']==prefs['video_interval_s']==180
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
intent=dict(read(OLD/'stage_intent.json'));intent.update(created_unix=time.time(),deadline=deadline,requested_duration_s=prefs['default_duration_s'],parent=str(PARENT),parent_sha256=sha(PARENT),headline='当前主线：前足摩擦1 · 偏航纠偏 · 1Hz/50%',description='从v41最终'+str(p['iter'])+'轮继续20分钟。保留已改善的1Hz、支撑/摆动各50%。前足名义对地摩擦从0.02恢复1，与后足一致；训练沿用摩擦随机化约0.9–1.1，独立回放为1。新增偏航角速度跟踪误差平方惩罚−2，按实际角速度减heading纠偏指令计算，负分在正奖励截断后加入；保留原角速度跟踪奖励和其他奖励。网页每3分钟同步，重点比较航向偏差与偏航RMSE。',front_friction_scale=1.,nominal_front_floor_friction=1.,nominal_rear_floor_friction=1.,yaw_rate_error_weight=-2.,unchanged_physics=False,unchanged_physical_assets=True,gait_period_s=1.,gait_stance_fraction=.5,stance_duration_s=.5,swing_duration_s=.5,initial_performance_gate_passed=parent_survival_gate_passed,initial_survival_fraction=baseline['summary']['survival_16s'],stage_mode='validated' if parent_survival_gate_passed else 'experimental_user_requested_friction1',baseline_scope='Actual parent weights with restored front friction1; not a learned result. All raw metrics retained.',user_authorization='用户要求继续训练、前足摩擦恢复1、关注走歪和角速度惩罚，并确认50%支撑改善步态；保留1Hz/50%，直接开新20分钟轮次。',deadline_basis='Twenty minutes from formal launch.',automatic_resume=False)
if not parent_survival_gate_passed:intent['description']+=' 新摩擦迁移基线'+str(round(baseline['summary']['survival_16s']*24))+'/24站稳16秒，当前为适应新摩擦的试验训练；保留原始失败记录。'
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
