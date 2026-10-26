from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time
from isaacgym import gymapi
import torch
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_angles_width_accel_20261027';V=D/'verification';OLD=ROOT/'runs/stand5_20261027_v37_cadence1s';RUN=ROOT/'runs/stand5_20261027_v38_angles_width_accel';SMOKE=ROOT/'runs/angles_width_accel_ppo_smoke_v2_20261027';PARENT=OLD/'checkpoints/model_008007.pt'
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
p=torch.load(str(PARENT),map_location='cpu');a=torch.load(str(SMOKE/'checkpoints/model_008007.pt'),map_location='cpu');b=torch.load(str(SMOKE/'checkpoints/model_008008.pt'),map_location='cpu')
assert all(torch.equal(v,a['model_state_dict'][k]) for k,v in p['model_state_dict'].items() if k!='std')
assert not a['optimizer_state_dict']['state'] and b['optimizer_state_dict']['state']
for prefix in ['actor.','critic.']:assert any(not torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith(prefix))
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values())
cfg=read(SMOKE/'config.json');assert cfg['env']['commands']['ranges']['lin_vel_x']==[.09,.12] and cfg['env']['rewards']['scales']['ankle_phase_motion']==8 and cfg['env']['rewards']['scales']['hip_stand_exp']==-5 and cfg['env']['rewards']['scales']['hind_landing_width']==-4
assert cfg['env']['posture']['knee_stand_reference_offset_deg']==-10 and cfg['env']['stride']['reward_cap_m']==.1 and cfg['env']['stride']['reward_exponent']==4
assert cfg['env']['stride']['hind_zero_m']==.05 and cfg['env']['ankle_phase']['landing_min_deg']==30 and cfg['env']['rewards']['scales']['ankle_landing_low']==-2 and cfg['env']['rewards']['scales']['hind_stride_shortfall']==-2
assert cfg['env']['posture']['knee_stand_tolerance_deg']==10 and cfg['env']['rewards']['scales']['knee_stand_exp']==-2.5
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256'] and sha(PARENT)==read(OLD/'latest_evaluation.json')['checkpoint_sha256']
assert sha(D/'supervise.py')==sha(OLD/'runtime/supervise.py')
assert (D/'train.py').read_text()==(OLD/'runtime/train.py').read_text().replace("    runner.status['ankle_stance_target_deg']=cfg.ankle_stance.excursion_target_deg", "    runner.status.update(ankle_landing_target_deg=cfg.ankle_phase.landing_angle_deg,ankle_takeoff_target_deg=cfg.ankle_phase.takeoff_angle_deg,hind_landing_width_minimum_m=cfg.hind_landing_width.minimum_m,proximal_acceleration_threshold_rad_s2=cfg.proximal_acceleration.threshold_rad_s2)")
for f in (D/'assets').rglob('*'):
 if f.is_file():assert sha(f)==sha(OLD/'runtime/assets'/f.relative_to(D/'assets')),str(f)
for n,h in read(SMOKE/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
baseline=read(OLD/'latest_evaluation.json');assert baseline['iteration']==8007 and baseline['checkpoint_sha256']==sha(PARENT) and baseline['summary']['survival_16s']>=.9
probe=read(SMOKE/'evaluations/schema_probe.json');assert probe['horizon_s']==2 and probe['proximal_acceleration_diagnostics']['threshold_rad_s2']==150 and probe['hind_landing_width_diagnostics']['minimum_m']==.06
assert cfg['env']['gait']['period_s']==1. and cfg['env']['ankle_phase']['landing_angle_deg']==30 and cfg['env']['ankle_phase']['takeoff_angle_deg']==110
assert cfg['env']['rewards']['scales']['proximal_acceleration_peak']==-4
assert not any(k in cfg['env']['rewards']['scales'] for k in ['foot_spacing','ankle_stance_motion','ankle_stance_reversal','hind_foot_angle','stand_still','dof_vel','feet_stumble'])
validation={'passed':True,'real_ppo_updates':1,'exact_network_transfer_except_std':True,'actor_critic_updated':True,'fresh_optimizer':True,'finite_parameters':True,'contact_phase_direction_and_angle_reward_checks':True,'signed_hind_stride_threshold_and_negative_clipping_checks':True,'physical_assets_unchanged':True,'unchanged_control_parent_24_environment_baseline_reused':True,'width60_and_acceleration_threshold_checked':True,'new_evaluator_two_second_schema_check':True,'positive_stride_height_gate_independent':True,'baseline_new_reward_not_trained':True,'deadline_proof_reused_train_schedule_unchanged_metadata_only':str(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'),'parent_sha256':sha(PARENT)}
atomic(V/'launch_validation.json',validation)
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for f in D.glob('*.py'):shutil.copy2(f,rt/f.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(OLD/'latest_evaluation.json',RUN/'preflight/parent_prior_reward_evaluation.json');shutil.copy2(SMOKE/'evaluations/schema_probe.json',RUN/'preflight/two_second_schema_probe.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
for n in ['completion.json','backup_receipt.json']:shutil.copy2(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'/n,RUN/'preflight'/('deadline_'+n))
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(minutes=10)).isoformat()
intent=dict(read(OLD/'stage_intent.json'));intent.update(created_unix=time.time(),deadline=deadline,requested_duration_s=600,parent=str(PARENT),parent_sha256=sha(PARENT),headline='当前主线：30°落地 · 110°抬脚 · 后足60mm · 髋肩削峰',description='本轮10分钟，从v37最终8007轮继续。脚板对地角改30°落地、110°抬脚，落地过小指数惩罚下界同步30°。后脚落点沿身体左右横向间距至少60mm，替换旧前/后脚20mm平面距离惩罚，髋轴15mm方案被用户修正未启用。新增髋肩7个独立电机角加速度150rad/s²以上平方超额惩罚，权重−4；按20ms实际关节速度差分，非硬裁剪动作。清理6项零权重奖励，旧踝范围模块只留测量。保留1秒周期、50mm步长分界、100mm封顶、9–12cm/s指令及膝髋姿态惩罚。每5分钟录像。',knee_stand_tolerance_deg=10,knee_stand_reward_weight=-2.5,knee_stand_scale_deg=15,knee_standard_deg=[91,91],knee_reward_center_deg=[81,81],knee_free_lower_deg=[71,71],knee_free_upper_deg=[91,91],knee_reference_offset_deg=-10,ankle_stance_motion_weight=0,ankle_phase_motion_weight=8,ankle_phase_angle_weight=-4,ankle_phase_event_weight=-2,ankle_phase_reverse_weight=-.5,ankle_landing_target_deg=30,ankle_takeoff_target_deg=110,hip_penalty_weight=-5,foot_spacing_min_m=None,foot_spacing_weight=0,hind_landing_width_minimum_m=.06,hind_landing_width_weight=-4,proximal_acceleration_threshold_rad_s2=150,proximal_acceleration_weight=-4,ankle_landing_min_deg=30,ankle_landing_low_weight=-2,ankle_landing_shortfall_scale_deg=10,ankle_landing_shortfall_max_exponent=3,hind_stride_zero_m=.05,hind_stride_shortfall_weight=-2,stride_reward_cap_m=.10,stride_reward_exponent=4,command_range_m_s=[.09,.12],reference_table_max_speed_m_s=.09,gait_period_s=1.,swing_duration_s=.25,stride_positive_peak20_gate=False,independent_lift_acceptance_min_m=.020,video_interval_s=300,user_authorization='用户要求继续训练、30/110脚板目标、后足横向60mm、清理旧奖励、髋肩加速度削峰并展示图表；默认10分钟',deadline_basis='Ten minutes from formal launch.',automatic_resume=False)
atomic(RUN/'stage_intent.json',intent)
cmd=[sys.executable,'-u',str(rt/'supervise.py'),'--run',str(RUN),'--deadline',deadline,'--resume',str(PARENT),'--phase','guided','--reference-strength','.75','--mirror-policy','--front-friction-scale','.02','--num-envs','1024']
with (RUN/'supervisor.log').open('a') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(300):
 if (RUN/'status.json').exists():break
 assert proc.poll() is None;time.sleep(.2)
assert (RUN/'status.json').exists();atomic(ROOT/'runs/current.json',{'run_dir':str(RUN)})
atomic(ROOT/'results/legged_gym/active_training_plan.json',dict(intent,run=RUN.name,run_dir=str(RUN),status='training',supervisor_pid=proc.pid,progress_interval_s=300,web_port=11111))
sys.path.insert(0,str(rt));from progress_report import publish
publish(RUN);print(json.dumps({'started':str(RUN),'deadline':deadline,'supervisor_pid':proc.pid}),flush=True)
