from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time
from isaacgym import gymapi
import torch
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_stride80_nohip_20261027';V=D/'verification';OLD=ROOT/'runs/stand5_20261027_v38_angles_width_accel';RUN=ROOT/'runs/stand5_20261027_v39_stride80_nohip';SMOKE=ROOT/'runs/stride80_nohip_ppo_smoke_20261027';PARENT=OLD/'checkpoints/model_008152.pt'
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
p=torch.load(str(PARENT),map_location='cpu');a=torch.load(str(SMOKE/'checkpoints/model_008152.pt'),map_location='cpu');b=torch.load(str(SMOKE/'checkpoints/model_008153.pt'),map_location='cpu')
assert all(torch.equal(v,a['model_state_dict'][k]) for k,v in p['model_state_dict'].items() if k!='std')
assert not a['optimizer_state_dict']['state'] and b['optimizer_state_dict']['state']
for prefix in ['actor.','critic.']:assert any(not torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith(prefix))
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values())
cfg=read(SMOKE/'config.json');assert cfg['env']['commands']['ranges']['lin_vel_x']==[.09,.12] and cfg['env']['rewards']['scales']['ankle_phase_motion']==8 and 'hip_stand_exp' not in cfg['env']['rewards']['scales'] and cfg['env']['rewards']['scales']['hind_landing_width']==-4
assert cfg['env']['posture']['knee_stand_reference_offset_deg']==-10 and cfg['env']['stride']['reward_cap_m']==.1 and cfg['env']['stride']['reward_exponent']==4
assert cfg['env']['stride']['hind_zero_m']==.08 and cfg['env']['ankle_phase']['landing_min_deg']==30 and cfg['env']['rewards']['scales']['ankle_landing_low']==-2 and cfg['env']['rewards']['scales']['hind_stride_shortfall']==-2
assert cfg['env']['posture']['knee_stand_tolerance_deg']==10 and cfg['env']['rewards']['scales']['knee_stand_exp']==-2.5
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256'] and sha(PARENT)==read(OLD/'latest_evaluation.json')['checkpoint_sha256']
assert (D/'supervise.py').read_text()==(OLD/'runtime/supervise.py').read_text().replace('next_report=time.time()+300.','next_report=time.time()+180.')
expected_train=(OLD/'runtime/train.py').read_text().replace("    runner.status['hip_preferred_max_deg']=cfg.posture.hip_preferred_max_deg","    runner.status['hip_preferred_max_deg']=None\n    runner.status['hip_posture_penalty_weight']=0.\n    runner.status['progress_interval_s']=180\n    runner.status['hind_stride_zero_m']=cfg.stride.hind_zero_m").replace('self.completed-self.start_iteration==50 or time.time()-self.last_evaluation_unix>=300',"(self.completed-self.start_iteration==50 and not (self.out/'latest_evaluation.json').exists()) or time.time()-self.last_evaluation_unix>=180")
assert (D/'train.py').read_text()==expected_train

for f in (D/'assets').rglob('*'):
 if f.is_file():assert sha(f)==sha(OLD/'runtime/assets'/f.relative_to(D/'assets')),str(f)
for n,h in read(SMOKE/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
baseline=read(OLD/'latest_evaluation.json');assert baseline['iteration']==8152 and baseline['checkpoint_sha256']==sha(PARENT) and baseline['summary']['survival_16s']>=.9
probe=read(SMOKE/'evaluations/schema_probe.json');assert probe['horizon_s']==2 and probe['proximal_acceleration_diagnostics']['threshold_rad_s2']==150 and probe['hind_landing_width_diagnostics']['minimum_m']==.06
assert probe['hip_posture_penalty_weight']==0 and probe['hip_preferred_max_deg'] is None and probe['stride_events']['hind_zero_m']==.08 and len(probe['active_reward_scales'])==29
assert cfg['env']['gait']['period_s']==1. and cfg['env']['ankle_phase']['landing_angle_deg']==30 and cfg['env']['ankle_phase']['takeoff_angle_deg']==110
assert cfg['env']['rewards']['scales']['proximal_acceleration_peak']==-4
assert not any(k in cfg['env']['rewards']['scales'] for k in ['foot_spacing','ankle_stance_motion','ankle_stance_reversal','hind_foot_angle','stand_still','dof_vel','feet_stumble'])
validation={'passed':True,'real_ppo_updates':1,'exact_network_transfer_except_std':True,'actor_critic_updated':True,'fresh_optimizer':True,'finite_parameters':True,'contact_phase_direction_and_angle_reward_checks':True,'signed_hind_stride_threshold_and_negative_clipping_checks':True,'physical_assets_unchanged':True,'unchanged_control_parent_24_environment_baseline_reused':True,'width60_and_acceleration_threshold_checked':True,'new_evaluator_two_second_schema_check':True,'positive_stride_height_gate_independent':True,'baseline_new_reward_not_trained':True,'deadline_proof_reused_stop_save_path_unchanged_report_and_eval_cadence_only':str(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'),'parent_sha256':sha(PARENT)}
atomic(V/'launch_validation.json',validation)
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for f in D.glob('*.py'):shutil.copy2(f,rt/f.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(OLD/'latest_evaluation.json',RUN/'preflight/parent_prior_reward_evaluation.json');shutil.copy2(SMOKE/'evaluations/schema_probe.json',RUN/'preflight/two_second_schema_probe.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
for n in ['completion.json','backup_receipt.json']:shutil.copy2(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'/n,RUN/'preflight'/('deadline_'+n))
prefs=read(ROOT/'results/legged_gym/training_preferences.json');assert prefs['default_duration_s']==1200 and prefs['progress_interval_s']==prefs['evaluation_interval_s']==prefs['video_interval_s']==180
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
intent=dict(read(OLD/'stage_intent.json'));intent.update(created_unix=time.time(),deadline=deadline,requested_duration_s=prefs['default_duration_s'],parent=str(PARENT),parent_sha256=sha(PARENT),headline='当前主线：取消髋姿态惩罚 · 后足80mm分界 · 20分钟',description='本轮20分钟，从v38最终8152轮继续。取消髋姿态指数惩罚；后脚真实抬落步长80mm以下负指数、80mm零分、80–100mm正指数、100mm封顶。保留髋肩角加速度削峰、膝71–91°免罚、脚板30°落地/110°抬脚、后脚横向60mm、1秒周期及9–12cm/s指令。每3分钟同步报告、安排独立评估和录像，视频生成后发布并标明实际检查点。',hip_preferred_max_deg=None,hip_penalty_weight=0,hip_posture_penalty_weight=0,hind_stride_zero_m=.08,stride_reward_cap_m=.10,video_interval_s=180,progress_interval_s=180,evaluation_interval_s=180,user_authorization='用户取消髋姿态惩罚，后脚步长分界80mm；以后默认每轮20分钟，每3分钟同步进度。按既有授权完成检查后直接训练。',deadline_basis='Twenty minutes from formal launch.',automatic_resume=False)
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
