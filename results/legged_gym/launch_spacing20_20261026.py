from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time
from isaacgym import gymapi
import torch
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_spacing20_20261026';V=D/'verification';OLD=ROOT/'runs/stand5_20261026_v31_stride50_speed12';RUN=ROOT/'runs/stand5_20261026_v32_spacing20_double';SMOKE=ROOT/'runs/spacing20_double_ppo_smoke_20261026';PARENT=OLD/'checkpoints/model_006989.pt'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
 q=p.with_suffix('.launch.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
assert not RUN.exists()
for _ in range(100):
 if (SMOKE/'status.json').exists() and read(SMOKE/'status.json')['status']=='iterations_complete':break
 time.sleep(.2)
assert read(SMOKE/'status.json')['status']=='iterations_complete'
checks=read(V/'spacing_checks.json');assert checks['passed']
for n,h in checks['source_sha256'].items():assert sha(D/n)==h,n
p=torch.load(str(PARENT),map_location='cpu');a=torch.load(str(SMOKE/'checkpoints/model_006989.pt'),map_location='cpu');b=torch.load(str(SMOKE/'checkpoints/model_006990.pt'),map_location='cpu')
assert all(torch.equal(v,a['model_state_dict'][k]) for k,v in p['model_state_dict'].items() if k!='std')
assert not a['optimizer_state_dict']['state'] and b['optimizer_state_dict']['state']
for prefix in ['actor.','critic.']:assert any(not torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith(prefix))
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values())
cfg=read(SMOKE/'config.json');assert cfg['env']['commands']['ranges']['lin_vel_x']==[.09,.12] and cfg['env']['rewards']['scales']['ankle_stance_motion']==8 and cfg['env']['rewards']['scales']['hip_stand_exp']==-5 and cfg['env']['rewards']['scales']['foot_spacing']==-4
assert cfg['env']['posture']['knee_stand_tolerance_deg']==10 and cfg['env']['rewards']['scales']['knee_stand_exp']==-2.5
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256'] and sha(PARENT)==read(OLD/'evaluations/eval_006989.json')['checkpoint_sha256']
assert sha(D/'supervise.py')==sha(OLD/'runtime/supervise.py')
assert sha(D/'train.py')==sha(OLD/'runtime/train.py')
for f in (D/'assets').rglob('*'):
 if f.is_file():assert sha(f)==sha(OLD/'runtime/assets'/f.relative_to(D/'assets')),str(f)
for n,h in read(SMOKE/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
baseline=read(OLD/'evaluations/eval_006989.json');assert baseline['iteration']==6989 and baseline['checkpoint_sha256']==sha(PARENT)
validation={'passed':True,'real_ppo_updates':1,'exact_network_transfer_except_std':True,'actor_critic_updated':True,'fresh_optimizer':True,'finite_parameters':True,'spacing20_boundary_checks':True,'physical_assets_unchanged':True,'same_speed_parent_baseline_reused':True,'baseline_new_reward_not_trained':True,'deadline_proof_reused_train_schedule_unchanged_speed_literal_only':str(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'),'parent_sha256':sha(PARENT)}
atomic(V/'launch_validation.json',validation)
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for f in D.glob('*.py'):shutil.copy2(f,rt/f.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(OLD/'evaluations/eval_006989.json',RUN/'baseline_evaluation.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
for n in ['completion.json','backup_receipt.json']:shutil.copy2(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'/n,RUN/'preflight'/('deadline_'+n))
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(minutes=10)).isoformat()
intent=dict(read(OLD/'stage_intent.json'));intent.update(created_unix=time.time(),deadline=deadline,requested_duration_s=600,parent=str(PARENT),parent_sha256=sha(PARENT),headline='当前主线：脚距≥20mm · 踝奖励8 · 髋惩罚−5',description='本轮10分钟，9–12cm/s。从较稳的v31-6989继续（22/24存活，最终7060为18/24且12cm/s组6/8摔倒）。同一对左右前/后足中心地面间距<20mm扣分，权重−4。踝支撑活动奖励4→8，髋>−19°指数惩罚−2.5→−5；后膝标准±10°、50mm步长奖励封顶和膝触地终止保留。每5分钟更新录像。',knee_stand_tolerance_deg=10,knee_stand_reward_weight=-2.5,knee_stand_scale_deg=15,knee_standard_deg=[91,91],ankle_stance_motion_weight=8,hip_penalty_weight=-5,foot_spacing_min_m=.02,foot_spacing_weight=-4,stride_reward_cap_m=.05,command_range_m_s=[.09,.12],reference_table_max_speed_m_s=.09,video_interval_s=300,user_authorization='修改奖励后直接训练；每轮约10分钟',deadline_basis='Ten minutes from formal launch.',automatic_resume=False)
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
