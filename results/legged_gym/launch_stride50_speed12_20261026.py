from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time
from isaacgym import gymapi
import torch
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_stride50_speed12_20261026';V=D/'verification';OLD=ROOT/'runs/stand5_20261026_v30_knee_stand10';RUN=ROOT/'runs/stand5_20261026_v31_stride50_speed12';SMOKE=ROOT/'runs/stride50_speed12_ppo_smoke_20261026';PARENT=OLD/'checkpoints/model_006939.pt'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
 q=p.with_suffix('.launch.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
assert not RUN.exists()
for _ in range(100):
 if (SMOKE/'status.json').exists() and read(SMOKE/'status.json')['status']=='iterations_complete':break
 time.sleep(.2)
assert read(SMOKE/'status.json')['status']=='iterations_complete'
checks=read(V/'stride50_checks.json');assert checks['passed']
for n,h in checks['source_sha256'].items():assert sha(D/n)==h,n
p=torch.load(str(PARENT),map_location='cpu');a=torch.load(str(SMOKE/'checkpoints/model_006939.pt'),map_location='cpu');b=torch.load(str(SMOKE/'checkpoints/model_006940.pt'),map_location='cpu')
assert all(torch.equal(v,a['model_state_dict'][k]) for k,v in p['model_state_dict'].items() if k!='std')
assert not a['optimizer_state_dict']['state'] and b['optimizer_state_dict']['state']
for prefix in ['actor.','critic.']:assert any(not torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith(prefix))
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values())
cfg=read(SMOKE/'config.json');assert cfg['env']['commands']['ranges']['lin_vel_x']==[.09,.12] and cfg['env']['rewards']['scales']['ankle_stance_motion']==4
assert cfg['env']['posture']['knee_stand_tolerance_deg']==10 and cfg['env']['rewards']['scales']['knee_stand_exp']==-2.5
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256'] and sha(PARENT)==backup['checkpoint']['sha256']
assert sha(D/'supervise.py')==sha(OLD/'runtime/supervise.py')
assert (D/'train.py').read_text().replace('[.09,.12]','[.03,.09]').replace('reference_plus_residual_0.09_to_0.12','reference_plus_residual_0.03_to_0.12') == (OLD/'runtime/train.py').read_text().replace('reference_plus_residual_0.03_to_0.09','reference_plus_residual_0.03_to_0.12')
for f in (D/'assets').rglob('*'):
 if f.is_file():assert sha(f)==sha(OLD/'runtime/assets'/f.relative_to(D/'assets')),str(f)
for n,h in read(SMOKE/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
baseline=read(OLD/'latest_evaluation.json');assert baseline['iteration']==6939 and baseline['checkpoint_sha256']==sha(PARENT)
validation={'passed':True,'real_ppo_updates':1,'exact_network_transfer_except_std':True,'actor_critic_updated':True,'fresh_optimizer':True,'finite_parameters':True,'stride50_boundary_checks':True,'physical_assets_unchanged':True,'parent_prior_speed_evaluation_not_current_speed_baseline':True,'baseline_new_reward_not_trained':True,'deadline_proof_reused_train_schedule_unchanged_speed_literal_only':str(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'),'parent_sha256':sha(PARENT)}
atomic(V/'launch_validation.json',validation)
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for f in D.glob('*.py'):shutil.copy2(f,rt/f.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(OLD/'latest_evaluation.json',RUN/'preflight/parent_prior_speed_evaluation.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
for n in ['completion.json','backup_receipt.json']:shutil.copy2(ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'/n,RUN/'preflight'/('deadline_'+n))
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(minutes=10)).isoformat()
intent=dict(read(OLD/'stage_intent.json'));intent.update(created_unix=time.time(),deadline=deadline,requested_duration_s=600,parent=str(PARENT),parent_sha256=sha(PARENT),headline='当前主线：9–12cm/s · 踝活动加强 · 50mm步长奖励封顶',description='本轮10分钟，速度指令9–12cm/s。支撑期踝活动奖励权重2→4；有效抬脚到落脚的足板中心向前位移奖励，50mm封顶。后膝标准站姿±10°及髋≤−19°免罚、脚板角<90°目标保留。前足摩擦0.02、后足约1，膝触地终止。每5分钟更新回放视频。',knee_stand_tolerance_deg=10,knee_stand_reward_weight=-2.5,knee_stand_scale_deg=15,knee_standard_deg=[91,91],ankle_stance_motion_weight=4,stride_reward_cap_m=.05,command_range_m_s=[.09,.12],reference_table_max_speed_m_s=.09,video_interval_s=300,user_authorization='修改奖励后直接训练；每轮约10分钟',deadline_basis='Ten minutes from formal launch.',automatic_resume=False)
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
