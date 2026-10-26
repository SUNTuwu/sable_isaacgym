"""Launch a separate 20-minute reward stage after real PPO/deadline verification."""
from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time,tarfile
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_hip19_ankle_stance_20261026';V=D/'verification'
OLD=ROOT/'runs/stand5_20261026_v28_ankle_inline';RUN=ROOT/'runs/stand5_20261026_v29_hip19_ankle_stance';PARENT=OLD/'checkpoints/model_006621.pt';PROOF=ROOT/'runs/hip19_ankle_stance_deadline_smoke_20261026'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
 q=p.with_suffix('.launch.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
assert not RUN.exists()
checks=read(V/'reward_checks.json');assert checks['passed']
for n,h in checks['source_sha256'].items():assert sha(D/n)==h,n
assert read(V/'ppo_deadline_checks.json')['passed']
assert read(V/'timeout_flags.json')['passed'] and read(V/'timeout_flags.json')['env_sha256']==sha(D/'sable_env.py')
for n,h in read(PROOF/'runtime_manifest.json')['files'].items():assert sha(D/n)==h,n
assert sha(PARENT)==read(OLD/'backup_receipt.json')['checkpoint']['sha256']
backup=read(OLD/'backup_receipt.json');assert read(OLD/'completion.json')['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256']
for p in (D/'assets').rglob('*'):
 if p.is_file():assert sha(p)==sha(OLD/'runtime/assets'/p.relative_to(D/'assets')),str(p)
assert read(PROOF/'completion.json')['reason']=='deadline' and read(PROOF/'completion.json')['training_exit_code']==0
assert sha(Path(read(PROOF/'backup_receipt.json')['archive']))==read(PROOF/'backup_receipt.json')['sha256']
baseline=read(V/'new_reward_inference.json');assert baseline['checkpoint_sha256']==sha(PARENT) and baseline['summary']['survival_16s']==1 and baseline['num_envs']==24
assert baseline['asset_sha256']==sha(D/'assets/sable.xml') and baseline['manifest_sha256']==sha(D/'assets/physical_manifest.json')
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir();(RUN/'diagnostics').mkdir()
for p in D.glob('*.py'):shutil.copy2(p,rt/p.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copy2(V/'new_reward_inference.json',RUN/'baseline_evaluation.json')
for n in ['completion.json','backup_receipt.json','runtime_manifest.json']:shutil.copy2(PROOF/n,RUN/'preflight'/('deadline_'+n))
for n in ['standing.json','fixed.json','symmetry_geometry.json']:shutil.copy2(OLD/'preflight'/n,RUN/'preflight'/('unchanged_physics_'+n))
shutil.copy2(Path(__file__),RUN/'launch_used.py')
# Deadline starts at launch after all verification/copying has finished.
start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(start+datetime.timedelta(minutes=20)).isoformat()
intent={'created_unix':time.time(),'deadline':deadline,'requested_duration_s':1200,'parent':str(PARENT),'parent_sha256':sha(PARENT),'rig_sha256':sha(ROOT/'blender/sable_rig.blend'),'headline':'当前主线：髋≤−19°免罚 · 脚板朝前 · 支撑期踝活动','description':'本轮20分钟。髋角≤−19°免罚，超过后指数扣分；脚板踝→脚尖相对身体前向地面射线，89°起扣分、目标<90°。有效落地后的支撑期踝活动奖励，15°封顶，重复摆动不重复计分并扣回摆分。前足摩擦0.02、后足约1，速度奖励及膝触地终止保留。旧策略基线不算新奖励学习结果。','hip_preferred_max_deg':-19,'hip_penalty_weight':-2.5,'hip_exponential_scale_deg':15,'foot_angle_free_max_deg':89,'foot_angle_weight':-4,'ankle_stance_target_deg':15,'ankle_stance_motion_weight':2,'ankle_stance_reversal_weight':-.5,'ankle_initial_std':.08,'unchanged_physics':True,'automatic_resume':False,'user_authorization':'训20分钟','deadline_basis':'Twenty minutes from formal launch; preflight not counted.'}
atomic(RUN/'stage_intent.json',intent)
cmd=[sys.executable,'-u',str(rt/'supervise.py'),'--run',str(RUN),'--deadline',deadline,'--resume',str(PARENT),'--phase','guided','--reference-strength','.75','--mirror-policy','--front-friction-scale','.02','--num-envs','1024']
with (RUN/'supervisor.log').open('a') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(300):
 if (RUN/'status.json').exists():break
 assert p.poll() is None,'supervisor exited';time.sleep(.2)
assert (RUN/'status.json').exists();atomic(ROOT/'runs/current.json',{'run_dir':str(RUN)})
atomic(ROOT/'results/legged_gym/active_training_plan.json',dict(intent,run=RUN.name,run_dir=str(RUN),status='training',supervisor_pid=p.pid,progress_interval_s=300,web_port=11111))
sys.path.insert(0,str(rt));from progress_report import publish
publish(RUN);print(json.dumps({'started':str(RUN),'deadline':deadline,'supervisor_pid':p.pid}),flush=True)
