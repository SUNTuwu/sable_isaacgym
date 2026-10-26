from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time,tarfile
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_ankle_inline_20261026';V=D/'verification';OLD=ROOT/'runs/stand5_20261026_v27_powered_ankles';RUN=ROOT/'runs/stand5_20261026_v28_ankle_inline';PARENT=OLD/'checkpoints/model_006401.pt'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
 q=p.with_suffix('.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
assert not RUN.exists()
for name in ['standing','fixed','symmetry_geometry','timeout_flags','inline_transfer_checks']:assert read(V/(name+'.json'))['passed'],name
baseline=read(V/'parent_baseline.json');assert baseline['summary']['survival_16s']>=.9 and baseline['num_envs']==24 and baseline['horizon_s']==16
assert baseline['evaluator_sha256']==sha(D/'evaluate.py') and baseline['asset_sha256']==sha(D/'assets/sable.xml') and baseline['manifest_sha256']==sha(D/'assets/physical_manifest.json')
assert read(V/'inline_transfer_checks.json')['parent_sha256']==sha(PARENT)
backup=read(OLD/'backup_receipt.json');done=read(OLD/'completion.json');assert done['training_exit_code']==0 and sha(Path(backup['archive']))==backup['sha256']
count=0
with tarfile.open(backup['archive']) as tar:
 for item in tar.getmembers():
  if item.isfile() and Path(item.name).name!='supervisor.log':
   assert hashlib.sha256(tar.extractfile(item).read()).hexdigest()==sha(OLD.parent/item.name),item.name;count+=1
proof=ROOT/'runs/powered_ankle_deadline_smoke_20261026';pr=read(proof/'completion.json');assert pr['training_exit_code']==0 and pr['reason']=='deadline'
for n in ['supervise.py','train.py','sable_env.py','sable_config.py','equivariant_policy.py','policy_symmetry.py']:assert sha(proof/'runtime'/n)==sha(D/n),n
# Same proven supervisor/train code; new mass/collision model passed actual standing/PPO/replay.
deadline=read(OLD/'config.json')['deadline'];assert datetime.datetime.fromisoformat(deadline).timestamp()-time.time()>600
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for p in D.glob('*.py'):shutil.copy2(p,rt/p.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight');shutil.copytree(ROOT/'runs/ankle_inline_transfer_smoke_20261026',RUN/'preflight/transfer_smoke');shutil.copy2(V/'parent_baseline.json',RUN/'baseline_evaluation.json')
shutil.copy2(proof/'completion.json',RUN/'preflight/reused_deadline_completion.json');shutil.copy2(proof/'backup_receipt.json',RUN/'preflight/reused_deadline_archive_receipt.json')
(RUN/'diagnostics').mkdir();shutil.copy2(ROOT/'blender/sable_rig.blend',RUN/'diagnostics/sable_rig_ankle_inline.blend')
for n in ['edit_receipt.json','validation.json']:shutil.copy2(ROOT/'results/model_sync/20261026_ankle_inline'/n,RUN/'diagnostics'/('model_'+n))
shutil.copy2(Path(__file__),RUN/'launch_used.py')
intent=dict(read(OLD/'stage_intent.json'));intent.update(created_unix=time.time(),deadline=deadline,parent=str(PARENT),parent_sha256=sha(PARENT),rig_sha256=sha(ROOT/'blender/sable_rig.blend'),headline='当前主线：踝电机沿小腿安装 · 髋≤−9°免罚',description='已修正左右踝电机壳体：中心对齐膝—踝连线，输出轴线/轴心不变。小腿55mm、后脚40mm，踝站姿70°/限位0–160°。髋≤−9°免罚，>−9°按指数扣分（权重−2.5、尺度15°，是软惩罚）；物理髋范围−85°～+24°。前足摩擦0.02、速度跟随权重8及直线速度奖励2继续；膝盖触地立即终止。',hip_penalty_verified=True,hip_penalty_weight=-2.5,hip_exponential_scale_deg=15.,previous_run_archive_verified_files=count,deadline_assumption='Preserve original round deadline20:54:58; do not extend for model revision.')
atomic(RUN/'stage_intent.json',intent);atomic(RUN/'preflight/launch_validation.json',{'passed':True,'parent':str(PARENT),'parent_sha256':sha(PARENT),'old_archive_verified_files':count,'standing24x16':True,'real_ppo':True,'exact_network_transfer':True,'hip_boundary_and_unclipped_negative_cost':True,'independent_replay24x16':baseline['summary'],'same_supervisor_deadline_proof_reused':True,'deadline':deadline})
cmd=[sys.executable,'-u',str(rt/'supervise.py'),'--run',str(RUN),'--deadline',deadline,'--resume',str(PARENT),'--phase','guided','--reference-strength','.75','--mirror-policy','--front-friction-scale','.02','--num-envs','1024']
with (RUN/'supervisor.log').open('a') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(300):
 if (RUN/'status.json').exists():break
 assert p.poll() is None,'supervisor exited';time.sleep(.2)
assert (RUN/'status.json').exists();atomic(ROOT/'runs/current.json',{'run_dir':str(RUN)})
atomic(ROOT/'results/legged_gym/active_training_plan.json',dict(intent,run=RUN.name,run_dir=str(RUN),status='training',supervisor_pid=p.pid,progress_interval_s=300,web_port=11111))
sys.path.insert(0,str(rt));from progress_report import publish
publish(RUN);print('STARTED',RUN,deadline,p.pid,flush=True)
