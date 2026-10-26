"""Launch the newly authorized powered-ankle round after actual GPU validation."""
from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,sys,time,tarfile
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');D=ROOT/'deployment/legged_gym_powered_ankles_20261026';V=D/'verification'
RUN=ROOT/'runs/stand5_20261026_v27_powered_ankles';PARENT=ROOT/'runs/stand5_20261026_v26_stride_hip9/checkpoints/model_006130.pt'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
 tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
assert not RUN.exists(),'Never overwrite a run'
for name in ['standing','fixed','symmetry_geometry','timeout_flags','transfer_checks']:
 assert read(V/(name+'.json'))['passed'],name
base=read(V/'parent_baseline.json');assert base['summary']['survival_16s']>=.9 and base['num_envs']==24 and base['horizon_s']==16
assert base['asset_sha256']==sha(D/'assets/sable.xml') and base['manifest_sha256']==sha(D/'assets/physical_manifest.json')
assert read(V/'transfer_checks.json')['parent_sha256']==sha(PARENT)
proof=ROOT/'runs/powered_ankle_deadline_smoke_20261026';complete=read(proof/'completion.json');backup=read(proof/'backup_receipt.json')
assert complete['training_exit_code']==0 and complete['reason']=='deadline' and complete['status']['status']=='stopped_at_deadline'
assert sha(Path(backup['archive']))==backup['sha256']
assert complete['final_evaluation']['return_code']==0
assert sha(proof/'runtime/supervise.py')==sha(D/'supervise.py')
# Training/control/geometry exactly match the version actually exercised in deadline proof.
for name in ['train.py','sable_env.py','sable_config.py','guided_reference.py','gait_support.py','equivariant_policy.py','policy_symmetry.py','powered_transfer.py','knee_contact.py','stride_progress.py','assets/sable.xml','assets/physical_manifest.json','assets/gait_reference.json']:
 assert sha(proof/'runtime'/name)==sha(D/name),name
processes=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any('python' in line and ('/train.py ' in line or '/supervise.py ' in line) for line in processes),'Existing project trainer/supervisor'
RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'evaluations').mkdir();(RUN/'checkpoints').mkdir()
for p in D.glob('*.py'):shutil.copy2(p,rt/p.name)
shutil.copytree(D/'assets',rt/'assets');shutil.copytree(V,RUN/'preflight')
shutil.copytree(ROOT/'runs/powered_ankle_transfer_smoke_20261026',RUN/'preflight/transfer_smoke')
shutil.copy2(proof/'completion.json',RUN/'preflight/deadline_completion.json');shutil.copy2(proof/'backup_receipt.json',RUN/'preflight/deadline_backup_receipt.json')
shutil.copy2(Path(backup['archive']),RUN/'preflight/deadline_smoke_backup.tar.gz')
shutil.copy2(V/'parent_baseline.json',RUN/'baseline_evaluation.json');shutil.copy2(Path(__file__),RUN/'launch_used.py')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));deadline=(now+datetime.timedelta(hours=1)).isoformat()
intent={'created_unix':time.time(),'deadline':deadline,'automatic_resume':False,'headline':'当前主线：动力脚踝 · 低前足摩擦 · 提高前向速度','description':'新模型15电机、14动作、58维观测；后踝70°起步，限位0–160°；小腿55mm、后脚胶囊40mm。前足摩擦0.02，后足约1；速度跟随奖励权重8，直线前向速度奖励权重2（0.12m/s封顶，停车不奖励），命令0.03–0.09m/s。膝盖触地终止；沿用髋≤−9°偏好和有效步长奖励。旧模型策略按关节映射迁移，新踝输出从零残差学习。','front_floor_mu':.02,'rear_floor_mu':1.,'parent':str(PARENT),'parent_sha256':sha(PARENT),'rig_sha256':sha(ROOT/'blender/sable_rig.blend'),'policy_actions':14,'observations':58,'independent_actuators':15,'tracking_velocity_weight':8,'forward_speed_weight':2,'forward_speed_cap_m_s':.12,'command_range_m_s':[.03,.09],'hip_preferred_max_deg':-9.,'reference_strength':.75,'power_reward_enabled':False,'deadline_assumption':'One hour from launch, stated to user; no reuse of historical 02:00 deadline.','training_purpose':'Learn active ankle residuals and faster stable forward motion with almost frictionless front feet; no hardware validation.'}
atomic(RUN/'stage_intent.json',intent)
atomic(RUN/'preflight/launch_validation.json',{'passed':True,'standing24x16':True,'mirror_static':True,'one_real_ppo':True,'new_ankle_weights_update':True,'timeout_regression':True,'deadline_save_archive':True,'initial_baseline':base['summary'],'source_hashes':{str(p.relative_to(rt)):sha(p) for p in rt.rglob('*') if p.is_file()},'baseline_evaluator_sha256':base['evaluator_sha256'],'new_evaluation_difference':'Add two-ankle telemetry only; heading step profile fix already in baseline source.'})
cmd=[sys.executable,'-u',str(rt/'supervise.py'),'--run',str(RUN),'--deadline',deadline,'--resume',str(PARENT),'--phase','guided','--reference-strength','.75','--mirror-policy','--front-friction-scale','.02','--num-envs','1024']
with (RUN/'supervisor.log').open('a') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(300):
 if (RUN/'status.json').exists():break
 assert p.poll() is None,'New supervisor exited; inspect its log'
 time.sleep(.2)
assert (RUN/'status.json').exists()
atomic(ROOT/'runs/current.json',{'run_dir':str(RUN)})
atomic(ROOT/'results/legged_gym/active_training_plan.json',dict(intent,run=RUN.name,run_dir=str(RUN),status='training',supervisor_pid=p.pid,progress_interval_s=300,web_port=11111))
sys.path.insert(0,str(rt));from progress_report import publish
publish(RUN)
print('STARTED',json.dumps({'run':str(RUN),'deadline':deadline,'supervisor_pid':p.pid,'status':read(RUN/'status.json')['status']},ensure_ascii=False),flush=True)
