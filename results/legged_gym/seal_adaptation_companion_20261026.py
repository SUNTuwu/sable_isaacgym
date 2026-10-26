"""Seal post-run evidence separately from immutable training archives."""
import hashlib,json,shutil,tarfile,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym';POST=ROOT/'postrun/adaptation_speedtrack20_20261026'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert sha(target)==sha(p),(p,target)
 else:shutil.copy2(p,target)
plan=read(OUT/'adaptation_round_plan_20261026.json');receipt=read(POST/'completion_receipt.json')
assert plan['status']=='complete' and receipt['training_stopped'] and read(OUT/'operator_monitor_status.json')['finished']
archive=POST.parent/(POST.name+'_companion.tar.gz');assert not archive.exists(),'Already sealed'
offline=OUT/'offline_feedback_speedtrack20_20261026.html';assert offline.exists()
for pattern in ['*20261026*.json','*20261026*.png','*20261026*.jsonl']:
 for p in OUT.glob(pattern):copy(p,POST/'snapshots'/p.name)
for name in ['active_training_plan.json','next_training_objectives.json','operator_monitor_status.json','operator_feedback.jsonl','side_video_preview.json']:
 copy(OUT/name,POST/'snapshots'/name)
for p in OUT.glob('*20261026*.py'):copy(p,POST/'source_used'/p.name)
for name in ['record_friction_video_360p.py','publish_friction_video.py','export_offline_feedback_round.py']:
 copy(OUT/name,POST/'source_used'/name)
for p in OUT.glob('*20261026*.cjs'):copy(p,POST/'source_used'/p.name)
for name in ['dashboard.py','dashboard.html','operator_feedback.py']:
 copy(ROOT/'deployment/legged_gym_knee_termination_20261025'/name,POST/'dashboard_source'/name)
for folder in [OUT/'fullstride_probe_20261026',OUT/'adaptation_ui_20261026',OUT/'offline_ui_20261026',OUT/'speed_heading_audit_20261026']:
 if folder.exists():shutil.copytree(folder,POST/'diagnostics'/folder.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
copy(offline,POST/offline.name);copy(offline.with_name(offline.stem+'_receipt.json'),POST/(offline.stem+'_receipt.json'))
video=read(OUT/'side_video_preview.json');final_dir=Path(video['path']).parent
video_dirs=[OUT/'videos/v25_5652_360p24',final_dir]
for folder in video_dirs:
 if folder.exists():shutil.copytree(folder,POST/'videos'/folder.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(POST/'RESTORE.txt').write_text('''This companion supplements the separate immutable v24/v25/v26 training archives.
Extract training archives into new empty directories. Retain each runtime, assets and config.
For the v26 transfer baseline, an identical archived checkpoint is in
preflight/transfer_smoke/checkpoints/model_005800.pt; its SHA matches the baseline report.
The main-run initial checkpoint has separate serialization/runtime metadata.
Use the existing remote Python/IsaacGym environment and projected SymmetricActorCritic inference.
The policy has52observations and12residualactions. Do not substitute bare actor inference.
Final preferred hip angle is <=-9deg via a reward, NOT a physical hard limit. Physical limits remain-85/+24deg.
Front effective floor friction is0.02, NOTzero. Per-foot horizontal-force calibration has not passed; no power reward.
Each all20 sample success requires >=2valid landed20mm steps per foot plus forward displacement and survival.
The separate completed-swing20mm percentages show consistency; sample success does not imply every swing passes.
Velocity diagnostics retain the actual command profile and horizon. Legacy survival_16s key in a24sstep test means completed24s.
Open offline_feedback_speedtrack20_20261026.html with its videos directory preserved. It is read-only.
All evidence here is server-local simulation evidence. No Windows transfer or hardware validation is implied.
''')
files={str(p.relative_to(POST)):sha(p) for p in POST.rglob('*') if p.is_file()}
(POST/'companion_manifest.json').write_text(json.dumps({'created_unix':time.time(),'files':files},indent=2))
tmp=archive.with_suffix('.partial')
with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:tar.add(POST,arcname=POST.name)
tmp.replace(archive);verified=0
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if m.isfile():
   assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(POST.parent/m.name),m.name;verified+=1
result={'passed':True,'created_unix':time.time(),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
 'verified_files':verified,'training_archives':receipt['stages'],'scope':'External verification receipt generated after sealing; not included in the sealed companion.'}
external=POST.parent/(POST.name+'_companion_receipt.json');assert not external.exists();external.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='training_archives'},ensure_ascii=False))
