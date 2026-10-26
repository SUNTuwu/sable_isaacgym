"""Run final inference and video only after PPO exit and archive receipt."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
plan=json.loads((OUT/'adaptation_round_plan_20261026.json').read_text());run=Path(plan['run'])
complete=json.loads((run/'completion.json').read_text());backup=json.loads((run/'backup_receipt.json').read_text())
assert complete['training_exit_code']==0 and complete['reason']=='deadline'
checkpoint=Path(backup['checkpoint']['path']);iteration=backup['checkpoint']['iteration'];tag=run.name.split('_')[2]
assert complete['final_evaluation']['return_code']==0
report=OUT/('speed_steps_%s_final_%06d_20261026.json'%(tag,iteration));assert not report.exists()
state=OUT/'final_collection_20261026.json'
def phase(name):
 state.write_text(json.dumps({'phase':name,'iteration':iteration,'updated_unix':time.time(),'ppo_running':False},indent=2));print(name,iteration,flush=True)
phase('verify_training_archives')
subprocess.run([sys.executable,str(OUT/'verify_adaptation_final_20261026.py')],check=True)
phase('final_speed_step_inference')
audit=run/'diagnostics/hip_audit_runtime/evaluate.py'
with (OUT/'final_speed_steps_20261026.log').open('w') as log:
 subprocess.run([sys.executable,str(audit),'--checkpoint',str(checkpoint),'--output',str(report),'--seconds','24','--command-profile','steps'],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(OUT/'publish_speed_steps_20261026.py'),str(report)],check=True)
phase('final_video_inference')
folder=OUT/'videos'/('%s_%d_final_360p24'%(tag,iteration));assert not folder.exists()
env=dict(os.environ,VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
with (OUT/'final_video_20261026.log').open('w') as log:
 subprocess.run([sys.executable,str(OUT/'record_friction_video_360p.py'),'--checkpoint',str(checkpoint),'--output-dir',str(folder),'--seconds','16','--speeds','.03','.09'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
subprocess.run([sys.executable,str(OUT/'publish_friction_video.py'),'--directory',str(folder)],check=True)
phase('evidence_ready_for_final_review')
