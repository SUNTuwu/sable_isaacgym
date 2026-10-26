"""Prepare a diagnostic-only copied policy; no PPO or mainline switch."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
SOURCE=ROOT/'deployment/legged_gym_hind20_friction0_20261025'
PARENT=ROOT/'runs/stand5_20261025_v23_friction_low_front'
OUT=ROOT/'results/legged_gym/hind20_zero_preflight'
complete=json.loads((PARENT/'completion.json').read_text())
assert complete['training_exit_code']==0 and complete['reason']=='deadline'
backup=json.loads((PARENT/'backup_receipt.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path(backup['archive']))==backup['sha256']
cp=Path(backup['checkpoint']['path']);assert sha(cp)==backup['checkpoint']['sha256']
OUT.mkdir(exist_ok=False);runtime=OUT/'runtime';runtime.mkdir();(OUT/'checkpoints').mkdir()
for p in SOURCE.glob('*.py'):shutil.copy2(p,runtime/p.name)
shutil.copytree(SOURCE/'assets',runtime/'assets')
shutil.copytree(SOURCE/'verification',OUT/'verification')
shutil.copy2(cp,OUT/'checkpoints'/cp.name)
cfg=json.loads((PARENT/'config.json').read_text())
cfg['env']['material_friction']['front_scale']=0.
cfg['env']['gait']['swing_heights_m']=[.018,.018,.023,.023]
cfg['env']['reference']['amplitude_multipliers']=[1.,1.,4./3.,4./3.]
cfg['env']['reference']['file']=str(runtime/'assets/gait_reference.json')
(OUT/'config.json').write_text(json.dumps(cfg,indent=2))
record={'created_unix':time.time(),'training':False,'parent_checkpoint':str(cp),'checkpoint_sha256':sha(cp),
        'front_floor_mu':0.,'hind_swing_target_m':.023,'hind_acceptance_peak_m':.020,
        'force_power_in_reward':False,'force_measurement_calibrated':False,
        'files':{str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file()}}
(OUT/'diagnostic_runtime_manifest.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'prepared':str(OUT),'checkpoint':str(OUT/'checkpoints'/cp.name),'iteration':complete['checkpoint']['iteration']}))
