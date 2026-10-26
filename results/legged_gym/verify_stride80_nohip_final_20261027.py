"""Read-only final checkpoint/archive audit; no simulation or optimization."""
from pathlib import Path
import hashlib,json,tarfile,time
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
RUN=ROOT/'runs/stand5_20261027_v39_stride80_nohip'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(RUN/'backup_receipt.json')
completion=read(RUN/'completion.json');cp=read(RUN/'checkpoint.json');ev=read(RUN/'latest_evaluation.json')
assert completion['training_exit_code']==0
assert digest(Path(receipt['archive']))==receipt['sha256']
assert cp['iteration']==ev['iteration'] and digest(Path(cp['path']))==cp['sha256']==ev['checkpoint_sha256']
files=[RUN/n for n in ['config.json','stage_intent.json','runtime_manifest.json','checkpoint.json','completion.json','metrics.jsonl','latest_evaluation.json']]
for directory in ['runtime','checkpoints','evaluations','preflight']:
    files.extend(p for p in (RUN/directory).rglob('*') if p.is_file())
with tarfile.open(receipt['archive'],'r:gz') as archive:
    for path in files:
        member=archive.extractfile(RUN.name+'/'+str(path.relative_to(RUN)))
        assert member is not None and hashlib.sha256(member.read()).hexdigest()==digest(path),str(path)
result={'passed':True,'created_unix':time.time(),'run':RUN.name,'iteration':cp['iteration'],'checkpoint_sha256':cp['sha256'],'archive':receipt,'archive_files_compared':len(files),'training_exit_code':0,'scope':'Remote archive and filesystem verification only; no claim of a second-machine backup.','summary':ev['summary'],'ankle_phase':ev['ankle_phase_diagnostics'],'hind_stride_threshold':ev['hind_stride_threshold_diagnostics']}
out=ROOT/'results/legged_gym/stride80_nohip_final_verification_20261027.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
