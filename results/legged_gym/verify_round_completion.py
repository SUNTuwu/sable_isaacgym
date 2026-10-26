"""Run only after the specified actual cutoff and supervisor archive verification."""
import json,time,datetime,hashlib,argparse
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--deadline',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--expected-reason',choices=['deadline','training_exited'],default='deadline');args=parser.parse_args()
run=args.run.resolve()
deadline=args.deadline;end=datetime.datetime.fromisoformat(deadline).timestamp()
if args.expected_reason=='deadline':assert time.time()>=end,'Actual cutoff has not happened'
completion=read(run/'completion.json');backup=read(run/'backup_receipt.json');proof=read(run/'final_verification.json')
state=read(run/'status.json');config=read(run/'config.json');manifest=read(run/'runtime_manifest.json')
assert completion['deadline']==deadline and completion['reason']==args.expected_reason and completion['training_exit_code']==0
if args.expected_reason=='deadline':assert completion['finished_unix']>=end and state['status']=='stopped_at_deadline'
else:assert state['status']=='stopped_requested'
assert proof['passed'] and proof['archive_sha256']==backup['sha256']
assert Path(backup['archive']).stat().st_size==backup['bytes']
assert all(sha(run/'runtime'/name)==digest for name,digest in manifest['files'].items())
assert config['mirror_policy'] and config['env']['env']['num_observations']==52
assert config['env']['termination']['knee_contact_enabled']
assert config['env']['posture']['hip_stand_tolerance_deg']==5
assert config['env']['rewards']['scales']['hip_stand_exp']==-2.5
assert config['env']['reference']['strength']==state['reference_strength']
checkpoint=backup['checkpoint'];assert sha(Path(checkpoint['path']))==checkpoint['sha256']
assert completion['final_evaluation']['return_code']==0
ev=read(Path(completion['final_evaluation']['report']))
assert ev['iteration']==checkpoint['iteration'] and ev['checkpoint_sha256']==checkpoint['sha256']
assert ev['horizon_s']==16 and ev['num_envs']==24
active=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:process_args=p.joinpath('cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    scripts=[x.decode(errors='replace') for x in process_args if x.endswith(b'.py')]
    if any(str(ROOT) in x and Path(x).name in ['train.py','supervise.py'] for x in scripts):active.append(int(p.name))
assert not active,active
report={'passed':True,'verified_unix':time.time(),'run':run.name,'deadline':deadline,'final_iteration':checkpoint['iteration'],
    'reference_strength':state['reference_strength'],'gait_clearance_weight':config['env']['rewards']['scales']['gait_clearance'],'swing_height_target_m':config['env']['gait']['swing_height_m']*state['reference_strength'],'swing_peak_balance_weight':config['env']['rewards']['scales'].get('swing_peak_balance',0.),'hip_stand_tolerance_deg':5,'hip_stand_penalty_weight':-2.5,
    'reason':completion['reason'],'verifier_sha256':sha(Path(__file__)),'command_range_m_s':state['command_range'],'training_exit_code':0,'project_training_processes':active,'frozen_runtime_files_verified':len(manifest['files']),
    'archive':backup,'supervisor_verified_archive_files':proof['verified_files'],'final_evaluation':ev['summary'],
    'final_priority_diagnostics':ev['priority_diagnostics']['median_across_environments'],
    'receipt_scope':'Generated after archive creation; this receipt is separate from the sealed archive. Server-local evidence, no automatic Windows backup.'}
output=args.output;output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
