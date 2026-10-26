"""Integration check of deadline handling and archive integrity, with a dummy child."""
import argparse,datetime,json,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from run_io import atomic_json,read_json,sha256
p=argparse.ArgumentParser();p.add_argument('--phase',default='walk');p.add_argument('--reference-strength',type=float,default=.25);p.add_argument('--mirror-policy',action='store_true');a=p.parse_args()
here=Path(__file__).parent
run=here/'verification'/('deadline_smoke_'+str(int(time.time())))
(run/'runtime').mkdir(parents=True)
for n in ['supervise.py','run_io.py','progress_report.py']:shutil.copy2(here/n,run/'runtime'/n)
(run/'runtime/train.py').write_text('''import argparse,signal,time,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--deadline');p.add_argument('--phase');p.add_argument('--reference-strength');p.add_argument('--mirror-policy',action='store_true');p.add_argument('--front-friction-scale');p.add_argument('--num-envs');a=p.parse_args()
stop=False
def finish(*args):
 global stop
 stop=True
signal.signal(signal.SIGTERM,finish)
while not stop:time.sleep(.05)
(a.out/'final_saved.txt').write_text('graceful save')
(a.out/'status.json').write_text(json.dumps({'status':'deadline_test_complete','mirror_policy':a.mirror_policy}))
''')
deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=3)).isoformat()
subprocess.run([sys.executable,str(run/'runtime/supervise.py'),'--run',str(run),'--deadline',deadline,'--phase',a.phase,'--reference-strength',str(a.reference_strength)]+(['--mirror-policy'] if a.mirror_policy else []),check=True,timeout=20)
r=read_json(run/'backup_receipt.json');assert sha256(r['archive'])==r['sha256']
assert (run/'final_saved.txt').read_text()=='graceful save'
with tarfile.open(r['archive']) as tar:assert any(n.endswith('/final_saved.txt') for n in tar.getnames())
assert read_json(run/'completion.json')['training_exit_code']==0
assert read_json(run/'completion.json')['reason']=='deadline'
assert read_json(run/'status.json')['mirror_policy']==a.mirror_policy
dest=here/'verification/deadline_stop_backup_verified.tar.gz';shutil.copy2(r['archive'],dest)
atomic_json(here/'verification/deadline_validation.json',{'passed':True,'phase':a.phase,'reference_strength':a.reference_strength,'mirror_policy':a.mirror_policy,'tested_same_supervisor':True,
    'supervisor_sha256':sha256(here/'supervise.py'),'test_archive':dest.name,'test_archive_sha256':sha256(dest),'created_unix':time.time()})
print('DEADLINE_STOP_AND_BACKUP_PASSED',str(run))
