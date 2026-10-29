"""Independent deadline guard, final evaluation and archive for this task only."""
import json,os,signal,subprocess,sys,time,tarfile
from pathlib import Path
b=Path(sys.argv[1]);pid=int(sys.argv[2]);proc=Path('/proc')/str(pid)
identity=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
def running():
    if not proc.exists():return False
    fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return fields[0]!='Z' and fields[19]==identity
startup=time.time()+300
while running() and not (b/'config.json').exists() and time.time()<startup:time.sleep(2)
deadline=json.loads((b/'config.json').read_text())['deadline_unix'] if (b/'config.json').exists() else time.time()
while running() and time.time()<deadline+120:time.sleep(2)
if running():
    os.kill(pid,signal.SIGTERM)
    stop=time.time()+30
    while running() and time.time()<stop:time.sleep(1)
    if running():os.kill(pid,signal.SIGKILL)
v=Path('/root/gpufree-data/envs/sable-isaacgym-py38');p=Path('/root/gpufree-data/projects/sable-isaacgym')
env=dict(os.environ,SABLE_GYM_ROOT=str(p),PYTHONPATH=str(b/'runtime'),SABLE_GYM_ENV=str(v),TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
env['PATH']=str(v/'bin')+':'+env.get('PATH','');env['LD_LIBRARY_PATH']=str(v/'lib')+':'+str(v/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
with (b/'final_evaluation.log').open('w') as f:r=subprocess.run([str(v/'bin/python'),str(b/'runtime/evaluate_yaw.py'),str(b)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
archive=b.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
    for path in b.iterdir():
        if path.name not in ('runtime','smoke'):t.add(path,arcname=b.name+'/'+path.name)
    for name in ('yaw_sine_teacher.py','evaluate_yaw.py'):t.add(b/'runtime'/name,arcname=b.name+'/runtime/'+name)
(b/'completion.json').write_text(json.dumps(dict(final_evaluation_exit_code=r.returncode,archive=str(archive),finished_unix=time.time()),indent=2))
