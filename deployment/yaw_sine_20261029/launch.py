import datetime,json,os,shutil,subprocess
from pathlib import Path
p=Path('/root/gpufree-data/projects/sable-isaacgym')
d=p/'deployment/yaw_sine_20261029';source=Path(json.loads((p/'results/teacher_bank/current.json').read_text())['bank'])
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');b=p/'runs'/('yaw_sine_stance_'+stamp)
b.mkdir();shutil.copytree(source/'runtime',b/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(d/'yaw_sine_teacher.py',b/'runtime/yaw_sine_teacher.py')
v=Path('/root/gpufree-data/envs/sable-isaacgym-py38')
env=dict(os.environ,SABLE_GYM_ROOT=str(p),SABLE_GYM_ENV=str(v),PYTHONPATH=str(b/'runtime'),PYTHONUNBUFFERED='1',TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
env['PATH']=str(v/'bin')+':'+env.get('PATH','');env['LD_LIBRARY_PATH']=str(v/'lib')+':'+str(v/'lib/python3.8/site-packages/torch/lib')+':'+env.get('LD_LIBRARY_PATH','')
cmd=[str(v/'bin/python'),str(b/'runtime/yaw_sine_teacher.py')]
(d/'pending.json').write_text(json.dumps(dict(run=str(b),status='smoke')))
with (b/'smoke.log').open('w') as f:r=subprocess.run(cmd+['--run',str(b/'smoke'),'--envs','32','--smoke'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
assert r.returncode==0,'Independent teacher smoke failed: '+str(b/'smoke.log')
assert json.loads((b/'smoke/status.json').read_text())['status']=='smoke_complete'
with (b/'train.log').open('w') as f:c=subprocess.Popen(cmd+['--run',str(b),'--envs','512','--duration','1200'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(run=str(b),pid=c.pid,source_assets=str(source/'runtime/assets'),duration_s=1200,status='starting',other_eight_untouched=True)
for f in [d/'launch_receipt.json',b/'launch_receipt.json']:f.write_text(json.dumps(r,indent=2))
print(json.dumps(r),flush=True)
