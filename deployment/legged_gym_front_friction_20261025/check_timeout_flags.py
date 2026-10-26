"""Remote bounded regression: timeout flags must describe this step only."""
import argparse,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.runtime.resolve()))
from isaacgym import gymapi
import torch
from preflight import create_env
from sable_config import SableCfg
from run_io import atomic_json,sha256
cfg=SableCfg();cfg.env.episode_length_s=.2
env=create_env(2,noise=False,cfg=cfg);env.reset_noise=0.;env.fixed_command=0.;env.reset()
records=[]
for k in range(25):
    _,_,_,done,info=env.step(torch.zeros(2,12,device=env.device))
    records.append({'step':k,'actual_timeout':env.time_out_buf.cpu().tolist(),'ppo_timeout':info['time_outs'].cpu().tolist(),
        'match':bool(torch.equal(info['time_outs'],env.time_out_buf))})
result={'passed':all(r['match'] for r in records),'mismatched_steps':sum(not r['match'] for r in records),
    'runtime':str(a.runtime),'env_sha256':sha256(a.runtime/'sable_env.py'),'records':records}
atomic_json(a.output,result);print('TIMEOUT_CHECK',json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
env.gym.destroy_sim(env.sim)
if not result['passed']:raise SystemExit(2)
