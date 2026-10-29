from isaacgym import gymapi
import sys,time,json
from pathlib import Path
import torch
from migration_config import student_config
from migration_env import create_migration_env
p=Path(__file__).resolve().parents[2]
cfg=student_config(p/'runs/stand5_20261027_v44_stride80_all',stage='S3',num_envs=2048)
cfg.migration.freeze_waist=False
cfg.migration.moving_start_training=False
cfg.migration.command_frontier_alpha=1.
cfg.migration.history_noise_level=0.
for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
cfg.rewards.scales.alive=1.
cfg.env.episode_length_s=4.
e=create_migration_env(cfg,noise=False)
o,c=e.reset();begin=time.time()
for i in range(100):
 o,c,r,d,info=e.step(torch.zeros(e.num_envs,15,device=e.device))
 assert torch.isfinite(o).all() and torch.isfinite(r).all()
result={'passed':True,'purpose':'2048env physics/PD numeric preflight and JIT warmup; no old weights or learning','steps':100,'elapsed':time.time()-begin,'gpu_memory_bytes':torch.cuda.memory_allocated(),'num_envs':e.num_envs}
Path(sys.argv[1]).write_text(json.dumps(result))
print(json.dumps(result),flush=True)
e.gym.destroy_sim(e.sim)
