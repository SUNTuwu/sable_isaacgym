from isaacgym import gymapi
import torch,types,json,hashlib
from pathlib import Path
from sable_config import SableCfg
from sable_env import SableRobot
D=Path(__file__).parent;cfg=SableCfg();dist=torch.tensor([[.0,.0],[.01,.01],[.019,.019],[.02,.02],[.03,.03],[.01,.03]])
o=types.SimpleNamespace(cfg=cfg,foot_pair_distance=dist);r=SableRobot._reward_foot_spacing(o);assert torch.allclose(r,torch.tensor([1.,.25,.0025,0.,0.,.125]),atol=1e-6)
o.rew_buf=torch.zeros(6);o.reward_names=['foot_spacing'];o.reward_functions=[lambda:r];o.reward_scales={'foot_spacing':-4.};o.episode_sums={'foot_spacing':torch.zeros(6)};SableRobot.compute_reward(o);assert torch.allclose(o.rew_buf,-4*r)
report={'passed':True,'boundary_20mm':True,'negative_cost_survives_clipping':True,'weight':-4,'scope':'Left/right foot-capsule centers, ground XY Euclidean distance, front and hind pairs','source_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['sable_env.py','sable_config.py','stride_progress.py','evaluate.py']}}
(D/'verification/spacing_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
