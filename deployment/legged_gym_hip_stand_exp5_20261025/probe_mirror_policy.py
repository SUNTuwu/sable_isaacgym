"""Remote inference-only mirror/ensemble comparison using the frozen evaluator.

No checkpoint changes, learning, geometry changes, or additional actuation.
"""
import argparse,sys,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['mirrored','symmetric'],required=True);a=p.parse_args()
run=a.checkpoint.resolve().parent.parent;sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi
import torch
import evaluate
from policy_symmetry import Mirror
from run_io import read_json,sha256
cfg=read_json(run/'config.json');m=read_json(run/'runtime/assets/physical_manifest.json')
assert not cfg.get('mirror_policy'),'This diagnostic compares only an unprojected parent policy'
mirror=Mirror(m['policy_joint_names'],cfg['env']['env']['num_observations'],'cuda:0')
original=evaluate.ActorCritic.act_inference;stats={'sum_sq':0.,'elements':0}
def inference(self,obs):
    direct=original(self,obs)
    reflected=mirror.actions(original(self,mirror.observations(obs)))
    stats['sum_sq']+=float((direct-reflected).square().sum());stats['elements']+=direct.numel()
    return reflected if a.mode=='mirrored' else (direct+reflected)*.5
evaluate.ActorCritic.act_inference=inference
write=evaluate.atomic_json
def write_with_scope(path,data):
    data.update(controller_variant=a.mode,probe_source_sha256=sha256(__file__),
        mirror_source_sha256=sha256(Path(__file__).with_name('policy_symmetry.py')),
        mean_policy_asymmetry_rms_action=math.sqrt(stats['sum_sq']/max(1,stats['elements'])),
        comparison_scope='Inference-only transformed or averaged policy means, plus unchanged fixed reference. Not a newly trained policy or gait acceptance.')
    write(path,data)
evaluate.atomic_json=write_with_scope
sys.argv=[str(run/'runtime/evaluate.py'),'--checkpoint',str(a.checkpoint),'--output',str(a.output)]
evaluate.main()
