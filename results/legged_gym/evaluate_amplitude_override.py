"""Explicit inference-only amplitude override around the unchanged evaluator."""
import argparse,sys,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--strength',type=float,required=True);a=p.parse_args()
assert 0<a.strength<=1 and not a.output.exists()
run=a.checkpoint.resolve().parent.parent;sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi
import evaluate
from run_io import read_json,atomic_json,sha256
used_config=read_json(run/'config.json');used_config['env']['reference']['strength']=a.strength
def override(path,*args,**kwargs):
    if Path(path).resolve()==run/'config.json':return used_config
    return read_json(path,*args,**kwargs)
evaluate.read_json=override
sys.argv=['evaluate.py','--checkpoint',str(a.checkpoint.resolve()),'--output',str(a.output.resolve())]
evaluate.main()
report=read_json(a.output);report['inference_overrides']={'reference_strength':a.strength};report['override_source_sha256']=sha256(__file__);report['parent_config_sha256']=sha256(run/'config.json');report['configuration_used']=used_config;report['scope']='Independent inference with explicit reference-strength override; frozen runtime and parent checkpoint unchanged. This is not a trained checkpoint at the new amplitude.'
atomic_json(a.output,report)
print('AMPLITUDE_OVERRIDE_SUMMARY',json.dumps({'strength':a.strength,'summary':report['summary'],'priority':report['priority_diagnostics']['median_across_environments'],'strict_steps':[sum(s['steps_per_foot'][i] for s in report['samples']) for i in range(4)]},ensure_ascii=False))
