"""Run the declared three independent evaluation processes, then assess."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from acceptance import assess
from run_io import read_json,atomic_json,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--stage',choices=['S2','S3','S5'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--yaw-error-scale',type=float)
    a=p.parse_args()
    a.checkpoint=a.checkpoint.resolve()
    parent_config=read_json(a.checkpoint.parent.parent/'config.json',{})
    yaw_scale=a.yaw_error_scale if a.yaw_error_scale is not None else parent_config.get('yaw_error_scale_rad_s',1.)
    if not math.isfinite(yaw_scale)or yaw_scale<=0:raise ValueError('Invalid evaluation yaw error scale')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    baseline=read_json(a.baseline)
    reports=[]
    for seed in baseline['new_capability_limits']['evaluation_seeds']:
        output=a.output_dir/('seed%d.json'%seed)
        if output.exists():raise FileExistsError('Do not overwrite independent evidence: '+str(output))
        command=[sys.executable,str(Path(__file__).with_name('evaluate_migration.py')),'--checkpoint',str(a.checkpoint),
            '--teacher',str(a.teacher),'--output',str(output),'--suite',a.stage,'--envs-per-scenario','8',
            '--seconds','28' if a.stage=='S5' else '16','--seed',str(seed),
            '--yaw-error-scale',str(yaw_scale)]
        atomic_json(a.output_dir/'progress.json',{'status':'evaluating','seed':seed,'stage':a.stage,
            'checkpoint_sha256':sha256(a.checkpoint),'yaw_error_scale_rad_s':yaw_scale,
            'started_unix':time.time(),'reports_completed':len(reports)})
        with (a.output_dir/('seed%d.log'%seed)).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
        reports.append(read_json(output))
    result=assess(a.stage,reports,baseline)
    result['reports']=[{'path':str((a.output_dir/('seed%d.json'%seed)).resolve()),
        'sha256':sha256(a.output_dir/('seed%d.json'%seed))} for seed in baseline['new_capability_limits']['evaluation_seeds']]
    atomic_json(a.output_dir/'assessment.json',result)
    atomic_json(a.output_dir/'progress.json',{'status':'complete','passed':result['passed'],'finished_unix':time.time()})
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
