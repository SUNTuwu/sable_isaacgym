"""Run the declared three independent evaluation processes, then assess."""
import argparse
import json
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
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    baseline=read_json(a.baseline)
    reports=[]
    for seed in baseline['new_capability_limits']['evaluation_seeds']:
        output=a.output_dir/('seed%d.json'%seed)
        if output.exists():raise FileExistsError('Do not overwrite independent evidence: '+str(output))
        command=[sys.executable,str(Path(__file__).with_name('evaluate_migration.py')),'--checkpoint',str(a.checkpoint),
            '--teacher',str(a.teacher),'--output',str(output),'--suite',a.stage,'--envs-per-scenario','8',
            '--seconds','28' if a.stage=='S5' else '16','--seed',str(seed)]
        atomic_json(a.output_dir/'progress.json',{'status':'evaluating','seed':seed,'stage':a.stage,
            'checkpoint_sha256':sha256(a.checkpoint),'started_unix':time.time(),'reports_completed':len(reports)})
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
