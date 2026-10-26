"""Independent180s checkpoint evaluations and videos; no optimizer or restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_io import atomic_json, read_json, sha256

ROOT = Path('/root/gpufree-data/projects/sable-isaacgym')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', default='S2')
    p.add_argument('--interval', type=int, default=180)
    a = p.parse_args()
    runtime = Path(__file__).resolve().parent
    teacher = a.run/'teacher_manifest.json'
    record = ROOT/'results/cmd16_hist6'/('watch_'+a.run.name+'.json')
    (a.run/'evaluations').mkdir(exist_ok=True)
    seen = None
    low_survival_count = 0
    next_due = 0.
    while True:
        state = read_json(a.run/'status.json', {})
        cp = read_json(a.run/'checkpoint.json', {})
        terminal = state.get('status') in ('stopped_at_deadline','stopped_requested','failed','iterations_complete')
        if not cp or (not terminal and time.time()<next_due) or (cp.get('sha256')==seen and not terminal):
            time.sleep(5)
            continue
        stamp = time.strftime('%H%M%S')
        start = time.time()
        marker = {'pid': os.getpid(), 'run': str(a.run), 'iteration': cp['iteration'],
                  'checkpoint_sha256': cp['sha256'], 'started_unix': start, 'final': terminal}
        atomic_json(record, dict(marker, status='evaluating'))
        output = a.run/'evaluations'/('independent_%06d_%s.json' % (cp['iteration'],stamp))
        command = [sys.executable, str(runtime/'evaluate_migration.py'), '--checkpoint', cp['path'],
            '--teacher', str(teacher), '--output', str(output), '--suite', a.stage,
            '--envs-per-scenario', '4', '--seconds', '16']
        try:
            with (a.run/'independent_evaluation.log').open('a') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240)
            if result.returncode:
                raise RuntimeError('Evaluator exit%d'%result.returncode)
            report = read_json(output)
            assert report['checkpoint_sha256']==cp['sha256'] and report['reference_execution_calls']==0
            atomic_json(a.run/'latest_evaluation.json', report)
            if cp['sha256'] != seen:
                low_survival_count = low_survival_count+1 if report['survival_fraction']<.5 else 0
            if low_survival_count >= 3 and not terminal:
                (a.run/'STOP').write_text('Three independent checkpoint evaluations below50% survival; save for diagnosis, do not resume this run.\n')
            directory = ROOT/'results/legged_gym/videos'/('%s_%06d_%s'%(a.run.name,cp['iteration'],stamp))
            directory.mkdir(parents=True,exist_ok=True)
            atomic_json(record, dict(marker,status='recording'))
            with (directory/'record.log').open('w') as log:
                result = subprocess.run([sys.executable,str(runtime/'record_migration_video.py'),
                    '--checkpoint',cp['path'],'--teacher',str(teacher),'--output-dir',str(directory),
                    '--stage',a.stage,'--seconds','8'],stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if result.returncode:
                raise RuntimeError('Recorder exit%d'%result.returncode)
            receipt = read_json(directory/'recording_receipt.json')
            assert receipt['checkpoint_sha256']==cp['sha256'] and sha256(receipt['video'])==receipt['video_sha256']
            subprocess.run(['ffmpeg','-v','error','-i',receipt['video'],'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            video = {'status':'ready','path':receipt['video'],'run':a.run.name,'iteration':cp['iteration'],
                'checkpoint_sha256':cp['sha256'],'sha256':receipt['video_sha256'],'published_unix':time.time(),
                'created_unix':receipt['created_unix'],'description':'学生实际独立回放；无教师与参考执行。固定种子947，未筛选；'+('发生跌倒，保留失败片段。' if receipt['fell'] else '本片段未跌倒。'),
                'recording_receipt':str(directory/'recording_receipt.json'),'final':terminal}
            atomic_json(directory/'published_receipt.json', video)
            if read_json(ROOT/'runs/current.json',{}).get('run_dir')==str(a.run):
                atomic_json(ROOT/'results/legged_gym/side_video_preview.json',video)
            atomic_json(record,dict(marker,status='published',finished_unix=time.time(),evaluation=str(output),video=video))
        except Exception as error:
            atomic_json(record,dict(marker,status='failed',finished_unix=time.time(),error=type(error).__name__+': '+str(error)))
        seen = cp['sha256']
        if terminal:
            break
        next_due = max(start+a.interval,time.time()+5)


if __name__ == '__main__':
    main()
