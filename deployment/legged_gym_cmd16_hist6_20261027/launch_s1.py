"""Seal teacher, freeze S1 runtime, collect complete targets, then train BC."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path('/root/gpufree-data/projects/sable-isaacgym')
SOURCE = Path(__file__).resolve().parent
OUT = ROOT / 'results/cmd16_hist6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)


def main():
    from training_hold import require_training_enabled
    require_training_enabled()
    OUT.mkdir(exist_ok=True, parents=True)
    lock = (OUT / 's1_launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run([sys.executable, '-u', str(SOURCE/'prepare_teacher.py'), '--output', str(OUT/'s0')], check=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run = ROOT / 'runs' / ('cmd16_hist6_s1_bc_' + stamp)
    run.mkdir()
    runtime = run / 'runtime'
    runtime.mkdir()
    for p in SOURCE.glob('*.py'):
        shutil.copy2(p, runtime/p.name)
    shutil.copytree(SOURCE/'assets', runtime/'assets')
    shutil.copytree(SOURCE/'tests', runtime/'tests', ignore=shutil.ignore_patterns('__pycache__'))
    files = {str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file()}
    write(run/'runtime_manifest.json', {'files': files, 'source': str(SOURCE), 'created_unix': time.time()})
    shutil.copy2(OUT/'s0/teacher_manifest.json', run/'teacher_manifest.json')
    shutil.copy2(OUT/'s0/baseline_metrics.json', run/'baseline_metrics.json')
    write(OUT/'current_migration.json', {'run': str(run), 'stage': 'S1', 'launcher_pid': os.getpid()})
    data = OUT / ('teacher_data_' + stamp)
    write(run/'status.json', {'status': 'collecting_teacher', 'stage': 'S1', 'pid': os.getpid(),
                             'started_unix': time.time(), 'run_dir': str(run)})
    with (run/'collection.log').open('w') as log:
        collected = subprocess.run([sys.executable, '-u', str(runtime/'collect_teacher.py'),
            '--teacher', str(run/'teacher_manifest.json'), '--output', str(data),
            '--num-envs', '256', '--seconds', '40'], stdout=log, stderr=subprocess.STDOUT)
    if collected.returncode:
        write(run/'status.json', {'status': 'failed', 'stage': 'teacher_collection', 'exit_code': collected.returncode})
        raise RuntimeError('Teacher collection failed; retained run '+str(run))
    prefs = json.loads((ROOT/'results/legged_gym/training_preferences.json').read_text())
    start = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    deadline = (start + datetime.timedelta(seconds=prefs['default_duration_s'])).isoformat()
    command = [sys.executable, '-u', str(runtime/'train_bc.py'), '--dataset', str(data),
               '--run', str(run), '--deadline', deadline]
    write(run/'launch.json', {'command': command, 'deadline': deadline, 'preferences': prefs,
                             'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip()})
    with (run/'train.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    write(run/'completion.json', {'training_exit_code': result.returncode, 'completed_unix': time.time(),
                                 'deadline': deadline, 'automatic_resume': False})
    archive = OUT / (run.name+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(run, arcname=run.name)
    write(run/'backup_receipt.json', {'archive': str(archive), 'sha256': sha(archive),
                                     'created_unix': time.time()})
    print(json.dumps({'run': str(run), 'exit_code': result.returncode}), flush=True)


if __name__ == '__main__':
    main()
