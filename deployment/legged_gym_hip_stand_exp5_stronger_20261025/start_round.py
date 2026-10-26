"""Explicit new deadline, fresh validation, immutable run, five-minute reporting.

Run only in the existing GPU-enabled server environment after sourcing runtime.sh.
No old run, checkpoint, credential, package, or Blender source is modified.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

from run_io import atomic_json, read_json, sha256

ROOT = Path('/root/gpufree-data/projects/sable-isaacgym')
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'runs/stand5_20261025_v16_hip_stand_exp5/checkpoints/model_003600.pt'
PARENT_HASH = '6bf550a70a238febded9c234964d282388380bbf06c6454e28bc50c825129651'


def project_processes(names):
    found = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            args = (folder / 'cmdline').read_bytes().split(b'\0')
            args = [s.decode(errors='replace') for s in args if s]
            scripts = [s for s in args if s.endswith('.py')]
            if any(str(ROOT) in s and Path(s).name in names for s in scripts):
                found.append((int(folder.name), scripts))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def checked(script, args, log_path, deadline, allowed_codes=(0,)):
    remaining = deadline - time.time()
    if remaining < 30:
        raise RuntimeError('Deadline too close to perform another validation')
    with log_path.open('w') as log:
        result = subprocess.run([sys.executable, str(HERE / script)] + [str(x) for x in args],
                                stdout=log, stderr=subprocess.STDOUT,
                                timeout=min(600, remaining - 5), cwd=str(ROOT))
    if result.returncode not in allowed_codes:
        raise RuntimeError('%s failed, exit %s; see %s' % (script, result.returncode, log_path))


def dashboard(run):
    known = [ROOT / 'deployment/legged_gym/dashboard.py', ROOT / 'deployment/legged_gym_hip24_20261025/dashboard.py', ROOT / 'deployment/legged_gym_hip24_target_limits_20261025/dashboard.py', HERE / 'dashboard.py']
    running = project_processes({'dashboard.py'})
    operator_dashboard=ROOT/'deployment/legged_gym_knee_termination_20261025/dashboard.py'
    if len(running)==1 and str(operator_dashboard) in running[0][1]:
        subprocess.run([sys.executable,str(HERE/'smoke_services.py')],check=True,timeout=20)
        atomic_json(run/'dashboard_service.json',{'pid':running[0][0],'port':11111,'script':str(operator_dashboard),'reused':True})
        return
    if len(running)==1 and str(HERE/'dashboard.py') in running[0][1]:
        subprocess.run([sys.executable,str(HERE/'smoke_services.py')],check=True,timeout=20)
        atomic_json(run/'dashboard_service.json',{'pid':running[0][0],'port':11111,'script':str(HERE/'dashboard.py'),'reused':True})
        return
    for pid, scripts in running:
        if not any(str(p) in scripts for p in known):
            raise RuntimeError('A different dashboard owns this project; refusing to stop it')
    for pid, _ in running:
        os.kill(pid, signal.SIGTERM)
    until = time.time() + 10
    while True:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('0.0.0.0', 11111))
                break
            except OSError:
                if time.time() > until:
                    raise RuntimeError('Port 11111 is unavailable; no unrelated process was stopped')
        time.sleep(.2)
    with (run / 'dashboard.log').open('a') as log:
        server = subprocess.Popen([sys.executable, '-u', str(HERE / 'dashboard.py'), '--port', '11111'],
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(run / 'dashboard_service.json', {'pid': server.pid, 'port': 11111, 'script': str(HERE / 'dashboard.py')})
    time.sleep(1)
    if server.poll() is not None:
        raise RuntimeError('Dashboard startup failed; see dashboard.log')
    subprocess.run([sys.executable, str(HERE / 'smoke_services.py')], check=True, timeout=20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deadline', required=True, help='Explicit ISO8601 timestamp with time zone, including validation time')
    p.add_argument('--name', default='stand5_20261025_v17_hip_posture_weight')
    a = p.parse_args()
    stamp = datetime.datetime.fromisoformat(a.deadline)
    if stamp.tzinfo is None:
        p.error('Deadline must include its UTC offset')
    deadline = stamp.timestamp()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', a.name):
        p.error('Invalid run name')
    if not 600 <= deadline - time.time() <= 4*3600:
        p.error('Need 10 minutes to 4 hours remaining, including validation')
    run = ROOT / 'runs' / a.name
    if run.exists():
        raise FileExistsError('Refusing to overwrite run: ' + str(run))
    lock = (ROOT / 'runs/.hip24_round.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert sha256(PARENT) == PARENT_HASH
    # Import order is required by Isaac Gym. Fail before changing the current run.
    from isaacgym import gymapi
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; no training or current-run switch has occurred')
    if '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected the existing remote RTX 4090')
    if project_processes({'train.py', 'supervise.py'}):
        raise RuntimeError('An existing project trainer/supervisor is active; refusing another')
    # Verify provenance again immediately before the new validation chain.
    model_sync = read_json(ROOT / 'results/model_sync/20261025_hip_pitch24/sync_manifest.json')
    for name, record in model_sync['files'].items():
        assert sha256(ROOT / name) == record['sha256'], name
    proof = read_json(HERE / 'verification/static_model.json')
    assert proof['passed']
    for name, digest in proof['source_sha256'].items():
        assert sha256(HERE / name) == digest, name
    posture_proof=read_json(HERE/'verification/hip_stand_reward.json')
    assert posture_proof and posture_proof['passed']
    assert all(sha256(HERE/n)==digest for n,digest in posture_proof['source_sha256'].items())
    knee_proof=read_json(HERE/'verification/knee_termination.json')
    assert knee_proof and knee_proof['passed']
    assert all(sha256(HERE/n)==digest for n,digest in knee_proof['source_sha256'].items())
    validation = HERE / 'verification' / ('attempt_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    validation.mkdir()
    for name, args in [
            ('preflight.py', ['--fixed', '--seconds', '2']),
            ('preflight.py', ['--seconds', '16', '--num-envs', '4']),
            ('check_timeout_flags.py', ['--runtime', HERE, '--output', HERE / 'verification/timeout_fixed.json']),
            ('check_gait_signals.py', ['--period-s', '.8', '--guided', '--reference-strength', '.5']),
            ('check_deadline.py', ['--phase', 'guided', '--reference-strength', '.5', '--mirror-policy'])]:
        label = name.replace('.py', '') + ('_fixed' if '--fixed' in args else '')
        checked(name, args, validation / (label + '.log'), deadline)
    smoke = ROOT / 'runs' / (a.name + '_transfer_' + str(int(time.time())))
    checked('train.py', ['--out', smoke, '--deadline', a.deadline, '--resume', PARENT,
                        '--phase', 'guided', '--reference-strength', '.5', '--mirror-policy',
                        '--num-envs', '64', '--iterations', '1'], validation / 'transfer.log', deadline)
    initial = smoke / 'checkpoints' / PARENT.name
    baseline = validation / 'parent_new_model.json'
    reference = validation / 'reference_only.json'
    checked('evaluate.py', ['--checkpoint', initial, '--output', baseline], validation / 'baseline.log', deadline)
    checked('evaluate.py', ['--checkpoint', initial, '--output', reference, '--zero'], validation / 'reference.log', deadline)
    checked('check_guided_stage.py', ['--run', smoke, '--parent', PARENT, '--strength', '.5',
                                    '--baseline', baseline, '--reference-only', reference, '--mirror-policy'],
            validation / 'transfer_check.log', deadline, allowed_codes=(0, 2))
    transfer = read_json(HERE / 'verification/guided_transfer.json')
    assert transfer and all(v for k, v in transfer['checks'].items() if k != 'parent_survival_gate')
    assert transfer['parent_sha256'] == PARENT_HASH
    for name, digest in transfer['source_sha256'].items():
        assert sha256(HERE / name) == digest, name
    # Preserve the established experimental exception solely for initial survival.
    # No structural/numerical/transfer failure is allowed through this condition.
    mode = 'validated' if transfer['passed'] else 'experimental'
    if deadline - time.time() < 600:
        raise RuntimeError('Validation consumed the available window; no formal training was started')
    run.mkdir()
    runtime = run / 'runtime'
    runtime.mkdir()
    for source in HERE.glob('*.py'):
        shutil.copy2(source, runtime / source.name)
    shutil.copytree(HERE / 'assets', runtime / 'assets')
    shutil.copytree(HERE / 'verification', run / 'preflight')
    shutil.copytree(smoke, run / 'preflight/transfer_smoke')
    (run / 'checkpoints').mkdir()
    (run / 'evaluations').mkdir()
    atomic_json(run / 'stage_intent.json', {
        'phase': 'guided', 'mode': mode, 'mirror_policy': True, 'reference_strength': .5,
        'parent_checkpoint': str(PARENT), 'parent_sha256': PARENT_HASH,
        'deadline': a.deadline, 'initial_performance_gate_passed': transfer['checks']['parent_survival_gate'],
        'initial_performance': transfer['parent_with_reference_summary'],
        'priorities': ['kneeling/crawling', 'elbow range', 'foot lift', 'short steps'],
        'changes': ['hip pitch upper limits +24 degrees', 'recomputed reference provenance',
                    'existing contact reward and its proximal mesh ownership retained',
                    'target_limits reward -2 times sum of squared normalized target overshoot; physical hard limits unchanged',
                    'any knee collision shape entering the 0.5mm ground contact envelope terminates the episode, no grace period',
                    'hip pitch standard-stand exponential deviation penalty, weight -2.5, tolerance +/-5deg, scale 15deg; applied after positive reward clipping',
                    'post-settling diagnostics and 300-second reports'],
        'accepted_locomotion': False, 'created_unix': time.time()})
    atomic_json(run / 'status.json', {'status': 'preflight', 'run_dir': str(run), 'deadline': a.deadline,
                                    'updated_unix': time.time(), 'iteration': int(PARENT.stem.split('_')[-1]), 'updates_this_run': 0})
    shutil.copy2(baseline, run / 'baseline_evaluation.json')
    atomic_json(ROOT / 'runs/current.json', {'run_dir': str(run)})
    dashboard(run)
    print('STARTING_NEW_ROUND', str(run), 'deadline', a.deadline, 'mode', mode, flush=True)
    with (run / 'supervisor.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-u', str(runtime / 'supervise.py'), '--run', str(run),
                                  '--deadline', a.deadline, '--resume', str(PARENT), '--phase', 'guided',
                                  '--reference-strength', '.5', '--mirror-policy'], stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    (run / 'supervisor.pid').write_text(str(child.pid))
    result = child.wait()
    if result:
        raise RuntimeError('Supervisor failed; see ' + str(run / 'supervisor.log'))
    completion = read_json(run / 'completion.json')
    backup = read_json(run / 'backup_receipt.json')
    assert completion['training_exit_code'] == 0 and completion['deadline'] == a.deadline
    assert backup and sha256(backup['archive']) == backup['sha256']
    checked_files = 0
    import hashlib
    with tarfile.open(backup['archive']) as archive:
        for item in archive.getmembers():
            if item.isfile():
                current = run.parent / item.name
                if current == run/'dashboard.log' or current == run/'supervisor.log':
                    continue  # Service logs can append after archive creation; compare all frozen evidence.
                assert current.is_file(), item.name
                assert hashlib.sha256(archive.extractfile(item).read()).hexdigest() == sha256(current), item.name
                checked_files += 1
    atomic_json(run / 'final_verification.json', {'passed': True, 'created_unix': time.time(),
                'verified_files': checked_files, 'archive_sha256': backup['sha256'],
                'training_exit_code': completion['training_exit_code'], 'reason': completion['reason'],
                'checkpoint': backup['checkpoint'], 'scope': 'Remote archive integrity; no Windows copy was made by this process.'})
    print('ROUND_FINISHED_AND_ARCHIVE_VERIFIED', str(run), flush=True)


if __name__ == '__main__':
    main()
