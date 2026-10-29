"""Explicit entrypoint for a prepared new round; never called by migration checks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from prepare_resume import IDS


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    args = parser.parse_args()
    bank = args.bank.resolve()
    project = Path(os.environ['SABLE_GYM_ROOT']).resolve()
    if bank.parent != project / 'runs' or not bank.name.startswith('teacher_bank_fullref_'):
        raise ValueError('Prepared bank must be a new project run')
    receipt_path = bank / 'launch_receipt.json'
    if receipt_path.exists() or (bank / 'STOP').exists():
        raise FileExistsError('Refuse to restart an already launched or stopped run')
    prepared = json.loads((bank / 'migration_preparation.json').read_text())
    for name, expected in prepared['runtime_hashes'].items():
        if sha(bank / 'runtime' / name) != expected:
            raise ValueError('Prepared frozen runtime changed: ' + name)
    command = prepared['command'][:]
    command[0] = sys.executable
    # Freeze reward settings per run rather than reading the old live monitor.
    os.environ['SABLE_NIGHT_CONTROL_FILE'] = str(bank / 'control.json')
    deadline = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=prepared['duration_s'])
    command += ['--deadline', deadline.isoformat()]
    receipt = dict(status='starting', bank=str(bank), command=command,
                   source_bank=prepared['source_bank'], source_iteration=2084,
                   deadline=deadline.isoformat(), control_sha256=sha(bank / 'control.json'))
    # Reserve this run atomically, including when invoked without start.sh/flock.
    with receipt_path.open('x') as handle:
        json.dump(receipt, handle, indent=2)
    with (bank / 'train.log').open('xb') as train_log, (bank / 'watchdog.log').open('xb') as watch_log:
        child = subprocess.Popen(command, cwd=project, stdout=train_log, stderr=subprocess.STDOUT)
        receipt['pid'] = child.pid
        save(receipt_path, receipt)
        watch = None
        try:
            watch = subprocess.Popen(
                [sys.executable, str(bank / 'runtime/watch_teacher_bank.py'), '--bank', str(bank),
                 '--pid', str(child.pid), '--deadline', deadline.isoformat(), '--final-grace-s', '180'],
                cwd=project, stdout=watch_log, stderr=subprocess.STDOUT)
            # Confirm actual watchdog startup, not just successful Popen.
            limit = time.monotonic() + 15
            while not (bank / 'deadline_watchdog.json').exists():
                if child.poll() is not None or watch.poll() is not None or time.monotonic() >= limit:
                    raise RuntimeError('Trainer/watchdog exited or failed startup; inspect logs')
                time.sleep(.2)
            state = json.loads((bank / 'deadline_watchdog.json').read_text())
            if state['target_pid'] != child.pid:
                raise ValueError('Watchdog targets a different process')
            receipt.update(status='training', watchdog_pid=watch.pid, watchdog_startup_verified=True)
            save(receipt_path, receipt)
            code = child.wait()
            watch.wait(timeout=30)
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if watch is not None and watch.poll() is None:
                watch.terminate()
                watch.wait(timeout=30)
            receipt.update(status='failed', finished_unix=time.time())
            save(receipt_path, receipt)
            raise
    state = json.loads((bank / 'teacher_bank.json').read_text())
    resume = json.loads((bank / 'resume_verification.json').read_text())
    if code != 0 or not resume['passed'] or state['status'] != 'stopped_at_deadline':
        receipt.update(status='failed_or_stopped_early', exit_code=code, training_status=state['status'])
        save(receipt_path, receipt)
        raise RuntimeError('Round did not complete normally; preserved run and logs')
    if (set(item['id'] for item in state['teachers']) != set(IDS)
            or set(item['teacher_id'] for item in resume['sources']) != set(IDS)
            or state['iteration'] <= prepared['source_iteration']):
        raise ValueError('Expected all eight resumed teachers and a new training iteration')
    expected_checkpoints = {}
    for teacher in state['teachers']:
        directory = bank / teacher['id']
        checkpoint = json.loads((directory / 'checkpoint.json').read_text())
        path = Path(checkpoint['path']).resolve()
        report = json.loads((directory / 'latest_evaluation.json').read_text())
        if bank not in path.parents or sha(path) != checkpoint['sha256']:
            raise ValueError('Final checkpoint mismatch: ' + teacher['id'])
        if (report['checkpoint']['sha256'] != checkpoint['sha256']
                or report['checkpoint_iteration'] != checkpoint['iteration']):
            raise ValueError('Final evaluation does not match checkpoint: ' + teacher['id'])
        expected_checkpoints[str(path.relative_to(bank.parent))] = checkpoint['sha256']
    archive = bank.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError('Refuse to overwrite a run archive')
    temporary = archive.with_suffix('.gz.partial')
    receipt.update(status='archiving', exit_code=code, iteration=state['iteration'],
                   resume_verified=True, final_evaluations_verified=True, finished_unix=time.time())
    save(receipt_path, receipt)
    with tarfile.open(temporary, 'w:gz') as handle:
        handle.add(bank, arcname=bank.name)
    with tarfile.open(temporary, 'r:gz') as handle:
        for name, expected in expected_checkpoints.items():
            with handle.extractfile(name) as stream:
                if hashlib.sha256(stream.read()).hexdigest() != expected:
                    raise ValueError('Archived checkpoint mismatch: ' + name)
    temporary.replace(archive)
    archive_sha256 = sha(archive)
    save(bank / 'archive_verification.json', dict(passed=True, path=str(archive), sha256=archive_sha256,
                                                 verified_checkpoints=expected_checkpoints))
    receipt.update(status='completed', archive_sha256=archive_sha256, archive_contents_verified=True)
    save(receipt_path, receipt)
    print(json.dumps(dict(status='completed', archive=str(archive), iteration=state['iteration'])))


if __name__ == '__main__':
    main()
