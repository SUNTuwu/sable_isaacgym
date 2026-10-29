"""Verify the straight-motion cross-velocity coordinate repair before preserving and resuming all eight teachers.

The independent yaw process and live reward-control file are never changed here.
The old bank keeps training through the GPU smoke test. Model and Adam equality
are checked by the frozen core before any new optimization step.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import signal
import subprocess
import time

P = Path('/root/gpufree-data/projects/sable-isaacgym')
D = P / 'deployment/night_20261029'
V = Path('/root/gpufree-data/envs/sable-isaacgym-py38')
END = datetime.datetime(2026, 10, 29, 8, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
OVERLAYS = {'night_tuning_straight_cross_candidate.py': 'night_tuning.py'}


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = Path(path).with_name(Path(path).name + '.timing.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def identity(pid):
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        args = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
        if stat[0] == 'Z' or not args: return None
        return dict(pid=pid, start_ticks=stat[19], args=args)
    except FileNotFoundError: return None


def stop_exact(saved):
    if identity(saved['pid']) != saved: raise RuntimeError('Process identity changed before cutover')
    os.kill(saved['pid'], signal.SIGTERM)
    for _ in range(240):
        if identity(saved['pid']) != saved: return
        time.sleep(.5)
    raise RuntimeError('Graceful stop did not finish; refusing an overlapping replacement')


def snapshot_checkpoints(source, destination, ids):
    for _ in range(40):
        paths = [source / ident / 'checkpoint.json' for ident in ids]
        contents = [path.read_bytes() for path in paths]
        values = [json.loads(content) for content in contents]
        if len({v['iteration'] for v in values}) == 1 and all(path.read_bytes() == content for path, content in zip(paths, contents)):
            for ident, value in zip(ids, values):
                assert sha(value['path']) == value['sha256']
                (destination / ident).mkdir(parents=True)
                atomic(destination / ident / 'checkpoint.json', value)
            return values
        time.sleep(.25)
    raise RuntimeError('Cannot pin a coherent eight-teacher smoke source')


def main():
    assert time.time() < END.timestamp() - 600
    receipt_path = P / 'deployment/reference_v2_20261028/launch_receipt.json'
    receipt = read(receipt_path)
    old = Path(receipt['bank'])
    state = read(old / 'teacher_bank.json')
    old_process = identity(state['pid'])
    assert state['status'] == 'training' and old_process and old_process['args'] == receipt['command']
    yaw_receipt = read(P / 'deployment/yaw_sine_20261029/launch_receipt.json')
    yaw_process = identity(yaw_receipt['pid'])
    assert yaw_process and str(yaw_receipt['run']) in yaw_process['args']
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bank = P / 'runs' / ('teacher_bank_fullref_referencev2_straightcross_' + stamp)
    bank.mkdir()
    status = dict(status='preflighting', launcher_pid=os.getpid(), bank=str(bank), source_bank=str(old), source_pid=state['pid'], created_unix=time.time())
    def save(**changes):
        status.update(changes, updated_unix=time.time())
        atomic(bank / 'transition.json', status)
        atomic(D / 'straight_cross_frame_transition.json', status)
    save()
    shutil.copytree(old / 'runtime', bank / 'runtime', ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {}
    manifest = read(D / 'night_tuning_straight_cross_manifest.json')
    for source, target in OVERLAYS.items():
        assert manifest['candidate'] == source and manifest['source'] == target
        assert sha(D / source) == manifest['candidate_sha256']
        assert sha(bank / 'runtime' / target) == manifest['source_runtime_sha256']
        shutil.copy2(D / source, bank / 'runtime' / target)
        py_compile.compile(str(bank / 'runtime' / target), doraise=True)
        hashes[target] = sha(bank / 'runtime' / target)
    source_reference = Path(receipt['command'][receipt['command'].index('--reference') + 1])
    shutil.copytree(source_reference, bank / 'reference')
    ids = [t['id'] for t in state['teachers']]
    assert len(ids) == 8
    pinned = snapshot_checkpoints(old, bank / 'smoke_source', ids)
    (bank / 'smoke').mkdir()
    atomic(bank / 'smoke/teacher_sources.json', {
        ident: dict(checkpoint=checkpoint, source_teacher_id=ident)
        for ident, checkpoint in zip(ids, pinned)
    })
    smoke_control = read(D / 'control.json')
    for settings in smoke_control.get('teachers', {}).values(): settings.pop('reload', None)
    atomic(bank / 'smoke_control.json', smoke_control)
    env = dict(os.environ, SABLE_GYM_ROOT=str(P), SABLE_GYM_ENV=str(V), PYTHONPATH=str(bank / 'runtime'), PYTHONUNBUFFERED='1', TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
    env['PATH'] = str(V / 'bin') + ':' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(V / 'lib') + ':' + str(V / 'lib/python3.8/site-packages/torch/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    command = list(receipt['command'])
    command[1] = str(bank / 'runtime/teacher_bank_core.py')
    for flag, value in [('--bank', str(bank)), ('--reference', str(bank / 'reference')), ('--deadline', END.isoformat())]:
        command[command.index(flag) + 1] = value
    if '--resume-bank' in command: command[command.index('--resume-bank') + 1] = str(old)
    else: command.extend(['--resume-bank', str(old)])
    assert '--allow-objective-change' not in command
    smoke = list(command)
    smoke[smoke.index('--bank') + 1] = str(bank / 'smoke')
    # The source bank remains a direct child of runs; explicit selections pin
    # its immutable checkpoint files while that source keeps training.
    smoke[smoke.index('--envs-per-teacher') + 1] = '32'
    smoke.append('--smoke')
    with (bank / 'ppo_smoke.log').open('w') as log:
        result = subprocess.run(smoke, cwd=P, env=dict(env, SABLE_NIGHT_CONTROL_FILE=str(bank / 'smoke_control.json')), stdout=log, stderr=subprocess.STDOUT, timeout=240)
    assert result.returncode == 0 and read(bank / 'smoke/teacher_bank.json')['status'] == 'smoke_complete', 'Smoke failed; existing trainer untouched'
    assert read(bank / 'smoke/resume_verification.json')['passed']
    save(status='preflight_passed', smoke_source_checkpoints=pinned, frozen_module_sha256=hashes)
    assert read(receipt_path)['bank'] == str(old) and identity(state['pid']) == old_process
    # Let the existing monitor acknowledge any already-applied baseline reload.
    # Carrying an unacknowledged token into a fresh environment could repeat it.
    for _ in range(90):
        if not any(v.get('reload') for v in read(D / 'control.json').get('teachers', {}).values()): break
        time.sleep(1)
    else: raise RuntimeError('Pending bank rollback remains; old training continues, no cutover')
    monitor_receipt = read(D / 'monitor_pid.json')
    monitor_process = identity(monitor_receipt['pid'])
    assert monitor_process and monitor_process['args'][1] == str(D / 'monitor.py')
    stop_exact(monitor_process)
    original_receipt = json.loads(json.dumps(receipt))
    original_pointer = read(P / 'results/teacher_bank/current.json')
    child = watch = None
    canonical_backups = {}
    try:
        control_before = (D / 'control.json').read_bytes()
        assert not any(v.get('reload') for v in json.loads(control_before).get('teachers', {}).values())
        (bank / 'control_at_cutover.json').write_bytes(control_before)
        stop_exact(old_process)
        final = read(old / 'teacher_bank.json')
        assert final['status'] == 'stopped_requested'
        final_sources = [read(old / ident / 'checkpoint.json') for ident in ids]
        assert len({cp['iteration'] for cp in final_sources}) == 1 and all(cp['reason'] == 'final' for cp in final_sources)
        save(status='source_saved', source_final_checkpoints=final_sources)
        # Do not publish new recovery overlays until the replacement has run.
        for source, target in [('night_tuning_straight_cross_candidate.py', 'night_tuning.py')]:
            shutil.copy2(D / target, bank / (target + '.before_straightcross'))
            canonical_backups[target] = bank / (target + '.before_straightcross')
        with (bank / 'train.log').open('w') as log:
            child = subprocess.Popen(command, cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with (bank / 'watchdog.log').open('w') as log:
            watch = subprocess.Popen([str(V / 'bin/python'), str(bank / 'runtime/watch_teacher_bank.py'), '--bank', str(bank), '--pid', str(child.pid), '--deadline', END.isoformat(), '--final-grace-s', '120'], cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(bank=str(bank), pid=child.pid, watchdog_pid=watch.pid, command=command, source_bank=str(old), source_iteration=final['iteration'], deadline=END.isoformat(), status='starting', modified_teachers=['01_FORWARD_POLICY', '06_BACKWARD', '09_BACKWARD_WALK', '10_FORWARD_TROT'], reward_revision='straight_cross_velocity_in_episode_start_frame_v1', frozen_module_sha256=hashes)
        atomic(receipt_path, receipt)
        atomic(bank / 'launch_receipt.json', receipt)
        save(status='verifying_resume', pid=child.pid, watchdog_pid=watch.pid)
        for _ in range(240):
            if child.poll() is not None: raise RuntimeError('Replacement trainer exited; monitor recovery must take over')
            verification_path = bank / 'resume_verification.json'
            if verification_path.exists() and (bank / 'teacher_bank.json').exists():
                verification = read(verification_path)
                live_state = read(bank / 'teacher_bank.json')
                if live_state.get('iteration', 0) > final['iteration']:
                    assert verification['passed'] and len(verification['sources']) == 8
                    assert all(t['optimizer_equal'] and t['policy_equal_before_intentional_change'] and not t['intentional_change'] for t in verification['sources'])
                    assert {t['sha256'] for t in verification['sources']} == {cp['sha256'] for cp in final_sources}
                    assert read(P / 'results/teacher_bank/current.json')['bank'] == str(bank)
                    assert identity(yaw_receipt['pid']) == yaw_process
                    assert (D / 'control.json').read_bytes() == control_before
                    # Recovery now clones this complete runtime, including its
                    # matching reward and phase-clock implementation.
                    for source, target in [('night_tuning_straight_cross_candidate.py', 'night_tuning.py')]:
                        shutil.copy2(D / source, D / target)
                    save(status='completed', resume_verified=True, control_unchanged=True, yaw_process_unchanged=True, new_iteration=live_state['iteration'])
                    break
            time.sleep(.5)
        else: raise RuntimeError('Replacement did not finish a training iteration before verification deadline')
    except BaseException as error:
        failed_stage = status['status']
        # Restore coherent old recovery inputs only after the replacement exits.
        # The monitor can then create a separate baseline-recovery run normally.
        cleanup_error = None
        try:
            for process in (child, watch):
                if process is not None and process.poll() is None:
                    current = identity(process.pid)
                    if current: stop_exact(current)
            for target, backup in canonical_backups.items(): shutil.copy2(backup, D / target)
            atomic(receipt_path, original_receipt)
            atomic(P / 'results/teacher_bank/current.json', original_pointer)
        except BaseException as cleanup:
            cleanup_error = repr(cleanup)
        save(status='failed', failed_stage=failed_stage, error=repr(error), cleanup_error=cleanup_error,
             replacement_pid=child.pid if child is not None else None,
             replacement_watchdog_pid=watch.pid if watch is not None else None)
        raise
    finally:
        if identity(monitor_receipt['pid']) != monitor_process:
            with (D / 'monitor.py.log').open('a') as log:
                monitor = subprocess.Popen(monitor_process['args'], cwd=P, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic(D / 'monitor_pid.json', dict(pid=monitor.pid, profile=monitor_receipt['profile']))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        path = D / 'straight_cross_frame_transition.json'
        if path.exists():
            status = read(path)
            if status.get('launcher_pid') == os.getpid() and status.get('status') != 'failed':
                status.update(failed_stage=status['status'], status='failed', error=repr(error), updated_unix=time.time())
                atomic(path, status)
                atomic(Path(status['bank']) / 'transition.json', status)
        raise
