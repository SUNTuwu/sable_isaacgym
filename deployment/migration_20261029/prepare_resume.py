"""Prepare a new teacher-bank round without starting training or changing old runs.

Uses only the standard library; run after restoring the private baseline package.
The original 94-observation / 15-output reference teachers are preserved verbatim.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil

BASELINE = 'baselines/user_accepted_20261029_120250'
IDS = ('01_FORWARD_POLICY', '04_TURN_LEFT', '05_TURN_RIGHT', '06_BACKWARD',
       '02_STRAFE_LEFT', '03_STRAFE_RIGHT', '09_BACKWARD_WALK', '10_FORWARD_TROT')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def verify_baseline(baseline):
    baseline = baseline.resolve()
    manifest = json.loads((baseline / 'manifest.json').read_text())
    failures = []
    for name, expected in manifest['files'].items():
        path = (baseline / name).resolve()
        if baseline not in path.parents:
            raise ValueError('Manifest path escapes baseline: ' + name)
        if not path.is_file() or sha(path) != expected:
            failures.append(name)
    if failures:
        raise ValueError('Missing or changed baseline files: ' + ', '.join(failures))
    if manifest['iteration'] != 2084 or not manifest['yaw_excluded']:
        raise ValueError('Unexpected baseline identity')
    return manifest


def prepare(project, name, envs_per_teacher):
    project = project.resolve()
    if Path(name).name != name or not name.startswith('teacher_bank_fullref_'):
        raise ValueError('Use a single run name beginning teacher_bank_fullref_')
    if envs_per_teacher < 1:
        raise ValueError('envs-per-teacher must be positive')
    baseline = project / BASELINE
    manifest = verify_baseline(baseline)
    sources = json.loads((baseline / 'new_baseline_sources.json').read_text())
    preferences = json.loads((project / 'results/legged_gym/training_preferences.json').read_text())
    if any(preferences.get(key) != 180 for key in
           ('progress_interval_s', 'evaluation_interval_s', 'video_interval_s')):
        raise ValueError('Frozen teacher runtime reports every 180s; preferences differ')
    duration = int(preferences['default_duration_s'])
    if duration <= 0:
        raise ValueError('Training duration must be positive')
    destination = project / 'runs' / name
    source = project / 'runs' / (name + '_source2084')
    if destination.exists() or source.exists():
        raise FileExistsError('Refuse to overwrite a source or a prepared run')
    # Validate all inputs before writing either new directory.
    for ident in IDS:
        checkpoint = baseline / 'models' / ident / 'model_002084.pt'
        if sha(checkpoint) != sources[ident]['checkpoint']['sha256']:
            raise ValueError('Checkpoint identity mismatch: ' + ident)
    source.mkdir(parents=True)
    for ident in IDS:
        directory = source / ident / 'checkpoints'
        directory.mkdir(parents=True)
        checkpoint = directory / 'model_002084.pt'
        shutil.copy2(baseline / 'models' / ident / checkpoint.name, checkpoint)
        write_json(directory.parent / 'checkpoint.json',
                   dict(path=str(checkpoint), sha256=sha(checkpoint), iteration=2084,
                        reason='byte-identical copy of protected user baseline'))
    write_json(source / 'migration_source.json',
               dict(baseline=str(baseline), baseline_manifest_sha256=sha(baseline / 'manifest.json'),
                    iteration=manifest['iteration'], contains_optimizer_state=True))
    destination.mkdir()
    shutil.copytree(baseline / 'runtime', destination / 'runtime')
    shutil.copytree(baseline / 'reference', destination / 'reference')
    shutil.copy2(baseline / 'previous_control.json', destination / 'control.json')
    runtime = destination / 'runtime'
    command = ['python', str(runtime / 'teacher_bank_core.py'), '--bank', str(destination),
               '--reference', str(destination / 'reference'), '--resume-bank', str(source),
               '--duration', str(duration), '--envs-per-teacher', str(envs_per_teacher),
               '--reward-cleanup-profile', 'reference_local_v2',
               '--control-mode', 'reference_plus_residual', '--reference-strength', '1',
               '--time-scale', '1', '--offset-limit-deg', '20', '--imitation-scale', '0',
               '--exploration-std', '.15', '--minimum-action-std', '.02']
    # No --allow-objective-change: retain saved means, std and Adam state.
    receipt = dict(status='prepared_not_started', source_bank=str(source),
                   bank=str(destination), command=command, duration_s=duration,
                   evaluation_interval_s=180, video_type='kinematic_trace',
                   prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   source_iteration=2084, runtime_hashes={
                       str(p.relative_to(runtime)): sha(p)
                       for p in runtime.rglob('*') if p.is_file()})
    write_json(destination / 'migration_preparation.json', receipt)
    start = destination / 'start.sh'
    # Watchdog gets the actual trainer PID and a fresh deadline at execution time.
    lines = [
        '#!/usr/bin/env bash', 'set -Eeuo pipefail',
        'source ' + shlex.quote(str(project / 'deployment/runtime.sh')),
        'cd ' + shlex.quote(str(project)),
        'export PYTHONPATH=' + shlex.quote(str(runtime)),
        'export SABLE_NIGHT_CONTROL_FILE=' + shlex.quote(str(destination / 'control.json')),
        'export PYTHONUNBUFFERED=1',
        'test ! -e ' + shlex.quote(str(destination / 'launch_receipt.json')),
        # flock prevents two invocations of this generated start script.
        'exec 9>' + shlex.quote(str(destination / 'launch.lock')),
        'flock -n 9',
        'test ! -e ' + shlex.quote(str(destination / 'launch_receipt.json')),
        'python ' + shlex.quote(str(project / 'deployment/migration_20261029/start_prepared.py'))
        + ' --bank ' + shlex.quote(str(destination)),
    ]
    start.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=receipt['status'], bank=str(destination),
                          source_bank=str(source), start_command='bash ' + shlex.quote(str(start))),
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('--envs-per-teacher', type=int, default=1024)
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_baseline(args.project / BASELINE)
        print(json.dumps(dict(passed=True, verified_files=len(manifest['files']), iteration=2084)))
    elif args.name:
        prepare(args.project, args.name, args.envs_per_teacher)
    else:
        parser.error('Specify --verify-only or --name; no training starts in this script')


if __name__ == '__main__':
    main()
