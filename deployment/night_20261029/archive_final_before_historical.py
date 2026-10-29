"""Bounded, read-only collection of the 08:30 final artifacts; stdlib only.

--plan never writes. --output creates one NEW ZIP after the fixed deadline and
terminal finalizer receipt. No models, receipts, pointers, or processes mutate.
An evaluation finishing is distinct from its policy passing acceptance.
"""
import argparse
import ast
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import time
import zipfile

END = datetime.datetime(2026, 10, 29, 8, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
IDS = ('01_FORWARD_POLICY', '04_TURN_LEFT', '05_TURN_RIGHT', '06_BACKWARD',
       '02_STRAFE_LEFT', '03_STRAFE_RIGHT', '09_BACKWARD_WALK', '10_FORWARD_TROT')
TERMINAL = ('completed', 'partial_failed_yaw', 'failed')
LIMIT = 256 * 1024 * 1024


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def same_checkpoint(left, right):
    return all(left.get(k) == right.get(k) for k in ('path', 'iteration', 'sha256'))


class Collector:
    def __init__(self, project, max_bytes=LIMIT):
        self.project = Path(project).resolve()
        self.files = {}
        self.issues = []
        self.max_bytes = max_bytes

    def path(self, value):
        path = Path(value)
        path = (path if path.is_absolute() else self.project / path).resolve()
        if self.project not in path.parents:
            raise ValueError('Source path outside project: ' + str(path))
        return path

    def add(self, value, role, required=True, digest=None):
        path = self.path(value)
        if not path.is_file():
            if required: self.issues.append('Missing ' + role + ': ' + str(path))
            return None
        before = identity(path)
        if before[2] > self.max_bytes:
            raise ValueError('File exceeds archive size bound: ' + str(path))
        raw = path.read_bytes()
        if before != identity(path):
            raise RuntimeError('Source changed during read: ' + str(path))
        actual = sha_bytes(raw)
        if digest is not None and actual != digest:
            raise ValueError('SHA mismatch for ' + role + ': ' + str(path))
        relative = path.relative_to(self.project).as_posix()
        previous = self.files.get(relative)
        if previous and previous['sha256'] != actual:
            raise RuntimeError('Source changed between collection passes: ' + str(path))
        self.files[relative] = dict(source=str(path), archive_path='project/' + relative,
                                   size=len(raw), sha256=actual, identity=list(before),
                                   roles=sorted(set((previous or {}).get('roles', []) + [role])))
        if sum(item['size'] for item in self.files.values()) > self.max_bytes:
            raise ValueError('Selected artifacts exceed archive size bound')
        return raw

    def json(self, value, role, required=True, digest=None):
        raw = self.add(value, role, required, digest)
        if raw is None: return None
        try:
            result = json.loads(raw)
            if not isinstance(result, dict): raise ValueError('Expected JSON object')
            return result
        except (ValueError, TypeError) as error:
            self.issues.append('Invalid JSON ' + role + ': ' + str(error))
            return None

    def subset(self, folder, role, suffixes):
        folder = self.path(folder)
        if not folder.is_dir():
            self.issues.append('Missing directory ' + role + ': ' + str(folder))
            return
        for path in sorted(folder.rglob('*')):
            if '__pycache__' not in path.parts and path.suffix in suffixes and path.is_file():
                self.add(path, role)


def select_model(c, selected, folder, ident):
    folder = c.path(folder)
    model = c.path(selected.get('path', ''))
    digest = selected.get('sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest.lower()):
        raise ValueError('Missing or malformed selected model SHA: ' + ident)
    if folder not in model.parents:
        raise ValueError('Model outside selected teacher: ' + ident)
    raw = c.add(model, 'final_model:' + ident, digest=selected.get('sha256'))
    receipt = c.json(folder / 'checkpoint.json', 'checkpoint_receipt:' + ident)
    valid = bool(raw is not None and receipt and receipt.get('reason') == 'final'
                 and receipt.get('path') == selected.get('path')
                 and receipt.get('iteration') == selected.get('iteration')
                 and (receipt.get('sha256') is None or receipt.get('sha256') == selected.get('sha256')))
    if not valid: c.issues.append('Selected final model lacks matching final receipt: ' + ident)
    return dict(teacher_id=ident, source_checkpoint=selected, model_verified=valid,
                evaluation_completed=False, accepted=None)


def runtime_dependencies(c, folder, entrypoints, role):
    """Copy current local import closure, not old launchers/tests/candidate trees."""
    folder = c.path(folder); pending = [folder / name for name in entrypoints]
    chosen = set(); external = set()
    while pending:
        path = c.path(pending.pop())
        if path in chosen: continue
        raw = c.add(path, role)
        if raw is None: continue
        chosen.add(path)
        tree = ast.parse(raw.decode('utf-8-sig'), filename=str(path))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): names.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1): base = base.parent
                    prefix = '.'.join(base.relative_to(folder).parts)
                    module = '.'.join(x for x in (prefix, node.module) if x)
                else: module = node.module or ''
                names.append(module)
                names.extend('.'.join(x for x in (module, alias.name) if x) for alias in node.names if alias.name != '*')
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                func = node.func
                if ((isinstance(func, ast.Name) and func.id == '__import__')
                        or (isinstance(func, ast.Attribute) and func.attr == 'import_module')):
                    names.append(node.args[0].value)
        for name in names:
            if not name: continue
            stem = folder.joinpath(*name.split('.'))
            matches = [p for p in (stem.with_suffix('.py'), stem / '__init__.py') if p.is_file()]
            if matches:
                pending.extend(matches)
                package = stem.parent
                while package != folder and folder in package.parents:
                    if (package / '__init__.py').is_file(): pending.append(package / '__init__.py')
                    package = package.parent
            else: external.add(name.split('.')[0])
    for name in ('bootstrap', 'assets'):
        c.subset(folder / name, role + '_assets', {'.json', '.xml', '.urdf', '.obj', '.stl', '.mtl'})
    for path in folder.glob('*.json'): c.add(path, role + '_configuration')
    external -= {p.relative_to(folder).parts[0].split('.')[0] for p in chosen}
    return dict(entrypoints=entrypoints, python_files=sorted(str(p.relative_to(folder)) for p in chosen),
                external_import_roots=sorted(external),
                method='Static local import closure plus frozen assets/bootstrap; external Python/Isaac Gym environment remains required')


def source_terminal(c, run, status_name, script, flag):
    status = c.json(run / status_name, 'source_terminal_status') or {}
    good = status.get('status') in ('completed', 'stopped_at_deadline', 'stopped_requested')
    if not good: c.issues.append('Source status is not terminal: ' + str(run))
    # Linux-only read; no signal/process startup/GPU calls. Windows fixtures use status evidence.
    active = []
    if os.name == 'posix' and Path('/proc').is_dir():
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit(): continue
            try:
                args = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
                if (len(args) >= 4 and Path(args[1]).resolve() == run / 'runtime' / script
                        and args.count(flag) == 1 and args.index(flag) + 1 < len(args)
                        and Path(args[args.index(flag)+1]).resolve() == run): active.append(int(proc.name))
            except (OSError, ValueError): continue
    if active: c.issues.append('Source trainer is still active: ' + str(active))
    return dict(terminal_status=status.get('status'), active_trainer_pids=active,
                proc_checked=os.name == 'posix' and Path('/proc').is_dir(), verified=good and not active)


def bank_evaluations(c, receipt, pointer, bank, models):
    if not pointer: return
    valid_link = (receipt.get('status') in ('completed', 'partial_failed_yaw')
                  and receipt.get('bank_final_evaluation_status') == 'completed'
                  and receipt.get('final_evaluation_pointer') == pointer
                  and pointer.get('schema') == 'sable-final-evaluation-pointer/v1'
                  and pointer.get('source_bank') == str(bank))
    current = c.json('results/teacher_bank/current.json', 'current_bank_pointer')
    valid_link = valid_link and current and current.get('bank') == str(bank)
    output = c.path(receipt.get('output', ''))
    manifest_path = c.path(pointer.get('manifest_path', ''))
    if manifest_path != output / 'final_evaluation_manifest.json':
        valid_link = False
    if not valid_link:
        c.issues.append('Final pointer is stale or not bound to this terminal finalizer receipt; ignored')
        return
    manifest = c.json(manifest_path, 'final_bank_manifest', digest=pointer.get('manifest_sha256'))
    contract = dict(schema='standalone-final-bank-evaluation/v2', status='completed', mode='evaluation_only',
                    source_bank=str(bank), output=str(output), optimization_steps=0, training_envs=0,
                    total_envs=32, envs_per_teacher=4, horizon_s=24, evaluation_seed=947,
                    actual_simulation_steps=1200, policy_and_optimizer_unchanged=True,
                    source_process_alive=False, first_episode_only=True)
    if not manifest or any(manifest.get(k) != v for k, v in contract.items()):
        c.issues.append('Final bank manifest does not satisfy the full 24-second contract')
        return
    c.add(bank / 'runtime/teacher_bank_core.py', 'manifest_frozen_core', digest=manifest.get('frozen_core_sha256'))
    for name, digest in manifest.get('runtime_python_sha256', {}).items():
        if Path(name).name != name: raise ValueError('Runtime hash name must be a simple filename')
        # The source manifest hashes all historical files. Verify selected live
        # dependencies without dragging every old launcher/probe into this ZIP.
        key = (bank / 'runtime' / name).relative_to(c.project).as_posix()
        if key in c.files: c.add(bank / 'runtime' / name, 'manifest_frozen_python', digest=digest)
    selected = manifest.get('selected_checkpoints', [])
    if len(selected) != 8 or {x.get('teacher_id') for x in selected} != set(IDS):
        c.issues.append('Final bank manifest does not identify exactly eight teachers')
        return
    for item in selected:
        ident = item['teacher_id']; row = models[ident]
        if not same_checkpoint(item, row['source_checkpoint']):
            c.issues.append('Manifest selected checkpoint differs from finalizer: ' + ident)
            continue
        reference = bank / 'reference' / (ident + '.npz')
        reference_sha = item.get('reference_sha256')
        if not isinstance(reference_sha, str) or len(reference_sha) != 64:
            c.issues.append('Final manifest lacks checkpoint/reference SHA binding: ' + ident)
            continue
        c.add(reference, 'reference:' + ident, digest=reference_sha)
        row['reference'] = dict(source=str(reference), archive_path='project/' + reference.relative_to(c.project).as_posix(),
                                sha256=reference_sha, verified_against='final_evaluation_manifest.selected_checkpoints.reference_sha256')
        report = c.json(output / ident / 'latest_evaluation.json', 'final_report:' + ident)
        video = c.json(output / ident / 'video_receipt.json', 'final_replay_receipt:' + ident)
        report_ok = bool(report and report.get('horizon_s') == 24 and report.get('evaluation_seed') == 947
                         and report.get('mode') == 'final_evaluation_only'
                         and report.get('optimization_steps') == 0
                         and report.get('num_envs') == 4
                         and abs(report.get('executed_simulation_s', 0) - 24.) < 1e-6
                         and len(report.get('first_episode_samples', [])) == 4
                         and len(report.get('terminal_episodes', [])) == 4
                         and report.get('policy_and_optimizer_unchanged') is True
                         and same_checkpoint(report.get('source_checkpoint', {}), item))
        video_ok = bool(video and video.get('horizon_s') == 24
                        and video.get('mode') == 'final_evaluation_only'
                        and video.get('iteration') == item['iteration']
                        and same_checkpoint(video.get('source_checkpoint', {}), item))
        trace = None
        if video and video.get('path'):
            trace_path = c.path(video['path'])
            if (output / ident / 'videos') not in trace_path.parents:
                raise ValueError('Replay path outside selected final teacher')
            trace = c.json(trace_path, 'final_replay:' + ident)
        frames = (trace or {}).get('frames', [])
        trace_ok = bool(trace and trace.get('teacher_id') == ident and trace.get('iteration') == item['iteration']
                        and len(frames) == 241 and abs(frames[-1].get('time_s', 0) - 24.) < 1e-6
                        and all(abs(frames[i].get('time_s', -1) - (.02+.1*i)) < 1e-6 for i in range(240)))
        row.update(evaluation_completed=bool(row['model_verified'] and report_ok and video_ok and trace_ok),
                   accepted=report.get('accepted') if report_ok else None,
                   evaluation_protocol='4 deterministic environments, 24 seconds, seed 947, first episode only')
        if not row['evaluation_completed']: c.issues.append('Incomplete or mismatched final bank evaluation: ' + ident)
    for name in ('resume_verification.json', 'teacher_sources.json', 'teacher_bank.json'):
        c.add(output / name, 'final_bank_reproduction', required=False)


def npz_shapes(raw):
    """Read bounded NPY headers only, without NumPy, Torch, unpickling or GPU."""
    shapes = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in ('age', 'fall', 'error', 'count'):
            with archive.open(name + '.npy') as f:
                if f.read(6) != b'\x93NUMPY': raise ValueError('Invalid NPY magic')
                version = tuple(f.read(2))
                size = struct.unpack('<H' if version == (1, 0) else '<I', f.read(2 if version == (1, 0) else 4))[0]
                if size > 65536: raise ValueError('NPY header exceeds bound')
                header = ast.literal_eval(f.read(size).decode('latin1' if version != (3, 0) else 'utf-8'))
                dtype = re.fullmatch(r'[<>=|][?biuf](\d+)', str(header.get('descr')))
                if dtype is None: raise ValueError('Unsupported replay dtype')
                shape = header['shape']
                if not isinstance(shape, tuple) or not all(isinstance(x, int) and x >= 0 for x in shape):
                    raise ValueError('Invalid replay shape')
                payload = math.prod(shape) * int(dtype.group(1))
                if archive.getinfo(name + '.npy').file_size != f.tell() + payload:
                    raise ValueError('Replay array payload length mismatch')
                shapes[name] = list(header['shape'])
    return shapes


def yaw_evaluation(c, receipt, yaw, row):
    result = receipt.get('yaw_final_evaluation', {})
    completion = c.json(yaw / 'completion.json', 'yaw_completion', required=result.get('status') == 'completed')
    report = c.json(yaw / 'evaluation.json', 'yaw_final_report', required=False)
    # A stale training report may be archived as evidence, never promoted to final.
    cp = row['source_checkpoint']
    report_path = yaw / 'evaluation.json'
    matches = bool(result.get('status') == 'completed' and completion
                   and completion.get('final_evaluation_exit_code') == 0
                   and result.get('report_path') == str(report_path)
                   and same_checkpoint(result.get('checkpoint', {}), cp)
                   and report and report.get('final') is True
                   and report.get('checkpoint_iteration') == cp['iteration']
                   and report.get('checkpoint') == cp['path'])
    report_record = c.files.get(report_path.relative_to(c.project).as_posix())
    matches = matches and report_record and report_record['sha256'] == result.get('report_sha256')
    replay = c.add(yaw / 'latest_replay.npz', 'yaw_final_replay' if matches else 'yaw_unverified_replay', required=bool(matches))
    replay_record = c.files.get((yaw / 'latest_replay.npz').relative_to(c.project).as_posix())
    replay_ok = False
    pair_evidence = {'matching_method': 'not_verified'}
    if matches and replay is not None:
        cfg = c.json(yaw / 'config.json', 'yaw_replay_configuration') or {}
        episode = cfg.get('episode_s', 1 + 4 * math.pi * math.radians(cfg.get('amplitude_deg', 30)) / math.radians(10))
        expected = int(math.ceil(episode / .02))
        gap = (replay_record['identity'][3] - report_record['identity'][3]) / 1e9
        try:
            shapes = npz_shapes(replay)
            replay_ok = (0 <= gap <= 60 and completion.get('finished_unix', 0) >= replay_record['identity'][3] / 1e9
                         and all(shapes[x] == [expected, 16] for x in ('age', 'fall', 'error'))
                         and shapes['count'] == [expected, 16, 4])
            pair_evidence = dict(matching_method='final_completion_and_stable_ordered_timestamps_and_shapes',
                                 embedded_checkpoint_verified=False, publish_gap_s=gap, shapes=shapes,
                                 note='Original NPZ has no embedded checkpoint; this is bounded provenance, not content-level checkpoint proof')
        except (ValueError, KeyError, zipfile.BadZipFile, SyntaxError, struct.error) as error:
            pair_evidence['error'] = repr(error)
    corrected = c.json(yaw / 'corrected_evaluation.json', 'yaw_corrected_report', required=False)
    corrected_matches = bool(matches and corrected and corrected.get('source_checkpoint_iteration') == cp['iteration']
                             and corrected.get('provenance', {}).get('source_report_sha256') == report_record['sha256']
                             and replay_record and corrected.get('provenance', {}).get('source_replay_sha256') == replay_record['sha256'])
    row.update(evaluation_completed=bool(row['model_verified'] and matches and replay_ok),
               accepted=report.get('accepted') if matches else None, corrected_report_matches=corrected_matches,
               replay_pair_evidence=pair_evidence,
               evaluation_protocol='16 deterministic environments, final yaw watchdog evaluation',
               finalizer_evaluation_result=result)
    if not row['evaluation_completed']:
        c.issues.append('Yaw final evaluation incomplete; included replay/report may be older diagnostics')


def protected_baselines(c, deployment):
    raw = c.add(deployment / 'protected_baselines_20261029.zip', 'protected_baseline_backup')
    sources = c.json(deployment / 'baseline_sources.json', 'protected_baseline_sources') or {}
    yaw = c.json(deployment / 'yaw_baseline.json', 'protected_yaw_baseline') or {}
    if raw is None: return {'verified': False}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if sum(x.file_size for x in z.infolist()) > 64 * 1024 * 1024:
                raise ValueError('Protected baseline ZIP exceeds expected 64 MiB bound')
            manifest = json.loads(z.read('manifest.json'))
            if set(manifest) != set(IDS) | {'YAW_SINE_STANCE'}:
                raise ValueError('Protected backup must identify nine baseline models')
            for ident, item in manifest.items():
                selected = yaw if ident == 'YAW_SINE_STANCE' else sources.get(ident, {}).get('checkpoint', {})
                if any(selected.get(k) != item.get(k) for k in ('path', 'sha256')):
                    raise ValueError('Protected backup differs from current baseline: ' + ident)
                if sha_bytes(z.read(item['archive_path'])) != item.get('sha256'):
                    raise ValueError('Protected baseline model SHA mismatch: ' + ident)
            return dict(verified=True, model_count=9, sources=manifest)
    except (ValueError, KeyError, zipfile.BadZipFile) as error:
        c.issues.append('Protected baseline backup verification failed: ' + repr(error))
        return dict(verified=False, error=repr(error))


def build_plan(project, now=None, max_bytes=LIMIT):
    now = time.time() if now is None else now
    c = Collector(project, max_bytes)
    d = c.project / 'deployment/night_20261029'
    receipt = c.json(d / 'deadline_finalize_receipt.json', 'deadline_finalizer_receipt') or {}
    status = receipt.get('status')
    base = dict(schema='sable-final-archive/v1', created_unix=now, fixed_deadline_unix=END,
                project=str(c.project), finalizer_status=status, source_read_only=True,
                optimization_steps=0, archive_is_training_dataset=False,
                acceptance_note='evaluation_completed does not mean accepted; inspect each teacher accepted value')
    if now < END or status not in TERMINAL:
        return dict(base, status='not_ready', ready_to_archive=False,
                    reason='Before 08:30 or finalizer has no terminal receipt', files=list(c.files.values()), issues=c.issues)
    if receipt.get('schema') != 'sable-deadline-finalizer/v1' or receipt.get('deadline_unix') != END:
        raise ValueError('Unexpected finalizer schema or deadline')
    if receipt.get('finished_unix', 0) < END:
        raise ValueError('Finalizer completion predates deadline')
    execution = receipt.get('execution_path')
    if execution: c.json(execution, 'finalizer_execution')
    pointer = c.json(d / 'final_evaluation_pointer.json', 'final_evaluation_pointer', required=False)
    models = {}
    runtime_selection = {}; stopped = {}
    bank = c.path(receipt['bank']) if receipt.get('bank') else None
    yaw = c.path(receipt['yaw']) if receipt.get('yaw') else None
    if bank and (bank.parent != c.project / 'runs' or not bank.name.startswith('teacher_bank_fullref_')):
        raise ValueError('Unexpected final bank scope')
    if yaw and (yaw.parent != c.project / 'runs' or not yaw.name.startswith('yaw_sine_')):
        raise ValueError('Unexpected yaw scope')
    selected = receipt.get('bank_final_checkpoints', [])
    if bank and len(selected) == 8 and {x.get('teacher_id') for x in selected} == set(IDS):
        for item in selected:
            ident = item['teacher_id']; models[ident] = select_model(c, item, bank / ident, ident)
            c.add(bank / 'reference' / (ident + '.npz'), 'reference:' + ident)
            c.add(bank / 'reference' / (ident + '.json'), 'reference_metadata:' + ident)
        c.add(bank / 'reference/manifest.json', 'reference_manifest')
        runtime_selection['bank'] = runtime_dependencies(c, bank / 'runtime', ['teacher_bank_core.py', 'watch_teacher_bank.py'], 'frozen_bank_runtime')
        stopped['bank'] = source_terminal(c, bank, 'teacher_bank.json', 'teacher_bank_core.py', '--bank')
        for name in ('teacher_bank.json', 'launch_receipt.json', 'teacher_sources.json', 'resume_verification.json'):
            c.add(bank / name, 'bank_source_configuration', required=name in ('teacher_bank.json', 'launch_receipt.json'))
        bank_evaluations(c, receipt, pointer, bank, models)
    else: c.issues.append('No verified selection of eight final bank models in finalizer receipt')
    if yaw and receipt.get('yaw_final_checkpoint'):
        row = select_model(c, receipt['yaw_final_checkpoint'], yaw, 'YAW_SINE_STANCE')
        models['YAW_SINE_STANCE'] = row
        runtime_selection['yaw'] = runtime_dependencies(c, yaw / 'runtime', ['yaw_sine_teacher.py', 'evaluate_yaw.py', 'watch.py'], 'frozen_yaw_runtime')
        stopped['yaw'] = source_terminal(c, yaw, 'status.json', 'yaw_sine_teacher.py', '--run')
        for name in ('reset_poses.pt', 'initial_poses.json', 'config.json', 'status.json', 'resume.json'):
            c.add(yaw / name, 'yaw_initial_poses_configuration', required=True)
        c.add('deployment/yaw_sine_20261029/launch_receipt.json', 'yaw_launch_receipt')
        yaw_evaluation(c, receipt, yaw, row)
    else: c.issues.append('No verified final yaw checkpoint in finalizer receipt')
    for name in ('control.json', 'monitor.py', 'monitor_status.json', 'monitor_memory.json', 'reviews.jsonl',
                 'baseline_sources.json', 'baseline_physical.json', 'yaw_baseline.json', 'yaw_baseline_verification.json',
                 'handoff.json', 'deadline_finalize.py', 'final_bank_evaluation.py',
                 'translation_failure_adjustments.py', 'yaw_task_severe_candidate.py'):
        c.add(d / name, 'management_and_protected_baselines', required=True)
    baseline_verification = protected_baselines(c, d)
    for name in ('urgent_actions.jsonl', 'manual_rollbacks.jsonl', 'manual_reward_reviews.jsonl',
                 'reload_acknowledgments.jsonl', 'turn_timing_transition.json'):
        c.add(d / name, 'additional_management_decisions', required=False)
    complete = status == 'completed' and len(models) == 9 and all(x['evaluation_completed'] for x in models.values()) and not c.issues
    return dict(base, status='completed' if complete else 'partial_failed', ready_to_archive=True,
                finalizer_failure=receipt.get('error'), teachers=models, issues=c.issues,
                protected_baselines=baseline_verification,
                runtime_selection=runtime_selection, source_terminal_evidence=stopped,
                files=sorted(c.files.values(), key=lambda x: x['archive_path']),
                file_count=len(c.files), total_uncompressed_bytes=sum(x['size'] for x in c.files.values()),
                omitted='All intermediate models, full run directories, videos not explicitly selected, caches, datasets and secrets')


def write_archive(plan, output, now=None):
    now = time.time() if now is None else now
    if now < END or not plan.get('ready_to_archive') or plan.get('status') not in ('completed', 'partial_failed'):
        raise RuntimeError('Refusing archive before fixed deadline or terminal finalizer receipt')
    output = Path(output).resolve(); project = Path(plan['project'])
    allowed = project / 'results/final_archives'
    if project in output.parents and allowed not in output.parents:
        raise ValueError('Within project, archive output must be under results/final_archives')
    if output.exists(): raise FileExistsError('Archive output already exists: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            created = True
            for item in plan['files']:
                path = Path(item['source']); before = identity(path)
                if list(before) != item['identity']: raise RuntimeError('Source changed after planning: ' + str(path))
                raw = path.read_bytes()
                if before != identity(path) or sha_bytes(raw) != item['sha256']:
                    raise RuntimeError('Source changed during archival: ' + str(path))
                archive.writestr(item['archive_path'], raw)
            # Recheck all source identities so a later read cannot hide receipt/pointer changes.
            for item in plan['files']:
                if list(identity(Path(item['source']))) != item['identity']:
                    raise RuntimeError('Source changed before archive completion: ' + item['source'])
            archive.writestr('ARCHIVE_MANIFEST.json', json.dumps(plan, ensure_ascii=False, indent=2))
        return dict(output=str(output), sha256=sha_bytes(output.read_bytes()), size=output.stat().st_size,
                    status=plan['status'], file_count=plan['file_count'], teachers=len(plan['teachers']))
    except BaseException:
        if created: output.unlink()  # Only the new output created by this call; never a source file.
        raise


def verify_archive(path, max_bytes=LIMIT):
    """Verify a downloaded ZIP offline; never follows its source filesystem paths."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or sum(x.file_size for x in z.infolist()) > max_bytes + 4*1024*1024:
            raise ValueError('Duplicate ZIP paths or oversized archive')
        manifest = json.loads(z.read('ARCHIVE_MANIFEST.json'))
        if manifest.get('schema') != 'sable-final-archive/v1' or manifest.get('status') not in ('completed', 'partial_failed'):
            raise ValueError('Unknown archive contract')
        expected = {'ARCHIVE_MANIFEST.json'}
        for item in manifest['files']:
            name = item['archive_path']
            if not name.startswith('project/') or '..' in name.split('/') or '\\' in name:
                raise ValueError('Unsafe archive member')
            expected.add(name)
            if z.getinfo(name).file_size != item['size'] or sha_bytes(z.read(name)) != item['sha256']:
                raise ValueError('Archived artifact SHA/size mismatch: ' + name)
        if set(names) != expected: raise ValueError('ZIP contents differ from manifest')
        for ident in IDS:
            row = manifest['teachers'].get(ident, {})
            if not row.get('evaluation_completed'): continue
            reference = row.get('reference', {})
            name = reference.get('archive_path')
            if not name or Path(name).name != ident + '.npz' or sha_bytes(z.read(name)) != reference.get('sha256'):
                raise ValueError('Offline checkpoint/reference binding failed: ' + ident)
    return dict(output=str(path), sha256=sha_bytes(path.read_bytes()), verified=True,
                status=manifest['status'], file_count=len(manifest['files']),
                evaluation_completed={k:v['evaluation_completed'] for k,v in manifest['teachers'].items()},
                accepted={k:v['accepted'] for k,v in manifest['teachers'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/root/gpufree-data/projects/sable-isaacgym'))
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-archive', type=Path)
    parser.add_argument('--max-mib', type=int, default=256)
    args = parser.parse_args()
    if args.max_mib <= 0 or args.max_mib > 1024: parser.error('--max-mib must be 1..1024')
    if args.verify_archive:
        print(json.dumps(verify_archive(args.verify_archive, args.max_mib*1024*1024), ensure_ascii=False, indent=2))
        return
    if not args.plan and not args.output: parser.error('Use --plan or specify a NEW --output ZIP')
    plan = build_plan(args.project, max_bytes=args.max_mib * 1024 * 1024)
    result = plan if args.plan else write_archive(plan, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
