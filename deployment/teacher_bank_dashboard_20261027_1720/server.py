"""Read-only multi-teacher dashboard. Python standard library, Python >=3.8."""
import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path('/root/gpufree-data/projects/sable-isaacgym')
FINAL_IDS = frozenset(('01_FORWARD_POLICY', '04_TURN_LEFT', '05_TURN_RIGHT', '06_BACKWARD',
                       '02_STRAFE_LEFT', '03_STRAFE_RIGHT', '09_BACKWARD_WALK', '10_FORWARD_TROT'))
TERMINAL_STATES = frozenset(('completed', 'stopped_at_deadline', 'stopped_requested', 'failed', 'smoke_complete'))


def final_inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return parent in path.parents


def final_checkpoint_equal(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        return (Path(left['path']).resolve() == Path(right['path']).resolve()
                and left['sha256'] == right['sha256'] and left['iteration'] == right['iteration'])
    except (KeyError, TypeError, ValueError):
        return False


def final_trainer_exited(source, proc=Path('/proc')):
    """Linux process evidence, not a stale status or a potentially reused PID."""
    if not proc.is_dir():
        return False  # Unknown process state must never enable a final overlay.
    try:
        for folder in proc.iterdir():
            if not folder.name.isdigit():
                continue
            try:
                if (folder / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                    continue
                argv = (folder / 'cmdline').read_bytes().decode(errors='replace').split('\0')
                if not any(Path(arg).name == 'teacher_bank_core.py' for arg in argv):
                    continue
                for i, arg in enumerate(argv):
                    bank = argv[i + 1] if arg == '--bank' and i + 1 < len(argv) else arg[7:] if arg.startswith('--bank=') else None
                    if bank and Path(bank).resolve() == source.resolve():
                        return False
            except FileNotFoundError:
                continue  # Exited during the scan.
    except (OSError, IndexError, ValueError):
        return False
    return True


def pick(value, *paths):
    for path in paths:
        x = value
        try:
            for part in path.split('.'):
                x = x[int(part)] if isinstance(x, list) else x[part]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if x is not None:
            return x
    return None


def finite_json(value):
    if isinstance(value, float) and not (-float('inf') < value < float('inf')):
        return None
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


class Store:
    def __init__(self, project, bank=None):
        self.project = Path(project).resolve()
        self.bank = Path(bank).resolve() if bank else None
        self.lock = threading.Lock()
        self.resources_at = 0
        self.resources = {}
        self.history_lock = threading.Lock()
        self.histories = {}
        self.history_directory = HERE / 'telemetry_cache'

    def record_history(self, bank, ident, status):
        iteration = pick(status, 'iteration', 'update')
        reward = pick(status, 'metrics.mean_reward_per_control_step', 'metrics.mean_reward', 'metrics.reward')
        if not isinstance(iteration, (int, float)) or not isinstance(reward, (int, float)):
            return
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', bank.name) + '-' + hashlib.sha256(str(bank).encode()).hexdigest()[:10]
        teacher = re.sub(r'[^A-Za-z0-9_.-]', '_', ident)
        path = self.history_directory / name / (teacher + '.jsonl')
        with self.history_lock:
            key = (name, teacher)
            if key not in self.histories:
                values = []
                try:
                    for line in path.read_text(encoding='utf8').splitlines()[-1200:]:
                        try: values.append(json.loads(line))
                        except ValueError: pass
                except OSError:
                    pass
                self.histories[key] = values
            values = self.histories[key]
            if values and values[-1].get('iteration') == iteration:
                return
            row = {'iteration': iteration, 'reward': reward,
                   'survival': pick(status, 'metrics.completed_episode_survival', 'metrics.survival_fraction'),
                   'observed_unix': time.time(), 'source_updated_unix': status.get('updated_unix'), 'source': 'status.json'}
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf8') as f:
                f.write(json.dumps(finite_json(row), allow_nan=False) + '\n')
            values.append(row)
            if len(values) > 1200:
                del values[:-1200]

    def start_collector(self):
        def collect():
            while True:
                try: self.snapshot()
                except (OSError, ValueError, TypeError, KeyError) as error:
                    print('Dashboard telemetry sample skipped:', type(error).__name__, flush=True)
                time.sleep(3)
        threading.Thread(target=collect, name='dashboard-status-sampler', daemon=True).start()

    @staticmethod
    def read(path):
        try:
            return json.loads(path.read_text(encoding='utf8')), None
        except FileNotFoundError:
            return {}, None
        except (OSError, ValueError) as exc:
            return {}, str(exc)

    def bank_path(self):
        if self.bank:
            return self.bank if self.bank.exists() else None
        pointer, _ = self.read(self.project / 'results/teacher_bank/current.json')
        target = pick(pointer, 'bank', 'bank_dir', 'run_dir')
        if isinstance(target, str):
            candidate = Path(target).resolve()
            if self.project in candidate.parents and (candidate / 'teacher_bank.json').exists():
                return candidate
        candidates = list((self.project / 'runs').glob('teacher_bank_*/teacher_bank.json'))
        return max(candidates, key=lambda p: p.stat().st_mtime).parent if candidates else None

    def allowed_media(self, text, base, suffixes=None):
        if not isinstance(text, str) or not text:
            return None
        p = Path(text)
        candidates = [p] if p.is_absolute() else [base / p, self.project / p]
        bank = self.bank_path()
        if bank:
            candidates.append(bank / p)
        for item in candidates:
            try:
                item = item.resolve()
                if self.project not in item.parents or not item.is_file():
                    continue
                if item.suffix.lower() not in (suffixes or ('.mp4', '.webm', '.gif', '.png', '.jpg', '.jpeg')):
                    continue
                return item
            except OSError:
                pass
        return None

    def final_evaluation(self, root, source_status):
        """Read-only, fail-closed overlay; training pointers/receipts stay untouched.

        Revalidated on every API snapshot. No positive-result cache may outlive a
        pointer change, a resumed trainer or altered report/replay files.
        """
        pointer_path = self.project / 'deployment/night_20261029/final_evaluation_pointer.json'
        if not pointer_path.exists():
            return None, None
        try:
            if not final_inside(pointer_path, self.project):
                raise ValueError('Final pointer symlink escaped project')
            pointer = json.loads(pointer_path.read_text(encoding='utf8'))
            if pointer.get('schema') != 'sable-final-evaluation-pointer/v1':
                raise ValueError('Unknown final pointer schema')
            current, error = self.read(self.project / 'results/teacher_bank/current.json')
            source = Path(pointer['source_bank']).resolve()
            if (not root or source != root.resolve() or source.parent != self.project / 'runs'
                    or Path(pick(current, 'bank', 'bank_dir', 'run_dir') or '').resolve() != source):
                raise ValueError('Final evaluation source is not the explicit current bank')
            if source_status.get('status') not in TERMINAL_STATES or not final_trainer_exited(source):
                raise ValueError('Source has no confirmed terminal status and exited trainer')
            manifest_path = Path(pointer['manifest_path']).resolve()
            if (not final_inside(manifest_path, self.project)
                    or not final_inside(manifest_path, self.project / 'results/final_evaluation')
                    or manifest_path.name != 'final_evaluation_manifest.json'):
                raise ValueError('Final manifest path escaped the final evaluation directory')
            raw = manifest_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != pointer['manifest_sha256']:
                raise ValueError('Final manifest SHA256 mismatch')
            manifest = json.loads(raw)
            expected = dict(schema='standalone-final-bank-evaluation/v2', status='completed',
                            mode='evaluation_only', total_envs=32, envs_per_teacher=4, horizon_s=24.,
                            evaluation_seed=947,
                            actual_simulation_steps=1200, optimization_steps=0, training_envs=0,
                            policy_and_optimizer_unchanged=True, first_episode_only=True,
                            natural_terminal_reward_included=True)
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError('Final evaluation protocol is incomplete or unsupported')
            output = manifest_path.parent
            if Path(manifest['source_bank']).resolve() != source or Path(manifest['output']).resolve() != output:
                raise ValueError('Final manifest provenance mismatch')
            selected = manifest['selected_checkpoints']
            if len(selected) != 8 or {item['teacher_id'] for item in selected} != FINAL_IDS:
                raise ValueError('Final manifest does not select the exact eight teachers')
            teachers = {}
            for checkpoint in selected:
                ident = checkpoint['teacher_id']
                cp_path = Path(checkpoint['path']).resolve()
                if not final_inside(cp_path, source / ident):
                    raise ValueError('Source checkpoint escaped its teacher directory')
                source_cp, error = self.read(source / ident / 'checkpoint.json')
                if error or not final_checkpoint_equal(source_cp, checkpoint):
                    raise ValueError('Final selection no longer matches source checkpoint: ' + ident)
                directory = (output / ident).resolve()
                if not final_inside(directory, output):
                    raise ValueError('Final teacher directory escaped output')
                for filename in ('latest_evaluation.json', 'video_receipt.json'):
                    if not final_inside(directory / filename, directory):
                        raise ValueError('Final metadata symlink escaped teacher directory')
                evaluation, ee = self.read(directory / 'latest_evaluation.json')
                video, ve = self.read(directory / 'video_receipt.json')
                if (ee or ve or evaluation.get('teacher_id') != ident
                        or evaluation.get('mode') != 'final_evaluation_only'
                        or evaluation.get('horizon_s') != 24. or evaluation.get('num_envs') != 4
                        or evaluation.get('evaluation_seed') != 947
                        or evaluation.get('optimization_steps') != 0
                        or not isinstance(evaluation.get('executed_simulation_s'), (float, int))
                        or not math.isfinite(evaluation['executed_simulation_s'])
                        or abs(evaluation['executed_simulation_s'] - 24.) > .001
                        or evaluation.get('policy_and_optimizer_unchanged') is not True
                        or not final_checkpoint_equal(evaluation.get('source_checkpoint'), checkpoint)
                        or video.get('mode') != 'final_evaluation_only' or video.get('horizon_s') != 24.
                        or video.get('optimization_steps') != 0
                        or not final_checkpoint_equal(video.get('source_checkpoint'), checkpoint)):
                    raise ValueError('Final report/video protocol or checkpoint mismatch: ' + ident)
                terminal = evaluation.get('terminal_episodes', [])
                samples = evaluation.get('first_episode_samples', [])
                if len(terminal) != 4 or len(samples) != 4:
                    raise ValueError('Final first-episode coverage incomplete: ' + ident)
                for end, count in zip(terminal, samples):
                    seconds = end.get('time_s')
                    if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                            or not 0 < seconds <= 24.001 or not 0 < count <= 1200
                            or abs(seconds - count * .02) > .001
                            or not (end.get('physical_fall') is True or end.get('natural_timeout') is True)
                            or (end.get('natural_timeout') and abs(seconds - 24.) > .001)):
                        raise ValueError('Invalid natural first-episode terminal evidence: ' + ident)
                trace_path = Path(video['path']).resolve()
                if video.get('type') != 'kinematic_trace' or not final_inside(trace_path, directory) or trace_path.suffix != '.json':
                    raise ValueError('Final replay escaped its teacher directory')
                if trace_path.stat().st_size > 25 * 1024 * 1024:
                    raise ValueError('Final replay exceeds size limit')
                trace, error = self.read(trace_path)
                frames = trace.get('frames', [])
                iteration = evaluation.get('iteration')
                if (error or trace.get('teacher_id') != ident or trace.get('iteration') != iteration
                        or video.get('iteration') != iteration or len(frames) != 241):
                    raise ValueError('Replay is not the matching complete 24-second recording: ' + ident)
                for index, frame in enumerate(frames):
                    stamp = frame.get('time_s')
                    expected_time = (index * 5 + 1) * .02 if index < 240 else 24.
                    if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or abs(stamp - expected_time) > .001:
                        raise ValueError('Replay sample timestamps are not a complete final evaluation: ' + ident)
                evaluation = dict(evaluation, source_bank=str(source), final_manifest_sha256=digest)
                video = dict(video, path=str(trace_path), final_manifest_sha256=digest,
                             final_evaluation=True, recorded_duration_s=frames[-1]['time_s'],
                             first_episode_duration_s=terminal[0]['time_s'],
                             scope='One recorded environment; first-episode validity is in each frame active flag')
                teachers[ident] = dict(evaluation=evaluation, video=video)
            return dict(source_bank=str(source), manifest_path=str(manifest_path), manifest_sha256=digest,
                        horizon_s=24., total_envs=32, teachers=teachers), None
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            return None, 'Final evaluation overlay rejected: ' + str(exc)

    def resource_snapshot(self):
        with self.lock:
            if time.time() - self.resources_at < 10:
                return self.resources
            bank = self.bank_path()
            if bank:
                monitor, error = self.read(bank / 'resource_latest.json')
                try:
                    fresh = time.time() - (bank / 'resource_latest.json').stat().st_mtime < 75
                except OSError:
                    fresh = False
                if fresh and monitor.get('gpu_utilization_percent') is not None:
                    result = {'updated_unix': monitor.get('updated_unix', time.time()), 'source': 'bank_resource_monitor',
                              'gpu': [{'name': monitor.get('gpu_name', 'NVIDIA GeForce RTX 4090'),
                                       'utilization_pct': monitor['gpu_utilization_percent'],
                                       'memory_used_mib': monitor.get('gpu_memory_used_mib'),
                                       'memory_total_mib': monitor.get('gpu_memory_total_mib'),
                                       'power_w': monitor.get('gpu_power_w'),
                                       'temperature_c': monitor.get('gpu_temperature_c')} ]}
                    self.resources, self.resources_at = result, time.time()
                    return result
            result = {'updated_unix': time.time(), 'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None, 'gpu': []}
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                for line in p.stdout.splitlines():
                    name, use, used, total, temp = [x.strip() for x in line.split(',')]
                    result['gpu'].append({'name': name, 'utilization_pct': float(use), 'memory_used_mib': float(used),
                                          'memory_total_mib': float(total), 'temperature_c': float(temp)})
                if p.returncode:
                    result['gpu_error'] = p.stderr.strip()
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                result['gpu_error'] = str(exc)
            disk = shutil.disk_usage(self.project)
            result['disk_free_gib'] = disk.free / 1024 ** 3
            self.resources, self.resources_at = result, time.time()
            return result

    def snapshot(self):
        root = self.bank_path()
        if root:
            manifest, error = self.read(root / 'teacher_bank.json')
            bank_config, _ = self.read(root / 'config.json')
        else:
            manifest, error = {}, None
            bank_config = {}
        jobs = pick(manifest, 'teachers', 'teacher_jobs', 'jobs') or []
        if isinstance(jobs, dict):
            jobs = [dict(v, teacher_id=k) if isinstance(v, dict) else {'teacher_id': k} for k, v in jobs.items()]
        if not jobs and root:
            jobs = [{'teacher_id': p.name} for p in root.iterdir() if p.is_dir() and (p / 'status.json').exists()]
        final, final_error = self.final_evaluation(root, manifest)
        teachers = [self.teacher(root, j, manifest, final) for j in jobs]
        yaw_receipt,_=self.read(self.project/'deployment/yaw_sine_20261029/launch_receipt.json')
        if yaw_receipt.get('run'):
            yaw_root=Path(yaw_receipt['run']).resolve()
            if self.project in yaw_root.parents and (yaw_root/'dashboard/YAW_SINE_STANCE/status.json').exists():
                teachers.append(self.teacher(yaw_root,{'id':'YAW_SINE_STANCE','title':'扭身 yaw｜正弦角度跟随','run_dir':str(yaw_root/'dashboard/YAW_SINE_STANCE')},{}))
        return finite_json({'schema': 'sable-teacher-bank-dashboard/1', 'server_unix': time.time(),
                            'bank_directory': str(root) if root else None, 'bank': manifest, 'bank_config': bank_config, 'bank_error': error,
                            'final_evaluation': {key:value for key,value in final.items() if key != 'teachers'} if final else None,
                            'final_evaluation_error': final_error,
                            'waiting_for_bank': root is None, 'teachers': teachers, 'resources': self.resource_snapshot(),
                            'refresh_seconds': 3, 'evaluation_interval_seconds': pick(manifest, 'evaluation_interval_s') or 180})

    def teacher(self, root, job, bank=None, final=None):
        if isinstance(job, str):
            job = {'teacher_id': job}
        ident = str(pick(job, 'teacher_id', 'id', 'name') or 'unknown')
        path = pick(job, 'run_dir', 'run_directory', 'directory') or ident
        d = Path(path)
        d = d if d.is_absolute() else root / d
        d = d.resolve()
        if root.resolve() not in d.parents:
            return {'id': ident, 'title': ident, 'state': 'invalid_path', 'errors': ['Teacher path escaped bank directory']}
        data, errors = {}, []
        for name in ['status', 'checkpoint', 'latest_evaluation', 'video_receipt', 'config', 'completion',
                     'adaptive_curriculum', 'reward_breakdown', 'phase_sampling', 'direction_metrics']:
            data[name], error = self.read(d / (name + '.json'))
            if error:
                errors.append(name + ': ' + error)
        status, cp, evaluation, video = [data[k] for k in ['status', 'checkpoint', 'latest_evaluation', 'video_receipt']]
        # An embedded long rollout can finish after a newer standalone report.
        # Choose by checkpoint, then horizon, rather than last writer wins.
        def evaluation_rank(value):
            step = pick(value, 'iteration', 'checkpoint.iteration')
            horizon = pick(value, 'horizon_s')
            return (step if isinstance(step, int) else -1,
                    horizon if isinstance(horizon, (int, float)) else 0)
        for candidate_path in (d / 'evaluations').glob('independent_small_*.json'):
            candidate, error = self.read(candidate_path)
            if error or candidate.get('teacher_id') != ident:
                continue
            cp_path = pick(candidate, 'checkpoint.path')
            if not isinstance(cp_path, str) or d not in Path(cp_path).resolve().parents:
                continue
            if evaluation_rank(candidate) > evaluation_rank(evaluation):
                evaluation = candidate
                errors = [e for e in errors if not e.startswith('latest_evaluation:')]
                trace_path = d / 'videos' / ('trace_small_%06d.json' % candidate['iteration'])
                if trace_path.is_file():
                    video = dict(type='kinematic_trace', path=str(trace_path),
                                 iteration=candidate['iteration'], checkpoint_path=cp_path,
                                 checkpoint_sha256=candidate['checkpoint'].get('sha256'),
                                 created_unix=candidate.get('created_unix'),
                                 source='Independent evaluation selected by newest checkpoint')
        if final and ident in final['teachers']:
            evaluation = final['teachers'][ident]['evaluation']
            video = final['teachers'][ident]['video']
            errors = [e for e in errors if not e.startswith(('latest_evaluation:', 'video_receipt:'))]
        state = pick(status, 'status', 'state') or pick(job, 'status', 'state') or 'pending'
        if isinstance(state, dict):
            state = pick(state, 'state', 'status') or 'pending'
        metrics = pick(status, 'metrics') or {}
        self.record_history(root, ident, status)
        checkpoint_iteration = pick(cp, 'iteration', 'update', 'step')
        iteration = pick(status, 'iteration', 'update', 'iterations')
        if iteration is None:
            iteration = checkpoint_iteration
        heartbeat = pick(status, 'updated_unix', 'updated_at_unix', 'heartbeat_unix')
        if not isinstance(heartbeat, (int, float)):
            try:
                heartbeat = (d / 'status.json').stat().st_mtime
            except OSError:
                heartbeat = None
        pid = pick(status, 'pid', 'trainer_pid') or pick(job, 'pid')
        pid_alive = Path('/proc/' + str(pid)).exists() if isinstance(pid, int) and pid > 0 else None
        age = time.time() - heartbeat if heartbeat else None
        running = str(state).lower() in ('running', 'training', 'started', 'evaluating', 'launching')
        stale = running and ((age is not None and age > 90) or pid_alive is False)
        if stale:
            errors.append('Training heartbeat is stale or recorded process has exited')
        evaluation_metrics = pick(evaluation, 'metrics', 'summary') or evaluation
        scenarios = pick(evaluation, 'scenarios') or []
        selected = scenarios[0] if isinstance(scenarios, list) and len(scenarios) == 1 else {}
        command = pick(evaluation, 'mean_command', 'command', 'command_xyz', 'target_velocity', 'summary.mean_command')
        command = command or pick(selected, 'mean_command', 'initial_command')
        actual = pick(evaluation, 'mean_actual_xyz_command_units', 'mean_actual', 'actual_velocity', 'summary.mean_actual')
        actual = actual or pick(selected, 'mean_actual_xyz_command_units', 'mean_actual')
        training_command = pick(metrics, 'target_command', 'mean_command', 'command_xyz', 'command') or pick(status, 'target_command', 'command') or pick(job, 'target_command', 'command')
        training_actual = pick(metrics, 'actual_velocity', 'mean_actual', 'mean_actual_xyz_command_units', 'actual_command_units')
        survival = pick(evaluation_metrics, 'survival_fraction', 'survival', 'survival_rate')
        if survival is None:
            survival = pick(selected, 'survival_fraction')
        training_survival = pick(metrics, 'completed_episode_survival', 'survival_fraction')
        video_text = pick(video, 'path', 'video_path', 'mp4_path', 'video', 'file') or pick(evaluation, 'video_receipt.path', 'video_receipt.video_path')
        media = self.allowed_media(video_text, d)
        trace = self.allowed_media(video_text, d, ('.json',)) if video.get('type') == 'kinematic_trace' else None
        evaluation_time = pick(evaluation, 'created_unix', 'updated_unix', 'evaluated_unix')
        if not evaluation_time and (d / 'latest_evaluation.json').exists():
            evaluation_time = (d / 'latest_evaluation.json').stat().st_mtime
        bank = bank or {}
        bank_start = pick(bank, 'started_unix', 'start_unix')
        evaluation_source = pick(evaluation, 'inherited_from_bank', 'source_bank', 'source_run')
        evaluation_inherited = bool(evaluation.get('inherited', False))
        if isinstance(evaluation_source, str):
            source_path = Path(evaluation_source).resolve()
            evaluation_inherited = evaluation_inherited or (source_path != root.resolve() and root.resolve() not in source_path.parents)
        if isinstance(evaluation_time, (int, float)) and isinstance(bank_start, (int, float)):
            evaluation_inherited = evaluation_inherited or evaluation_time < bank_start - 1
        video_time = pick(video, 'created_unix', 'updated_unix')
        video_inherited = bool(video.get('inherited', False) or pick(video, 'inherited_from_bank', 'source_bank', 'source_run'))
        if isinstance(video_time, (int, float)) and isinstance(bank_start, (int, float)):
            video_inherited = video_inherited or video_time < bank_start - 1
        ref = pick(job, 'reference_quality', 'quality') or pick(data['config'], 'reference_quality', 'quality') or {}
        if not ref:
            reference = pick(job, 'reference_metadata', 'metadata_file') or pick(data['config'], 'reference_metadata')
            if isinstance(reference, str):
                p = Path(reference)
                if not p.is_absolute():
                    p = self.project / p
                if self.project in p.resolve().parents:
                    reference_data, _ = self.read(p)
                    ref = pick(reference_data, 'quality') or {}
        quality = ref
        source_title = pick(job, 'title', 'label', 'motion_title') or ident
        titles = {'02_STRAFE_LEFT': '\u5de6\u6a2a\u79fb pace', '03_STRAFE_RIGHT': '\u53f3\u6a2a\u79fb pace', '01_FORWARD_POLICY': '\u524d\u8fdb walk', '04_TURN_LEFT': '\u5de6\u8f6c', '05_TURN_RIGHT': '\u53f3\u8f6c',
                  '06_BACKWARD': '\u540e\u9000 trot', '07_LATERAL_TROT_LEFT': '\u5de6\u6a2a\u79fb trot',
                  '08_LATERAL_TROT_RIGHT': '\u53f3\u6a2a\u79fb trot', '09_BACKWARD_WALK': '\u540e\u9000 walk', '10_FORWARD_TROT': '\u524d\u8fdb trot'}
        return finite_json({'id': ident, 'title': titles.get(ident, source_title), 'source_title': source_title,
                            'run_directory': str(d), 'bank_directory': str(root), 'job': job, 'state': str(state), 'iteration': iteration,
                            'checkpoint': cp, 'checkpoint_iteration': checkpoint_iteration,
                            'metrics': metrics, 'status': status, 'evaluation': evaluation,
                            'evaluation_iteration': pick(evaluation, 'iteration', 'checkpoint.iteration'),
                            'evaluation_inherited': evaluation_inherited,
                            'evaluation_source_bank': evaluation_source,
                            'evaluation_age_s': time.time() - evaluation_time if isinstance(evaluation_time, (int, float)) else None,
                            'command': command, 'actual': actual, 'training_command': training_command,
                            'training_actual': training_actual, 'survival_fraction': survival,
                            'training_survival_fraction': training_survival,
                            'reference_quality': quality, 'heartbeat_age_s': age, 'stale': stale,
                            'pid': pid, 'pid_alive': pid_alive, 'errors': errors,
                            'video': {'receipt': video, 'available': media is not None or trace is not None,
                                      'inherited': video_inherited,
                                      'mode': 'trace' if trace else 'media',
                                      'url': '/api/trajectory?id=' + quote(ident) + '&t=' + str(trace.stat().st_mtime) + '&final=' + str(video.get('final_manifest_sha256', '')) if trace else '/media?path=' + quote(str(media)) + '&t=' + str(media.stat().st_mtime) if media else None},
                            'completion': data['completion'],
                            'adaptive_curriculum': data['adaptive_curriculum'],
                            'reward_breakdown': data['reward_breakdown'],
                            'phase_sampling': data['phase_sampling'],
                            'direction_metrics': data['direction_metrics'],
                            'teacher_config': data['config']})

    def detail(self, ident):
        snapshot = self.snapshot()
        teacher = next((t for t in snapshot['teachers'] if t['id'] == ident), None)
        if teacher is None:
            return None
        if not teacher.get('run_directory'):
            return {'teacher': teacher, 'history': [], 'logs': []}
        root = Path(teacher['run_directory'])
        history = []
        history_source = 'trainer_metrics_jsonl'
        try:
            lines = (root / 'metrics.jsonl').read_text(encoding='utf8').splitlines()[-600:]
            for line in lines:
                try:
                    v = json.loads(line)
                    history.append({'iteration': pick(v, 'iteration', 'update', 'step'),
                                    'reward': pick(v, 'mean_reward_per_control_step', 'mean_reward', 'reward_mean', 'reward', 'metrics.mean_reward_per_control_step', 'metrics.reward_mean', 'metrics.reward'),
                                    'survival': pick(v, 'completed_episode_survival', 'survival_fraction', 'metrics.completed_episode_survival'),
                                    'velocity_rmse': pick(v, 'velocity_rmse_m_s', 'metrics.velocity_rmse_m_s')})
                except ValueError:
                    pass
        except OSError:
            pass
        if not history:
            bank = Path(teacher['bank_directory'])
            key = (re.sub(r'[^A-Za-z0-9_.-]', '_', bank.name) + '-' + hashlib.sha256(str(bank).encode()).hexdigest()[:10], re.sub(r'[^A-Za-z0-9_.-]', '_', ident))
            with self.history_lock:
                history = list(self.histories.get(key, []))
            history_source = 'dashboard_observed_status_no_backfill'
        logs = []
        for name in ['train.log', 'trainer.log', 'monitor.log', 'error.log']:
            try:
                with (root / name).open('rb') as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 10000))
                    logs.append({'name': name, 'tail': f.read().decode('utf8', errors='replace').splitlines()[-45:]})
            except OSError:
                pass
        return finite_json({'teacher': teacher, 'history': history, 'history_source': history_source, 'logs': logs})

    def trajectory(self, ident):
        snapshot = self.snapshot()
        teacher = next((t for t in snapshot['teachers'] if t['id'] == ident), None)
        if not teacher or teacher.get('video', {}).get('mode') != 'trace':
            return None
        receipt = teacher['video']['receipt']
        path = self.allowed_media(pick(receipt, 'path', 'video_path', 'file'), Path(teacher['run_directory']), ('.json',))
        if not path or path.stat().st_size > 25 * 1024 * 1024:
            return None
        data, error = self.read(path)
        return {'error': error} if error else data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith('GET /api/'):
            return
        super().log_message(fmt, *args)

    def json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path)
        params = parse_qs(route.query)
        if route.path == '/api/bank':
            return self.json(self.server.store.snapshot())
        if route.path == '/api/teacher':
            value = self.server.store.detail(params.get('id', [''])[0])
            return self.json(value or {'error': 'Teacher not found'}, 200 if value else 404)
        if route.path == '/api/night':
            value,_=self.server.store.read(self.server.store.project/'deployment/night_20261029/monitor_status.json');return self.json(value)
        if route.path == '/api/health':
            return self.json({'ok': True, 'service': 'sable-teacher-bank-dashboard', 'time_unix': time.time()})
        if route.path == '/api/trajectory':
            data = self.server.store.trajectory(params.get('id', [''])[0])
            return self.json(data or {'error': 'Trajectory not available'}, 200 if data else 404)
        if route.path == '/media':
            p = self.server.store.allowed_media(params.get('path', [''])[0], self.server.store.project)
            return self.file(p, range_enabled=True) if p else self.send_error(404)
        static = {'/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css', '/model.json': 'model.json', '/trace.js': 'trace.js', '/diagnostics.js': 'diagnostics.js'}
        if route.path in static:
            return self.file(HERE / static[route.path])
        self.send_error(404)

    def file(self, path, range_enabled=False):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        value = self.headers.get('Range') if range_enabled else None
        if value:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
            if not match or not any(match.groups()):
                return self.send_error(416)
            a, b = match.groups()
            if not a:
                start = max(0, size - int(b))
            else:
                start = int(a); end = min(size - 1, int(b)) if b else size - 1
            if start > end or start >= size:
                return self.send_error(416)
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'no-store' if not range_enabled else 'private,max-age=60')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if range_enabled:
            self.send_header('Accept-Ranges', 'bytes')
        if status == 206:
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
        self.end_headers()
        try:
            with path.open('rb') as f:
                f.seek(start); remaining = end - start + 1
                while remaining:
                    block = f.read(min(1024 * 1024, remaining))
                    if not block: break
                    self.wfile.write(block); remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=str(DEFAULT_PROJECT))
    parser.add_argument('--bank')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=11111)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.store = Store(args.project, args.bank)
    server.store.start_collector()
    print(json.dumps({'service': 'sable-teacher-bank-dashboard', 'port': args.port, 'project': args.project}), flush=True)
    server.serve_forever()
