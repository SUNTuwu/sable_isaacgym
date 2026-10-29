"""Wait until 08:30, gracefully save the live teachers, then evaluate without PPO.

Stdlib only; importing does nothing. This coordinator never sends SIGKILL,
restarts a trainer, writes the current-bank pointer, or shuts down the server.
The existing yaw watchdog owns its final GPU evaluation; this script waits for it.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

P = Path('/root/gpufree-data/projects/sable-isaacgym')
V = Path('/root/gpufree-data/envs/sable-isaacgym-py38')
END = datetime.datetime(2026, 10, 29, 8, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
IDS = ('01_FORWARD_POLICY', '04_TURN_LEFT', '05_TURN_RIGHT', '06_BACKWARD',
       '02_STRAFE_LEFT', '03_STRAFE_RIGHT', '09_BACKWARD_WALK', '10_FORWARD_TROT')
TERMINAL = ('completed', 'stopped_at_deadline', 'stopped_requested')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.finalize.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def inside(path, root):
    return Path(root).resolve() in Path(path).resolve().parents


class Host:
    """Injectable clock/process boundary for CPU-only tests."""
    def now(self): return time.time()
    def sleep(self, seconds): time.sleep(seconds)
    def identify(self, pid):
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            raise ValueError('Invalid process PID')
        try:
            root = Path('/proc') / str(pid)
            fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
            args = [x.decode(errors='replace') for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
            if fields[0] == 'Z' or not args: return None
            return dict(pid=pid, start_ticks=fields[19], args=args)
        except (FileNotFoundError, ProcessLookupError): return None
    def signal(self, pid): os.kill(pid, signal.SIGTERM)
    def gpu_processes(self, project):
        found = []
        scripts = {'teacher_bank_core.py', 'yaw_sine_teacher.py', 'evaluate_yaw.py', 'final_bank_evaluation.py'}
        for item in Path('/proc').iterdir():
            if not item.name.isdigit() or int(item.name) <= 1: continue
            proc = self.identify(int(item.name))
            if not proc: continue
            if any(Path(arg).name in scripts and inside(arg, project) for arg in proc['args']): found.append(proc)
        return found


def same_process(host, saved):
    return bool(saved and host.identify(saved['pid']) == saved)


def checked_trainer(host, receipt, run, script):
    """Full argv equality, script path, and option binding, not substring matching."""
    command = receipt.get('command')
    flag = '--bank' if script == 'teacher_bank_core.py' else '--run'
    if (not isinstance(command, list) or len(command) < 4
            or Path(command[1]).resolve() != run / 'runtime' / script
            or command.count(flag) != 1 or command.index(flag) + 1 >= len(command)
            or Path(command[command.index(flag) + 1]).resolve() != run):
        raise RuntimeError('Launch command does not bind the exact run: ' + str(run))
    current = host.identify(receipt['pid'])
    if current and current['args'] != command:
        raise RuntimeError('Live PID argv differs from the launch receipt: ' + str(receipt['pid']))
    expected_ticks = receipt.get('start_ticks') or receipt.get('process_start_ticks')
    if current and expected_ticks is not None and str(expected_ticks) != current['start_ticks']:
        raise RuntimeError('Trainer start_ticks differs from the launch receipt')
    return current


def checked_watch(host, receipt, run, project):
    current = host.identify(receipt['watchdog_pid'])
    if current:
        args = current['args']
        valid_scripts = {run / 'runtime/watch.py', project / 'deployment/yaw_sine_20261029/watch.py'}
        if (len(args) != 4 or Path(args[1]).resolve() not in valid_scripts
                or Path(args[2]).resolve() != run or args[3] != str(receipt['pid'])):
            raise RuntimeError('Yaw watchdog PID has the wrong exact run/target scope')
    return current


def signal_after_deadline(host, saved, deadline):
    if host.now() < deadline: raise RuntimeError('Refusing to signal before the fixed deadline')
    if not saved: return 'already_exited'
    current = host.identify(saved['pid'])
    if current is None: return 'already_exited'
    if current != saved: raise RuntimeError('Process identity changed before SIGTERM')
    try: host.signal(saved['pid'])
    except ProcessLookupError: return 'already_exited'
    return 'sigterm_sent'


def wait_trainers(host, saved, timeout=180):
    until = host.now() + timeout
    while any(same_process(host, proc) for proc in saved):
        if host.now() >= until: raise TimeoutError('Trainers did not finish their graceful save within 180 seconds')
        host.sleep(min(1, until - host.now()))


def checkpoints(run, bank, stopped_after=None):
    status = read(run / ('teacher_bank.json' if bank else 'status.json'))
    if status.get('status') not in TERMINAL: raise RuntimeError('Source did not finish successfully: ' + str(run))
    result = []
    for ident in IDS if bank else ('yaw',):
        folder = run / ident if bank else run
        receipt = read(folder / 'checkpoint.json'); model = Path(receipt['path']).resolve()
        if not inside(model, folder) or not model.is_file() or receipt.get('reason') != 'final':
            raise RuntimeError('Missing or invalid final checkpoint: ' + ident)
        digest = sha(model)
        if (bank or 'sha256' in receipt) and digest != receipt.get('sha256'):
            raise RuntimeError('Final checkpoint SHA mismatch: ' + ident)
        if stopped_after is not None and (model.stat().st_mtime < stopped_after
                or (folder / 'checkpoint.json').stat().st_mtime < stopped_after
                or (bank and receipt.get('updated_unix', 0) < stopped_after)):
            raise RuntimeError('Final checkpoint predates the graceful stop request: ' + ident)
        result.append(dict(teacher_id=ident, path=str(model), iteration=receipt['iteration'], sha256=digest))
    if len({item['iteration'] for item in result}) != 1: raise RuntimeError('Mixed final bank checkpoint iterations')
    return result


def verify_yaw_completion(run, checkpoint):
    completion = read(run / 'completion.json')
    if completion.get('final_evaluation_exit_code') != 0: raise RuntimeError('Existing yaw final evaluation failed')
    report = read(run / 'evaluation.json')
    if (report.get('final') is not True or report.get('checkpoint_iteration') != checkpoint['iteration']
            or Path(report.get('checkpoint', '')).resolve() != Path(checkpoint['path']).resolve()
            or sha(checkpoint['path']) != checkpoint['sha256']):
        raise RuntimeError('Yaw final evaluation does not identify the exact final checkpoint')
    if completion.get('finished_unix', 0) < Path(checkpoint['path']).stat().st_mtime:
        raise RuntimeError('Stale yaw completion receipt')
    return dict(completion=completion, report_path=str(run / 'evaluation.json'), report_sha256=sha(run / 'evaluation.json'),
                checkpoint=checkpoint, evaluation_owner='existing_yaw_watchdog', envs=16,
                accepted=report.get('accepted'), acceptance_not_required_for_completed_evaluation=True)


def wait_yaw(host, run, watchdog, checkpoint, timeout=960):
    until = host.now() + timeout
    while True:
        alive = same_process(host, watchdog)
        if not alive:
            if not (run / 'completion.json').is_file():
                raise RuntimeError('Yaw watchdog exited without a final evaluation completion receipt')
            return verify_yaw_completion(run, checkpoint)
        if host.now() >= until:
            raise TimeoutError('Existing yaw watchdog/final evaluation did not complete within 960 seconds; no second GPU evaluator started')
        host.sleep(min(2, until - host.now()))


def verify_plan(plan, bank, output, expected):
    if (plan.get('schema') != 'standalone-final-bank-evaluation/v2' or plan.get('mode') != 'evaluation_only'
            or Path(plan.get('source_bank', '')).resolve() != bank or Path(plan.get('output', '')).resolve() != output
            or plan.get('source_process_alive') is not False or plan.get('source_status') not in TERMINAL
            or plan.get('optimization_steps') != 0 or plan.get('training_envs') != 0
            or plan.get('total_envs') != 32 or plan.get('envs_per_teacher') != 4 or plan.get('horizon_s') != 24):
        raise RuntimeError('Final evaluation plan violates the isolated 32-environment/24-second contract')
    actual = [{key: row.get(key) for key in ('teacher_id', 'path', 'iteration', 'sha256')}
              for row in plan.get('selected_checkpoints', [])]
    if actual != expected: raise RuntimeError('Plan selected something other than the exact final checkpoints')


def verify_completed(manifest, bank, output, expected):
    verify_plan(manifest, bank, output, expected)
    if (manifest.get('status') != 'completed' or manifest.get('actual_simulation_steps') != 1200
            or manifest.get('policy_and_optimizer_unchanged') is not True):
        raise RuntimeError('Final evaluation did not complete unchanged-policy, unchanged-Adam verification')
    for checkpoint in expected:
        if sha(checkpoint['path']) != checkpoint['sha256']: raise RuntimeError('Source checkpoint changed after evaluation')
        report = read(output / checkpoint['teacher_id'] / 'latest_evaluation.json')
        if (report.get('horizon_s') != 24 or report.get('mode') != 'final_evaluation_only'
                or report.get('policy_and_optimizer_unchanged') is not True
                or any(report.get('source_checkpoint', {}).get(k) != v for k, v in checkpoint.items())):
            raise RuntimeError('Teacher final report does not match the exact source checkpoint')


class Finalizer:
    def __init__(self, project=P, python=V / 'bin/python', host=None):
        self.project = Path(project).resolve(); self.python = str(python); self.host = host or Host()
        self.deployment = self.project / 'deployment/night_20261029'
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_%s' % os.getpid()
        self.directory = self.project / 'results/final_evaluation' / ('deadline_' + stamp)
        self.output = self.directory / 'bank_32env_24s'
        self.state = dict(schema='sable-deadline-finalizer/v1', status='running', stage='starting',
                          pid=os.getpid(), deadline_unix=END, created_unix=self.host.now(), signals=[], children=[],
                          output=str(self.output), training_restarts=0, global_current_pointer_write_allowed=False)
        self.lock = None

    def save(self, stage=None, **fields):
        self.state.update(fields)
        if stage: self.state['stage'] = stage
        self.state['updated_unix'] = self.host.now()
        atomic(self.directory / 'execution.json', self.state)
        atomic(self.deployment / 'deadline_finalize_receipt.json', dict(self.state, execution_path=str(self.directory / 'execution.json')))

    def claim(self):
        import fcntl
        self.deployment.mkdir(parents=True, exist_ok=True)
        self.lock = (self.deployment / 'deadline_finalize.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def wait_deadline(self):
        self.save('waiting_for_0830')
        while self.host.now() < END:
            self.host.sleep(min(5, END - self.host.now()))

    def targets(self):
        if self.host.now() < END: raise RuntimeError('Cannot select/stop trainers before deadline')
        pointer = read(self.project / 'results/teacher_bank/current.json')
        bank = Path(pointer['bank']).resolve()
        yaw_receipt = read(self.project / 'deployment/yaw_sine_20261029/launch_receipt.json')
        yaw = Path(yaw_receipt['run']).resolve()
        if bank.parent != self.project / 'runs' or not bank.name.startswith('teacher_bank_fullref_'):
            raise RuntimeError('Current bank is outside the expected project run scope')
        if yaw.parent != self.project / 'runs' or not yaw.name.startswith('yaw_sine_'):
            raise RuntimeError('Current yaw is outside the expected project run scope')
        bank_receipt = read(bank / 'launch_receipt.json')
        if Path(bank_receipt.get('bank', '')).resolve() != bank: raise RuntimeError('Bank receipt points to another run')
        bank_proc = checked_trainer(self.host, bank_receipt, bank, 'teacher_bank_core.py')
        bank_status = read(bank / 'teacher_bank.json')
        if bank_status.get('pid') != bank_receipt['pid']: raise RuntimeError('Bank metadata/receipt PID mismatch')
        guard_path = bank / 'deadline_watchdog.json'
        if bank_proc and guard_path.is_file():
            guard = read(guard_path)
            if guard.get('target_pid') != bank_proc['pid'] or str(guard.get('target_start_ticks')) != bank_proc['start_ticks']:
                raise RuntimeError('Bank watchdog/start_ticks identity mismatch')
        yaw_proc = checked_trainer(self.host, yaw_receipt, yaw, 'yaw_sine_teacher.py')
        watch_proc = checked_watch(self.host, yaw_receipt, yaw, self.project)
        self.save('live_targets_verified', bank=str(bank), yaw=str(yaw), bank_process=bank_proc,
                  yaw_process=yaw_proc, yaw_watchdog=watch_proc, pointer_at_deadline=pointer)
        return bank, yaw, bank_proc, yaw_proc, watch_proc

    def run_child(self, args, label, timeout):
        env = dict(os.environ, SABLE_GYM_ROOT=str(self.project), SABLE_GYM_ENV=str(V), PYTHONUNBUFFERED='1',
                   TORCH_EXTENSIONS_DIR='/root/gpufree-data/cache/sable-isaacgym/torch_extensions')
        env.pop('SABLE_NIGHT_CONTROL_FILE', None)
        env['PATH'] = str(V / 'bin') + ':' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = str(V / 'lib') + ':' + str(V / 'lib/python3.8/site-packages/torch/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
        stdout = self.directory / (label + '.stdout.log'); stderr = self.directory / (label + '.stderr.log')
        with stdout.open('w') as out, stderr.open('w') as err:
            child = subprocess.Popen(args, cwd=str(self.project), env=env, stdout=out, stderr=err, start_new_session=True)
            entry = dict(label=label, pid=child.pid, command=args, started_unix=self.host.now(),
                         process=self.host.identify(child.pid), stdout=str(stdout), stderr=str(stderr))
            self.state['children'].append(entry); self.save(label)
            try:
                result = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Only our own subprocess receives SIGTERM. No SIGKILL and no second evaluator.
                if child.poll() is None: child.terminate()
                entry.update(status='timeout', timeout_s=timeout, terminate_requested_unix=self.host.now())
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: entry['still_running_after_sigterm'] = True
                self.save(label + '_timed_out')
                raise TimeoutError(label + ' timed out; no subsequent GPU stage will start')
            entry.update(status='completed' if result == 0 else 'failed', exit_code=result, finished_unix=self.host.now())
            self.save(label + '_exited')
            if result != 0: raise RuntimeError(label + ' failed, exit ' + str(result))
        return stdout

    def finish_yaw(self, run, watchdog, checkpoint):
        try:
            return dict(status='completed', **wait_yaw(self.host, run, watchdog, checkpoint))
        except (RuntimeError, TimeoutError, OSError, ValueError, KeyError) as error:
            # A failed, fully exited yaw evaluation must not prevent the eight-bank
            # evaluation. A live/unknown evaluator still blocks every GPU stage.
            if same_process(self.host, watchdog): raise
            processes = self.host.gpu_processes(self.project)
            if processes: raise RuntimeError('Yaw final evaluation failed and project GPU processes remain: ' + repr(processes)) from error
            result = dict(status='failed_gpu_released', error=repr(error), watchdog_exited=True,
                          verified_project_gpu_processes=processes, checked_unix=self.host.now(),
                          manual_yaw_evaluation_required=True, automatic_yaw_retry=False)
            self.save('yaw_final_evaluation_failed_gpu_released', yaw_final_evaluation=result)
            return result

    def execute(self):
        self.claim()
        try:
            self.wait_deadline()
            bank, yaw, bank_proc, yaw_proc, watch_proc = self.targets()
            stops = {}
            for name, process in (('bank', bank_proc), ('yaw', yaw_proc)):
                requested = self.host.now()
                event = dict(target=name, process=process, requested_unix=requested, status='pending_identity_recheck')
                self.state['signals'].append(event); self.save('requesting_graceful_stop')
                event['status'] = signal_after_deadline(self.host, process, END)
                if event['status'] == 'sigterm_sent': stops[name] = requested
                self.save('graceful_stop_requested')
            self.save('waiting_for_final_saves')
            wait_trainers(self.host, [bank_proc, yaw_proc])
            bank_cps = checkpoints(bank, True, stops.get('bank'))
            yaw_cp = checkpoints(yaw, False, stops.get('yaw'))[0]
            self.save('waiting_for_existing_yaw_final_evaluation', bank_final_checkpoints=bank_cps, yaw_final_checkpoint=yaw_cp)
            yaw_result = self.finish_yaw(yaw, watch_proc, yaw_cp)
            self.save('yaw_final_evaluation_checked', yaw_final_evaluation=yaw_result)
            if self.host.gpu_processes(self.project): raise RuntimeError('A project trainer/evaluator still runs; refusing overlapping GPU evaluation')
            current = read(self.project / 'results/teacher_bank/current.json')
            if Path(current['bank']).resolve() != bank: raise RuntimeError('Current bank changed after deadline; refusing stale evaluation')
            script = self.deployment / 'final_bank_evaluation.py'
            command = [self.python, str(script), '--source-bank', str(bank), '--output', str(self.output),
                       '--project', str(self.project), '--control', str(self.deployment / 'control.json')]
            plan = read(self.run_child(command + ['--plan'], 'bank_plan', 60))
            verify_plan(plan, bank, self.output, bank_cps)
            atomic(self.directory / 'verified_plan.json', plan)
            if self.host.now() < END or self.host.gpu_processes(self.project): raise RuntimeError('Unsafe concurrent evaluation or early start')
            self.run_child(command, 'bank_evaluation', 900)
            manifest_path = self.output / 'final_evaluation_manifest.json'
            manifest = read(manifest_path); verify_completed(manifest, bank, self.output, bank_cps)
            if self.host.gpu_processes(self.project): raise RuntimeError('Unexpected project GPU process before publication')
            if Path(read(self.project / 'results/teacher_bank/current.json')['bank']).resolve() != bank:
                raise RuntimeError('Current bank changed before final report publication')
            pointer = dict(schema='sable-final-evaluation-pointer/v1', source_bank=str(bank),
                           manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path))
            atomic(self.deployment / 'final_evaluation_pointer.json', pointer)
            outcome = 'completed' if yaw_result['status'] == 'completed' else 'partial_failed_yaw'
            self.save(outcome, status=outcome, bank_final_evaluation_status='completed',
                      final_evaluation_pointer=pointer, finished_unix=self.host.now())
            return self.state
        except BaseException as error:
            self.save(status='failed', failed_stage=self.state['stage'], error=repr(error), finished_unix=self.host.now())
            raise
        finally:
            if self.lock: self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=P)
    parser.add_argument('--python', type=Path, default=V / 'bin/python')
    args = parser.parse_args()
    result = Finalizer(args.project, args.python).execute()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
