"""Measure the active command curriculum without changing or restarting training.

Uses the run's frozen simulator/probe and matches every gate to its full endpoint
report. Outputs live outside the frozen run. Passing is progression evidence only.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from evaluate_frontier import assess_frontier
from run_io import atomic_json, read_json, sha256

ROOT = Path('/root/gpufree-data/projects/sable-isaacgym')
TERMINAL = {'stopped_at_deadline', 'iterations_complete'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--interval', type=int, default=360)
    p.add_argument('--minimum-iteration', type=int, default=10)
    a = p.parse_args()
    run = a.run.resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run/'config.json')
    assert config['stage'] == 'S3'
    alpha = config['command_frontier']['training_alpha']
    assert 0 < alpha < 1
    alphas = [alpha, min(1., alpha+.125)]
    next_due = 0.
    seen = set()
    terminal_since = None
    while True:
        state = read_json(run/'status.json', {})
        if state.get('status') in {'stopped_requested', 'failed'} or (run/'STOP').exists():
            atomic_json(a.output_dir/'status.json', {'status': 'cancelled', 'reason': 'Parent stopped or failed'})
            break
        if read_json(ROOT/'results/cmd16_hist6/current_migration.json', {}).get('run') != str(run):
            atomic_json(a.output_dir/'status.json', {'status': 'cancelled', 'reason': 'Active run changed'})
            break
        final = state.get('status') in TERMINAL
        if final:
            if terminal_since is None:terminal_since = time.time()
            if time.time()-terminal_since > 660:
                atomic_json(a.output_dir/'status.json', {'status': 'failed', 'reason': 'No usable final endpoint report within660s'})
                break
        final_cp = read_json(run/'checkpoint.json', {})
        candidates = []
        for path in (run/'evaluations').glob('independent_*.json'):
            report = read_json(path, {})
            if report.get('iteration', 0) < a.minimum_iteration or report.get('checkpoint_sha256') in seen:
                continue
            if final and report.get('checkpoint_sha256') != final_cp.get('sha256'):
                continue
            candidates.append((report['iteration'], path, report))
        if final and final_cp.get('sha256') in seen:
            break
        if not candidates or (not final and time.time() < next_due):
            time.sleep(5)
            continue
        _, endpoint_path, endpoint = max(candidates, key=lambda x: x[0])
        # Avoid adding another simulation while the periodic endpoint video is
        # rendering. The existing 180-second reporting cadence stays unchanged.
        monitor = read_json(ROOT/'results/cmd16_hist6'/('watch_'+run.name+'.json'), {})
        if monitor.get('status') == 'evaluating':
            time.sleep(5)
            continue
        cp = Path(endpoint['checkpoint'])
        assert cp.parent == run/'checkpoints'
        assert sha256(cp) == endpoint['checkpoint_sha256']
        destination = a.output_dir/('checkpoint_%06d' % endpoint['iteration'])
        destination.mkdir(exist_ok=True)
        output = destination/'frontier.json'
        command = [sys.executable, '-u', str(run/'runtime/probe_command_frontier.py'),
            '--checkpoint', str(cp), '--teacher', str(run/'teacher_manifest.json'),
            '--output', str(output), '--seconds', '16', '--alphas'] + [str(x) for x in alphas] + ['--full-families']
        started = time.time()
        marker = dict(status='evaluating', run=str(run), iteration=endpoint['iteration'],
            checkpoint_sha256=endpoint['checkpoint_sha256'], training_alpha=alpha,
            command=command, started_unix=started, final=final)
        atomic_json(a.output_dir/'status.json', marker)
        try:
            with (destination/'probe.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=360)
            report = read_json(output)
            baseline_path = run/'baseline_metrics.json'
            baseline = read_json(baseline_path)
            gates = []
            for value in alphas:
                gate = assess_frontier(report, endpoint, baseline, value)
                gate['sources'] = {name: dict(path=str(path.resolve()), sha256=sha256(path))
                    for name, path in [('frontier', output), ('endpoint', endpoint_path), ('baseline', baseline_path)]}
                gate_path = destination/('gate_alpha_%g.json' % value)
                atomic_json(gate_path, gate)
                gates.append(dict(alpha=value, passed=gate['passed'], failures=gate['failures'], path=str(gate_path)))
            marker.update(status='complete', finished_unix=time.time(), gates=gates, full_S3_accepted=False)
            atomic_json(destination/'receipt.json', marker)
            atomic_json(a.output_dir/'status.json', marker)
            if gates[0]['passed']:
                atomic_json(a.output_dir/'latest_passed.json', marker)
        except Exception as error:
            marker.update(status='failed', finished_unix=time.time(), error=type(error).__name__+': '+str(error))
            atomic_json(destination/'receipt.json', marker)
            atomic_json(a.output_dir/'status.json', marker)
        seen.add(endpoint['checkpoint_sha256'])
        if final:
            break
        next_due = max(started+a.interval, time.time()+5)


if __name__ == '__main__':
    main()
