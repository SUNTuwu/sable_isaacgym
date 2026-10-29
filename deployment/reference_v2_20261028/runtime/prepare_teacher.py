"""Wait for v44's real processes, then seal and verify the designated teacher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

ROOT = Path('/root/gpufree-data/projects/sable-isaacgym')
RUN = ROOT / 'runs/stand5_20261027_v44_stride80_all'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    state = read(RUN / 'status.json')
    while process_exists(state['pid']):
        state = read(RUN / 'status.json')
        print(json.dumps({'waiting_for_pid': state['pid'], 'iteration': state['iteration'],
                          'status': state['status'], 'time': time.time()}), flush=True)
        time.sleep(20)
    # Final evaluation and archive belong to the supervisor after PPO exits.
    until = time.time() + 900
    watch = ROOT / 'results/legged_gym/stride80_all_watch_status.json'
    while time.time() < until:
        w = read(watch)
        if w.get('finished'):
            if not w.get('archive_verified'):
                raise RuntimeError('v44 final archive audit failed')
            break
        print('Waiting for v44 final independent evaluation and archive verification', flush=True)
        time.sleep(15)
    else:
        raise TimeoutError('Finalization not complete; no teacher selected')
    state = read(RUN / 'status.json')
    completion, receipt = read(RUN / 'completion.json'), read(RUN / 'backup_receipt.json')
    checkpoint, evaluation = read(RUN / 'checkpoint.json'), read(RUN / 'latest_evaluation.json')
    assert completion['training_exit_code'] == 0
    assert checkpoint['iteration'] == evaluation['iteration']
    assert digest(checkpoint['path']) == checkpoint['sha256'] == evaluation['checkpoint_sha256']
    assert digest(receipt['archive']) == receipt['sha256']
    runtime = read(RUN / 'runtime_manifest.json')
    files = [RUN / 'runtime' / n for n in runtime['files']]
    for name, expected in runtime['files'].items():
        assert digest(RUN / 'runtime' / name) == expected
    files += [RUN / n for n in ['config.json', 'runtime_manifest.json', 'checkpoint.json',
                                'completion.json', 'latest_evaluation.json']]
    files += [Path(checkpoint['path'])]
    with tarfile.open(receipt['archive'], 'r:gz') as archive:
        for path in files:
            archived = archive.extractfile(RUN.name + '/' + str(path.relative_to(RUN)))
            assert archived is not None and hashlib.sha256(archived.read()).hexdigest() == digest(path)
    manifest = {'schema': 'sable-frozen-teacher/1', 'completion_verified': True,
                'selection': 'User-designated final v44 checkpoint, not best-checkpoint selection',
                'run': str(RUN), 'checkpoint': checkpoint['path'], 'sha256': checkpoint['sha256'],
                'iteration': checkpoint['iteration'], 'archive': receipt,
                'created_unix': time.time(), 'files': [{'path': str(f), 'sha256': digest(f)} for f in files]}
    (a.output / 'teacher_manifest.json').write_text(json.dumps(manifest, indent=2))
    groups = evaluation['velocity_tracking']['groups']
    count = sum(g['samples'] for g in groups)
    vx_rmse = (sum(g['samples'] * g['rmse_velocity_xyz_m_s'][0] ** 2 for g in groups) / count) ** .5
    xy_rmse = (sum(g['samples'] * sum(x*x for x in g['rmse_velocity_xyz_m_s'][:2]) for g in groups) / count) ** .5
    baseline = {'teacher_iteration': checkpoint['iteration'], 'teacher_sha256': checkpoint['sha256'],
                'source_evaluation': str(RUN / 'latest_evaluation.json'),
                'survival_fraction': evaluation['summary']['survival_16s'],
                'velocity_rmse_m_s': vx_rmse, 'xy_velocity_rmse_m_s': xy_rmse,
                'summary': evaluation['summary'], 'velocity_groups': groups,
                'scope': 'Final teacher24env16s old-domain baseline; no student success claim',
                'regression_limits': {'survival_fraction_min': .9,
                    'forward_rmse_max_m_s': max(vx_rmse * 1.1, vx_rmse + .003),
                    'xy_rmse_max_m_s': max(xy_rmse * 1.1, xy_rmse + .003)},
                'new_capability_limits': {'xy_rmse_m_s': .035, 'yaw_rmse_rad_s': .30,
                    'mean_command_gain_min': .5, 'frequency_error_hz': .10,
                    'frequency_switch_difference_min_hz': .25,
                    'evaluation_seeds': [947, 1147, 1347], 'environments_per_seed': 32}}
    (a.output / 'baseline_metrics.json').write_text(json.dumps(baseline, indent=2))
    print(json.dumps({'teacher_ready': True, 'iteration': checkpoint['iteration'],
                      'manifest': str(a.output / 'teacher_manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
