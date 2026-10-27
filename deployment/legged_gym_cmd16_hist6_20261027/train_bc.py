"""Deadline-bounded full-target behavior cloning with independent-environment validation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time
import torch
from history_policy import HistoryActorCritic
from observation_schema import mirror_raw, mirror_proprio, metadata as observation_metadata
from command_schema import CommandSpec, SCHEMA_VERSION
from action_schema import ActionSpec
from run_io import atomic_json, append_json, sha256

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def main():
    from training_hold import require_training_enabled
    require_training_enabled()
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    deadline = datetime.datetime.fromisoformat(a.deadline).timestamp()
    if time.time() >= deadline:
        raise ValueError('Expired deadline')
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    a.run.mkdir(parents=True, exist_ok=True)
    (a.run / 'checkpoints').mkdir(exist_ok=True)
    dataset = json.loads((a.dataset / 'dataset.json').read_text())
    torch.manual_seed(42)
    parts = []
    for shard in dataset['shards']:
        assert sha256(shard['path']) == shard['sha256']
        parts.append(torch.load(shard['path'], map_location='cpu'))
    data = {k: torch.cat([part[k] for part in parts]).to('cuda:0') for k in parts[0]}
    del parts
    row = torch.arange(len(data['raw']), device='cuda:0')
    valid_mask = (row % dataset['num_envs']) % 8 == 0
    validation = row[valid_mask]
    training = row[~valid_mask]
    policy = HistoryActorCritic().to('cuda:0')
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    iteration = 0
    if a.resume:
        cp = torch.load(str(a.resume), map_location='cuda:0')
        policy.load_state_dict(cp['model_state_dict'], strict=True)
        iteration = cp['iter']
    teacher_manifest = json.loads(Path(dataset['teacher_manifest']).read_text())
    teacher = torch.load(teacher_manifest['checkpoint'], map_location='cpu')
    atomic_json(a.run / 'parameter_mapping.json', policy.initialization_report(teacher['model_state_dict']))
    del teacher
    schema = {'schema': SCHEMA_VERSION, 'commands': CommandSpec.metadata(), 'actions': ActionSpec.metadata(),
              'observations': observation_metadata(), 'estimator_auxiliary': 'ROOT velocity MSE plus0.1 next clean proprio prediction MSE; reset pairs masked',
              'weight_transfer': 'explicitly none; full target behavior cloning', 'waist_frozen': True}
    atomic_json(a.run / 'schema.json', schema)
    atomic_json(a.run / 'config.json', {'stage': 'S1', 'teacher': teacher_manifest['checkpoint'],
        'teacher_sha256': teacher_manifest['sha256'], 'dataset': str(a.dataset), 'deadline': a.deadline,
        'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128],
        'learning_rate': 3e-4, 'batch_size': 2048, 'seed': 42, 'validation': 'every8th environment excluded from optimization'})
    status = {'status': 'training', 'stage': 'S1_behavior_cloning', 'pid': os.getpid(),
              'started_unix': time.time(), 'deadline': a.deadline, 'iteration': iteration,
              'teacher_iteration': teacher_manifest['iteration'], 'accepted_locomotion': False}
    atomic_json(a.run / 'status.json', status)

    def save(reason, metrics):
        cp = a.run / 'checkpoints' / ('model_%06d.pt' % iteration)
        if not cp.exists():
            partial = cp.with_suffix('.partial')
            torch.save({'model_state_dict': policy.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'iter': iteration, 'schema': schema, 'metrics': metrics,
                        'teacher_sha256': teacher_manifest['sha256'], 'reason': reason}, str(partial))
            partial.replace(cp)
        atomic_json(a.run / 'checkpoint.json', {'path': str(cp.resolve()), 'sha256': sha256(cp),
            'iteration': iteration, 'reason': reason, 'updated_unix': time.time()})
        return cp

    def validate():
        policy.eval()
        squares = torch.zeros(15, device='cuda:0')
        count = 0
        with torch.no_grad():
            for ids in validation.split(4096):
                diff = (policy.act_inference(data['raw'][ids]) - data['target'][ids]) * .35
                squares += diff.square().sum(0)
                count += len(ids)
        policy.train()
        return {'target_rmse_rad': float((squares[:14].sum() / (count * 14)).sqrt()),
                'target_rmse_rad_per_joint': (squares / count).sqrt().tolist(),
                'validation_samples': count, 'closed_loop_validated': False}

    last_report = 0.
    last_save = 0.
    metrics = {}
    save('initial', metrics)
    try:
        while time.time() < deadline and not STOP and not (a.run / 'STOP').exists():
            ids = training[torch.randint(len(training), (2048,), device='cuda:0')]
            raw = data['raw'][ids]
            labels = data['target'][ids]
            predictions = policy.act_inference(raw)
            bc = (predictions[:, :14] - labels[:, :14]).square().mean()
            aux, stats = policy.history_encoder.auxiliary_loss(raw[:, :306], data['velocity'][ids],
                data['next_proprio'][ids], data['done'][ids])
            mirrored = mirror_raw(raw)
            reflected_velocity = data['velocity'][ids] * raw.new_tensor([1., -1., 1.])
            aux_m, _ = policy.history_encoder.auxiliary_loss(mirrored[:, :306], reflected_velocity,
                mirror_proprio(data['next_proprio'][ids]), data['done'][ids])
            loss = bc + .5 * (aux + aux_m)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            iteration += 1
            if time.time() - last_report >= 30:
                metrics = validate()
                metrics.update(loss=float(loss), bc_loss=float(bc), auxiliary_velocity_mse=float(stats['velocity_mse']),
                               auxiliary_next_proprio_mse=float(stats['next_proprio_mse']))
                if not all(torch.isfinite(x).all() for x in policy.parameters()):
                    raise RuntimeError('Nonfinite student parameters')
                status.update(iteration=iteration, metrics=metrics, updated_unix=time.time())
                atomic_json(a.run / 'status.json', status)
                append_json(a.run / 'metrics.jsonl', dict(iteration=iteration, time=time.time(), **metrics))
                print(json.dumps(status), flush=True)
                last_report = time.time()
            if time.time() - last_save >= 180:
                save('periodic', metrics)
                last_save = time.time()
        status['status'] = 'stopped_at_deadline' if time.time() >= deadline else 'stopped_requested'
    except BaseException as error:
        status.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        save('final', metrics)
        status.update(iteration=iteration, updated_unix=time.time())
        atomic_json(a.run / 'status.json', status)


if __name__ == '__main__':
    main()
