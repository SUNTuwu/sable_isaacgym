"""Student-visited-state full-target DAgger. Mixed trajectories are never PPO."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time
from isaacgym import gymapi
import numpy as np
import torch
from migration_env import create_migration_env
from migration_config import student_config
from history_policy import HistoryActorCritic
from observation_schema import mirror_raw, mirror_proprio
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json, append_json, sha256

STOP = False


def stop(*_):
    global STOP
    STOP = True


class Replay:
    def __init__(self, capacity, device):
        self.data = {k: torch.zeros(capacity, *shape, device=device, dtype=dtype) for k, shape, dtype in
            [('raw', (326,), torch.float32), ('target', (15,), torch.float32),
             ('velocity', (3,), torch.float32), ('next_proprio', (51,), torch.float32), ('done', (), torch.bool)]}
        self.capacity, self.size, self.position = capacity, 0, 0

    def append(self, values):
        count = len(values['raw'])
        ids = (torch.arange(count, device=self.data['raw'].device)+self.position) % self.capacity
        for k in self.data:
            self.data[k][ids] = values[k]
        self.position = (self.position+count) % self.capacity
        self.size = min(self.size+count, self.capacity)

    def sample(self, count):
        ids = torch.randint(self.size, (count,), device=self.data['raw'].device)
        return {k:v[ids] for k,v in self.data.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--offline-dataset', type=Path, required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--initial-student-fraction', type=float, default=.25)
    p.add_argument('--max-updates', type=int, default=100000000)
    a = p.parse_args()
    deadline = datetime.datetime.fromisoformat(a.deadline).timestamp()
    assert deadline > time.time()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    teacher = verify_teacher_manifest(a.teacher)
    torch.manual_seed(43)
    np.random.seed(43)
    cfg = student_config(teacher['run'], num_envs=a.num_envs)
    env = create_migration_env(cfg, noise=True)
    env.attach_teacher(teacher['checkpoint'], teacher['sha256'], a.initial_student_fraction)
    policy = HistoryActorCritic().to(env.device)
    parent = torch.load(str(a.resume), map_location=env.device)
    policy.load_state_dict(parent['model_state_dict'], strict=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    offline = Replay(65536, env.device)
    dataset = json.loads((a.offline_dataset/'dataset.json').read_text())
    for shard in dataset['shards']:
        assert sha256(shard['path']) == shard['sha256']
        data = torch.load(shard['path'], map_location='cpu')
        ids = torch.randperm(len(data['raw']))[:2048]
        offline.append({k:v[ids].to(env.device) for k,v in data.items()})
    online = Replay(131072, env.device)
    startup = Replay(16384, env.device)
    a.run.mkdir(parents=True, exist_ok=True)
    (a.run/'checkpoints').mkdir(exist_ok=True)
    atomic_json(a.run/'schema.json', parent['schema'])
    atomic_json(a.run/'config.json', {'stage':'S2_DAgger','deadline':a.deadline,'num_envs':a.num_envs,
        'teacher_sha256':teacher['sha256'],'parent_checkpoint':str(a.resume),'parent_sha256':sha256(a.resume),
        'seed':43,'learning_rate':1e-4,'offline_fraction':.25,'initial_student_fraction':a.initial_student_fraction,
        'actor_hidden_dims':[512,256,128],'critic_hidden_dims':[512,256,128],
        'algorithm':'complete-target supervised DAgger; no PPO or on-policy claim for mixed executions'})
    start = time.time()
    status = {'status':'training','stage':'S2_DAgger','pid':os.getpid(),'started_unix':start,
        'deadline':a.deadline,'teacher_iteration':teacher['iteration'],'accepted_locomotion':False,
        'parent_checkpoint':{'path':str(a.resume),'sha256':sha256(a.resume),'iteration':parent['iter']}}
    iteration, control_steps = 0, 0
    last_report, last_save, last_advance = 0., 0., start
    metrics = {}

    def save(reason):
        cp = a.run/'checkpoints'/('model_%06d.pt'%iteration)
        if not cp.exists():
            part = cp.with_suffix('.partial')
            torch.save({'model_state_dict':policy.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'iter':iteration,'schema':parent['schema'],'teacher_sha256':teacher['sha256'],
                'student_fraction':env.student_fraction,'reason':reason,'metrics':metrics},str(part))
            part.replace(cp)
        atomic_json(a.run/'checkpoint.json',{'path':str(cp.resolve()),'sha256':sha256(cp),
            'iteration':iteration,'reason':reason,'updated_unix':time.time()})

    obs, _ = env.reset()
    env.last_episode_records = []
    save('initial')
    try:
        while time.time() < deadline and not STOP and not (a.run/'STOP').exists() and iteration < a.max_updates:
            policy.eval()
            with torch.no_grad():
                for _ in range(24):
                    before = obs.clone()
                    startup_ids = (env.episode_length_buf < 6).nonzero(as_tuple=False).flatten()
                    velocity = env.base_lin_vel.clone()
                    action = policy.act_inference(obs)
                    obs, _, _, done, info = env.step(action)
                    sample = {'raw':before,'target':env.teacher_target_label.clone(),'velocity':velocity,
                        'next_proprio':info['next_clean_proprio'].clone(),'done':done.clone()}
                    online.append(sample)
                    if len(startup_ids):
                        startup.append({k:v[startup_ids] for k,v in sample.items()})
                    control_steps += 1
            policy.train()
            losses = []
            for _ in range(80):
                if iteration >= a.max_updates:
                    break
                fresh, old = online.sample(1024), offline.sample(512)
                initial = startup.sample(512) if startup.size else online.sample(512)
                batch = {k:torch.cat((fresh[k],old[k],initial[k])) for k in fresh}
                raw = batch['raw']
                prediction = policy.act_inference(raw)
                bc = (prediction[:,:14]-batch['target'][:,:14]).square().mean()
                aux, _ = policy.history_encoder.auxiliary_loss(raw[:,:306],batch['velocity'],batch['next_proprio'],batch['done'])
                mirrored = mirror_raw(raw)
                aux_m, _ = policy.history_encoder.auxiliary_loss(mirrored[:,:306],batch['velocity']*raw.new_tensor([1.,-1.,1.]),mirror_proprio(batch['next_proprio']),batch['done'])
                loss = bc+.5*(aux+aux_m)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
                optimizer.step()
                iteration += 1
                losses.append(float(bc.detach()))
            rows = env.last_episode_records[-512:]
            survival = 1.-float(np.mean([r[1] for r in rows])) if rows else None
            # A blend increase additionally requires actual completed episodes.
            # It does not constitute the independent S2 acceptance gate.
            if len(rows)>=128 and survival>=.9 and time.time()-last_advance>=240 and env.student_fraction<1.:
                env.student_fraction = min(1.,env.student_fraction+.25)
                last_advance = time.time()
                append_json(a.run/'events.jsonl',{'event':'teacher_blend_decreased','student_fraction':env.student_fraction,
                    'completed_episode_survival':survival,'time':last_advance})
                env.last_episode_records=[]
            if time.time()-last_report>=30:
                metrics = {'bc_loss':float(np.mean(losses)),'training_target_rmse_rad':float(np.sqrt(np.mean(losses))*.35),
                    'student_fraction':env.student_fraction,'completed_episode_survival':survival,
                    'recent_completed_episodes':len(rows),'online_samples_in_replay':online.size,
                    'startup_samples_in_replay':startup.size,'control_steps':control_steps,'simulated_environment_seconds':control_steps*env.dt*env.num_envs,
                    'teacher_query_count':env.teacher.query_count,'mixed_trajectory_not_ppo':True}
                status.update(iteration=iteration,metrics=metrics,updated_unix=time.time())
                atomic_json(a.run/'status.json',status)
                append_json(a.run/'metrics.jsonl',dict(iteration=iteration,time=time.time(),**metrics))
                print(json.dumps(status),flush=True)
                if not all(torch.isfinite(v).all() for v in policy.parameters()):raise RuntimeError('Nonfinite student')
                last_report=time.time()
            if time.time()-last_save>=180:
                save('periodic');last_save=time.time()
        status['status']='iterations_complete' if iteration>=a.max_updates else 'stopped_at_deadline' if time.time()>=deadline else 'stopped_requested'
    except BaseException as error:
        status.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        save('final');status.update(iteration=iteration,updated_unix=time.time());atomic_json(a.run/'status.json',status)
        env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
