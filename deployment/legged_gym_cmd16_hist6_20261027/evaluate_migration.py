"""Fresh-process student/teacher evaluation with commanded and actual motion."""
import argparse
import json
import math
import time
from pathlib import Path
from isaacgym import gymapi
import numpy as np
import torch
from migration_config import student_config
from migration_env import create_migration_env
from history_policy import HistoryActorCritic
from model_mapping import heading_velocity, quat_apply
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json, sha256
from evaluation_telemetry import EvaluationTelemetry
from evaluation_contact_forces import ContactForceTelemetry


from evaluation_scenarios import scenarios, scheduled_commands


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher-control', action='store_true')
    p.add_argument('--suite', choices=['forward', 'S2', 'S3', 'S5'], default='S2')
    p.add_argument('--envs-per-scenario', type=int, default=8)
    p.add_argument('--seconds', type=float, default=16.)
    p.add_argument('--seed', type=int, default=947)
    p.add_argument('--video-dir', type=Path)
    p.add_argument('--deployment-trace', type=Path)
    p.add_argument('--frontier-alphas', type=float, nargs='+', default=[])
    p.add_argument('--yaw-error-scale', type=float)
    a = p.parse_args()
    if not a.teacher_control and not a.checkpoint:
        p.error('--checkpoint is required for student evaluation')
    teacher = verify_teacher_manifest(a.teacher)
    groups = scenarios(a.suite)
    from evaluation_frontier import extend_groups, split_frontier
    if a.frontier_alphas and (a.suite != 'S3' or a.teacher_control):
        p.error('Intermediate commands are only available for student S3 monitoring')
    groups, frontier_specs = extend_groups(groups, a.frontier_alphas)
    n = len(groups)*a.envs_per_scenario
    cfg = student_config(teacher['run'], stage='S2' if a.suite in ('forward', 'S2') else a.suite, num_envs=n)
    if a.yaw_error_scale is not None:
        if not math.isfinite(a.yaw_error_scale) or a.yaw_error_scale <= 0:
            p.error('Yaw error scale must be finite and positive')
        cfg.migration.yaw_error_scale_rad_s=a.yaw_error_scale
    cfg.env.episode_length_s = a.seconds+1.
    cfg.commands.resampling_time = 1000.
    if not a.teacher_control:
        cfg.reference.file = '/intentionally_absent_student_reference.json'
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    env = create_migration_env(cfg, noise=False, camera=bool(a.video_dir))
    env.reset_noise = .015
    env.capture_eval = True
    if a.teacher_control:
        policy = None
        env.attach_teacher(teacher['checkpoint'], teacher['sha256'], 0.)
        iteration = teacher['iteration']
        checkpoint_sha = teacher['sha256']
    else:
        cp = torch.load(str(a.checkpoint), map_location=env.device)
        policy = HistoryActorCritic().to(env.device).eval()
        policy.load_state_dict(cp['model_state_dict'], strict=True)
        iteration = cp['iter']
        checkpoint_sha = sha256(a.checkpoint)
    obs, _ = env.reset()
    video = None
    if a.video_dir:
        from evaluation_video import EvaluationVideo
        video = EvaluationVideo(env,a.video_dir,a.checkpoint,iteration,a.seed,groups,a.envs_per_scenario)
    commands = env.commands16.clone()
    alive = torch.ones(n, device=env.device, dtype=torch.bool)
    duration = torch.zeros(n, device=env.device)
    sums = torch.zeros(n, 10, device=env.device)
    falls = torch.zeros(n, dtype=torch.bool, device=env.device)
    knee_falls = torch.zeros_like(falls)
    prior_contact = torch.ones(n, 4, dtype=torch.bool, device=env.device)
    takeoff_xy = torch.zeros(n, 4, 2, device=env.device)
    takeoff_time = torch.zeros(n, 4, device=env.device)
    clear_time = torch.zeros(n, 4, device=env.device)
    max_clear_time = clear_time.clone()
    last_landing = torch.full((n, 4), -1., device=env.device)
    step_counts = torch.zeros(n, 4, dtype=torch.long, device=env.device)
    events, trace = [], []
    segment_sums = {}
    reward_sums = {}
    deployment_trace = []
    telemetry = EvaluationTelemetry(env)
    force_telemetry = ContactForceTelemetry(env,groups,a.envs_per_scenario)
    started = time.time()
    with torch.no_grad():
        for step in range(round(a.seconds/env.dt)):
            t = step*env.dt
            metric_valid = torch.ones(n, dtype=torch.bool, device=env.device)
            segments_this_step = []
            for g, (name, initial) in enumerate(groups):
                values, changed = scheduled_commands(name, initial, t)
                sl = slice(g*a.envs_per_scenario, (g+1)*a.envs_per_scenario)
                commands[sl, :3] = commands.new_tensor(values[:3])
                commands[sl, 5] = values[3]
                metric_valid[sl] = t >= 1. and (changed == 0. or t-changed >= .75)
                segments_this_step.append((g,name,changed,sl))
            env.set_commands(commands)
            obs = env.get_observations()
            action = torch.zeros(n, 15, device=env.device) if policy is None else policy.act_inference(obs)
            trace_input = obs.clone() if a.deployment_trace and not a.teacher_control else None
            obs, _, _, done, _ = env.step(action)
            s = env.eval_snapshot
            if trace_input is not None:
                # Full pre-reset target from this exact action, including failed
                # transitions; never substitute the post-reset default target.
                deployment_trace.append({'raw':trace_input.cpu(),'action':action.cpu(),
                    'target':s['executed_target'].cpu(),'time_s':t})
            if video is not None:
                video.sample(step,done,s,commands)
            world_velocity = quat_apply(s['base_quat'], s['velocity_local'])
            world_omega = quat_apply(s['base_quat'], s['angular_velocity_local'])
            xy_heading, yaw_heading = heading_velocity(s['base_quat'], world_velocity, world_omega)
            xy = xy_heading if env.omni else s['velocity_local'][:, :2]
            yaw = yaw_heading if env.omni else s['angular_velocity_local'][:, 2]
            desired = commands[:, :3].clone() if env.omni else s['commands'][:, :3]
            valid = alive & metric_valid
            for key,value in s['weighted_rewards'].items():
                if key not in reward_sums:reward_sums[key]=torch.zeros(n,device=env.device)
                reward_sums[key] += value*valid
            telemetry.update(s,commands,alive,valid)
            force_telemetry.update(s,commands,alive,done,step)
            error_xy = (xy-desired[:, :2]).square()
            error_yaw = (yaw-desired[:, 2]).square()
            values = torch.cat((error_xy, error_yaw[:, None], xy, yaw[:, None], desired, torch.ones(n, 1, device=env.device)), 1)
            sums += values*valid[:, None]
            for g,name,changed,sl in segments_this_step:
                key=(g,name,changed)
                if key not in segment_sums:segment_sums[key]=torch.zeros(a.envs_per_scenario,10,device=env.device)
                segment_sums[key] += values[sl]*valid[sl,None]
            duration += alive*env.dt
            contact = s['force'] > 1.
            takeoff = prior_contact & ~contact
            landing = ~prior_contact & contact
            takeoff_xy[takeoff] = s['foot_centers'][:, :, :2][takeoff]
            takeoff_time[takeoff] = t
            clear_time[takeoff] = 0.
            max_clear_time[takeoff] = 0.
            clear_time = torch.where(~contact & (s['clearance'] >= .002), clear_time+env.dt, torch.zeros_like(clear_time))
            max_clear_time = torch.maximum(max_clear_time, clear_time)
            travel = (s['foot_centers'][:, :, :2]-takeoff_xy).norm(dim=-1)
            qualified = landing & (max_clear_time >= .060-1e-7) & (travel >= .005) & (takeoff_time >= 1.) & alive[:, None]
            interval = t+env.dt-last_landing
            for e, f in qualified.nonzero(as_tuple=False).cpu().tolist():
                previous = float(last_landing[e, f])
                event = {'environment': e, 'scenario': groups[e//a.envs_per_scenario][0], 'foot': f,
                         'time_s': t+env.dt, 'travel_m': float(travel[e, f]),
                         'command_frequency_hz': float(commands[e, 5]),
                         'interval_s': float(interval[e, f]) if previous >= 0 else None}
                # Intervals straddling command changes are transition data, not steady frequency evidence.
                _, changed = scheduled_commands(event['scenario'], groups[e//a.envs_per_scenario][1], t)
                event['segment_start_s'] = changed
                event['steady_interval'] = previous >= max(1., changed+.75)
                events.append(event)
            last_landing[qualified] = t+env.dt
            step_counts += qualified
            prior_contact = contact
            if step % 5 == 0 and bool(alive[0]):
                trace.append({'t': t+env.dt, 'body_xyz': s['body_xyz'][0].cpu().tolist(),
                    'q': s['q'][0].cpu().tolist(), 'command': desired[0].cpu().tolist(),
                    'actual_xy_m_s': xy_heading[0].cpu().tolist(), 'actual_yaw_rad_s': float(yaw_heading[0]),
                    'foot_center_xyz': s['foot_centers'][0].cpu().tolist(), 'clearance_m': s['clearance'][0].cpu().tolist()})
            falls |= alive & s['fall']
            knee_falls |= alive & s['knee_ground_contact'].any(1)
            alive &= ~done
    rows = []
    for g, (name, initial) in enumerate(groups):
        sl = slice(g*a.envs_per_scenario, (g+1)*a.envs_per_scenario)
        total = sums[sl].sum(0).cpu().numpy()
        count = float(total[9])
        minimum_steps = max(2,int(max(a.seconds-2.,0.)*initial[3]*.5))
        steady_frequency = []
        for foot in range(4):
            intervals = [e['interval_s'] for e in events if e['scenario']==name and e['foot']==foot and e['steady_interval'] and e['interval_s']]
            steady_frequency.append(float(1/np.median(intervals)) if intervals else None)
        rows.append({'name': name, 'initial_command': initial,
            'survival_fraction': float((~falls[sl]).float().mean()), 'mean_duration_s': float(duration[sl].mean()),
            'knee_failure_fraction': float(knee_falls[sl].float().mean()), 'samples': int(count),
            'xy_rmse_m_s': float(np.sqrt(total[:2].sum()/count)) if count else None,
            'forward_rmse_m_s': float(np.sqrt(total[0]/count)) if count else None,
            'yaw_rmse_rad_s': float(np.sqrt(total[2]/count)) if count else None,
            'mean_actual_xyz_command_units': (total[3:6]/count).tolist() if count else None,
            'mean_command': (total[6:9]/count).tolist() if count else None,
            'steps_per_foot': step_counts[sl].sum(0).cpu().tolist(),
            'steps_per_environment':step_counts[sl].cpu().tolist(),
            'minimum_completed_steps_for_motion_check':minimum_steps,
            'four_feet_stepping_fraction':float((step_counts[sl].min(1).values>=minimum_steps).float().mean()),
            'steady_frequency_per_foot_hz':steady_frequency})
    forward_rows = rows[:3]
    count = sum(r['samples'] for r in forward_rows)
    velocity_rmse = (sum(r['samples']*r['forward_rmse_m_s']**2 for r in forward_rows if r['samples'])/count)**.5 if count else None
    frequencies = []
    for name, initial in groups:
        if not name.startswith('frequency'):
            continue
        segments = [(0., initial[3])] if name != 'frequency_switch' else [(0., .8), (8., 1.2), (20., .8)]
        for changed, target in segments:
            selected = [e for e in events if e['scenario']==name and abs(e['command_frequency_hz']-target)<1e-5 and e['segment_start_s']==changed and e['steady_interval'] and e['interval_s']]
            intervals = [e['interval_s'] for e in selected]
            feet = []
            for foot in range(4):
                fi = [e['interval_s'] for e in selected if e['foot']==foot]
                feet.append({'foot':foot, 'measured_frequency_hz':float(1/np.median(fi)) if fi else None,
                             'interval_count':len(fi), 'environments':len({e['environment'] for e in selected if e['foot']==foot})})
            frequencies.append({'name': name, 'command_frequency_hz': target,
                'segment_start_s':changed, 'feet':feet,
                'measured_frequency_hz': float(1/np.median(intervals)) if intervals else None,
                'interval_count': len(intervals)})
    result = {'schema': 'sable-cmd16-independent-evaluation/1', 'iteration': iteration,
        'checkpoint': str(a.checkpoint) if a.checkpoint else teacher['checkpoint'], 'checkpoint_sha256': checkpoint_sha,
        'teacher_control': a.teacher_control, 'teacher_sha256': teacher['sha256'], 'suite': a.suite,
        'horizon_s': a.seconds, 'seed': a.seed, 'num_envs': n, 'environments_per_scenario': a.envs_per_scenario,
        'survival_fraction': float((~falls).float().mean()), 'velocity_rmse_m_s': velocity_rmse,
        'reference_execution_calls': env.reference_execution_calls, 'scenarios': rows,
        'frequency_scenarios': frequencies, 'events': events, 'trace': trace, 'body_names': env.body_names,
        'dof_names': env.dof_names, 'foot_order': ['FL', 'FR', 'HL', 'HR'],
        'asset_sha256': sha256(cfg.asset.file), 'manifest_sha256': sha256(Path(cfg.asset.file).with_name('physical_manifest.json')),
        'evaluator_sha256': sha256(__file__), 'environment_sha256': sha256(Path(__file__).with_name('migration_env.py')),
        'created_unix': time.time(), 'wall_seconds': time.time()-started, 'accepted_locomotion': False,
        'measurement_scope': '20ms control-rate samples; original episodes only; genuine2mm/60ms/5mm unsigned landing events; missing events remain missing'}
    result['evaluation_yaw_error_scale_rad_s']=cfg.migration.yaw_error_scale_rad_s
    result['velocity_segments']=[]
    for (g,name,changed),values in segment_sums.items():
        total=values.sum(0).cpu().numpy();count=float(total[9])
        result['velocity_segments'].append({'name':name,'segment_start_s':changed,'samples':int(count),
            'mean_command':(total[6:9]/count).tolist() if count else None,
            'mean_actual':(total[3:6]/count).tolist() if count else None,
            'xy_rmse_m_s':float(np.sqrt(total[:2].sum()/count)) if count else None,
            'yaw_rmse_rad_s':float(np.sqrt(total[2]/count)) if count else None})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result['diagnostics'] = telemetry.finish(groups,a.envs_per_scenario)
    result['contact_force_analysis'] = force_telemetry.finish()
    result['weighted_rewards_per_second'] = []
    for g,(name,_) in enumerate(groups):
        sl=slice(g*a.envs_per_scenario,(g+1)*a.envs_per_scenario)
        count=float(sums[sl,9].sum())
        values={k:float(v[sl].sum())/(max(count,1)*env.dt) for k,v in reward_sums.items()}
        if count:
            assert abs(sum(v for k,v in values.items() if k!='total')-values['total'])<1e-3
        result['weighted_rewards_per_second'].append({'name':name,'samples':int(count),'values':values})
    if a.deployment_trace:
        if a.teacher_control:raise ValueError('Deployment parity needs a student checkpoint')
        a.deployment_trace.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'schema':'sable-deployment-trace/1','checkpoint_sha256':checkpoint_sha,
            'suite':a.suite,'seed':a.seed,'scenario_names':[name for name,_ in groups],
            'envs_per_scenario':a.envs_per_scenario,'samples':deployment_trace},str(a.deployment_trace))
        result['deployment_trace']={'path':str(a.deployment_trace),'sha256':sha256(a.deployment_trace)}
    if video is not None:
        video.close()
        result['video_receipt'] = str(a.video_dir/'recording_receipt.json')
    split_frontier(result, frontier_specs)
    result.update(created_unix=time.time(),wall_seconds=time.time()-started)
    atomic_json(a.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','trace','body_names','dof_names','diagnostics')}), flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
