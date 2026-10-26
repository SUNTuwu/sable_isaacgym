"""Controlled termination cases plus a forced physical knee-ground contact."""
from pathlib import Path
import json
from isaacgym import gymapi
import torch
from preflight import create_env
from sable_config import SableCfg, MODEL
from run_io import atomic_json, sha256

here = Path(__file__).resolve().parent
env = create_env(5, fixed=True, noise=False)
env.reset_noise = 0.; env.fixed_command = 0.; env.reset()
base = env.rigid_states.clone()
detector = env.knee_contact_detector
clearance = detector.clearance(base)
assert bool((clearance > .002).all()), clearance
# Isolated geometry cases: unchanged roots, tilt, joints, and torso forces.
env.contact_forces.zero_()
env.contact_forces[0, env.body_names.index('L_hip_pitch'), 0] = 100.
for row, sides in [(1, [0]), (2, [1]), (3, [0, 1]), (4, [0])]:
    for side in sides:
        owner = env.body_names.index(['L_hip_pitch', 'R_hip_pitch'][side])
        env.rigid_states[row, owner, 2] -= clearance[row, side] - .0004
env.episode_length_buf.zero_()
env.episode_length_buf[4] = env.max_episode_length
detector.update(); env.check_termination()
expected = torch.tensor([False, True, True, True, True], device=env.device)
isolated_pass = bool(torch.equal(env.true_fall, expected)) and not bool(env.time_out_buf.any())
assert bool(torch.equal(env._reward_termination(), expected.float()))
isolated = {'cases': ['airborne self-contact force', 'left knee touching', 'right knee touching', 'both knees touching', 'knee touch coincident with timeout'],
            'actual_death': env.true_fall.cpu().tolist(), 'timeout': env.time_out_buf.cpu().tolist(),
            'knee_clearance_m': env.knee_clearance.cpu().tolist(), 'passed': isolated_pass}
env.gym.destroy_sim(env.sim)

# Fixed root lowered into the plane to produce actual collision solver contact.
# This deliberately impossible placement is a narrow regression, not a gait test.
cfg = SableCfg(); cfg.init_state.pos = list(MODEL['init_position']); cfg.init_state.pos[2] -= .06
env = create_env(1, fixed=True, noise=False, cfg=cfg)
env.reset_noise = 0.; env.fixed_command = 0.; env.capture_eval = True
env.reset(); records = []
for k in range(5):
    _, _, _, done, extras = env.step(torch.zeros(1, 12, device=env.device))
    s = env.eval_snapshot
    owners = [env.body_names.index(n) for n in ['L_hip_pitch', 'R_hip_pitch']]
    records.append({'step': k, 'done': bool(done[0]), 'fall': bool(s['fall'][0]),
                    'knee_contact': s['knee_ground_contact'][0].cpu().tolist(),
                    'clearance_m': s['knee_clearance'][0].cpu().tolist(),
                    'knee_owner_force_n': s['body_force_norm'][0, owners].cpu().tolist(),
                    'timeout': bool(extras['time_outs'][0])})
    if records[-1]['fall'] and any(records[-1]['knee_contact']):
        break
physical_pass = any(row['done'] and row['fall'] and any(row['knee_contact']) and max(row['knee_owner_force_n']) > .01 and not row['timeout'] for row in records)
report = {'passed': isolated_pass and physical_pass, 'isolated_cases': isolated,
          'forced_physical_contact': records, 'contact_margin_m': cfg.termination.knee_contact_margin_m,
          'control_step_s': env.dt, 'checked_shapes': env.knee_contact_detector.names,
          'source_sha256': {n: sha256(here/n) for n in ['sable_env.py', 'sable_config.py', 'knee_contact.py', 'check_knee_termination.py', 'assets/sable.xml']},
          'scope': 'Single-step left/right/both termination, self-contact rejection, no timeout bootstrap on death, and real forced-plane collision. Normal 16s standing is checked separately.'}
atomic_json(here/'verification/knee_termination.json', report)
print('KNEE_TERMINATION', json.dumps(report), flush=True)
env.gym.destroy_sim(env.sim)
if not report['passed']:
    raise SystemExit(2)
