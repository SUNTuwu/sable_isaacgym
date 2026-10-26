"""Remote-only deterministic camera replay using a run's frozen runtime.

No PPO, checkpoint changes, or edits to the frozen runtime. Camera and HUD only.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=16.)
    p.add_argument('--speeds', type=float, nargs='+', default=[.03, .06])
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    run = args.checkpoint.resolve().parent.parent
    runtime = run / 'runtime'
    sys.path.insert(0, str(runtime))
    from isaacgym import gymapi
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import preflight
    from sable_config import SableCfg
    from sable_env import SableRobot
    from equivariant_policy import SymmetricActorCritic

    class CameraRobot(SableRobot):
        def create_sim(self):
            self.graphics_device_id = 0
            return super().create_sim()

    preflight.SableRobot = CameraRobot
    config = json.loads((run / 'config.json').read_text())
    assert config['mirror_policy'], 'This recorder expects the v11 projected actor'
    cfg = SableCfg()
    cfg.env.episode_length_s = args.seconds + 2.
    cfg.env.num_observations = config['env']['env']['num_observations']
    cfg.commands.resampling_time = 1000.
    cfg.commands.heading_command = config['env']['commands']['heading_command']
    for group in ['gait', 'reference']:
        for key, value in config['env'][group].items():
            if key != 'file':
                setattr(getattr(cfg, group), key, value)
    torch.manual_seed(947)
    np.random.seed(947)
    env = preflight.create_env(1, noise=False, cfg=cfg)
    env.reset_noise = .015
    # Display colors only: identify left/right limbs without changing physics.
    for body_index, name in enumerate(env.body_names):
        color = (.12, .66, .54) if name.endswith('_fixed') else (.27, .53, .76) if name.startswith('R_') else (.84, .56, .24) if name.startswith('L_') else (.48, .54, .60)
        env.gym.set_rigid_body_color(env.envs[0], env.actor_handles[0], body_index,
                                     gymapi.MESH_VISUAL, gymapi.Vec3(*color))
    actor = SymmetricActorCritic(env.num_obs, env.num_obs, 12,
                                **config['train']['policy']).to('cuda:0')
    cp = torch.load(str(args.checkpoint), map_location='cuda:0')
    actor.load_state_dict(cp['model_state_dict'])
    actor.eval()
    width, height, fps = 1280, 720, 25
    stride = round(1. / (fps * env.dt))
    assert abs(stride * env.dt * fps - 1.) < 1e-5
    prop = gymapi.CameraProperties()
    prop.width, prop.height = width, height
    prop.horizontal_fov = 48.
    prop.near_plane, prop.far_plane = .01, 30.
    camera = env.gym.create_camera_sensor(env.envs[0], prop)
    if camera < 0:
        raise RuntimeError('Isaac Gym camera creation failed')
    env.gym.set_light_parameters(env.sim, 0, gymapi.Vec3(.85, .85, .85),
                                 gymapi.Vec3(.5, .5, .5), gymapi.Vec3(.5, -.7, -1.))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 20)
    cp_hash = digest(args.checkpoint)
    all_reports = []
    started = time.time()
    try:
        for speed in args.speeds:
            torch.manual_seed(947)
            np.random.seed(947)
            env.fixed_command = speed
            env.reset()
            env.episode_length_buf.zero_()
            env.capture_eval = True
            if env.motion_reference is not None:
                env.motion_reference.reset(torch.arange(env.num_envs, device=env.device))
            env.commands[:, 0] = speed
            env.commands[:, 1:] = 0.
            env.compute_observations()
            obs = env.get_observations()
            start = env.root_states[0, :3].cpu().numpy().copy()
            label = 'v11_%06d_side_%03dmmps_720p' % (cp['iter'], round(speed * 1000))
            output = args.output_dir / (label + '.mp4')
            if output.exists():
                raise FileExistsError(output)
            ffmpeg = subprocess.Popen([
                'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (width, height),
                '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(output)], stdin=subprocess.PIPE)
            frames, fell, trace = 0, False, []
            try:
                # no_grad avoids inference-only tensors persisting across resets.
                with torch.no_grad():
                    for k in range(round(args.seconds / env.dt)):
                        obs, _, _, done, _ = env.step(actor.act_inference(obs))
                        snap = env.eval_snapshot
                        xyz = snap['xyz'][0].cpu().numpy()
                        fell = bool(snap['fall'][0])
                        # No automatic-reset frames are passed off as continuous motion.
                        if bool(done[0]):
                            print('EPISODE_ENDED', speed, (k + 1) * env.dt, 'fell', fell, flush=True)
                            break
                        if (k + 1) % stride:
                            continue
                        env.gym.fetch_results(env.sim, True)
                        aim_x = float(xyz[0]) + .02
                        aim_y = float(xyz[1])
                        env.gym.set_camera_location(camera, env.envs[0],
                            gymapi.Vec3(aim_x, aim_y - .52, .15),
                            gymapi.Vec3(aim_x, aim_y, .065))
                        env.gym.step_graphics(env.sim)
                        env.gym.render_all_camera_sensors(env.sim)
                        pixels = env.gym.get_camera_image(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR)
                        rgb = np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()
                        frame = Image.fromarray(rgb)
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, width, 94), fill=(17, 24, 32))
                        draw.text((25, 14), 'SABLE  |  v11 iteration %d  |  SIDE VIEW  |  1x real time' % cp['iter'], font=font, fill='white')
                        draw.text((25, 53), 'Isaac Gym / RTX 4090   -   fixed reference + learned residual (projected policy)', font=small, fill=(182, 202, 219))
                        draw.rectangle((0, 630, width, 720), fill=(17, 24, 32))
                        t = (k + 1) * env.dt
                        dx = float(xyz[0] - start[0])
                        draw.text((25, 643), 'Command %.3f m/s     Time %05.2f s     Forward displacement %+.3f m' % (speed, t, dx), font=font, fill='white')
                        draw.text((25, 684), '1280 x 720 / 25 fps   |   +X forward = screen right   |   SHA256 ' + cp_hash[:16], font=small, fill=(182, 202, 219))
                        ffmpeg.stdin.write(np.asarray(frame).tobytes())
                        if frames in (0, fps * 8, fps * 15):
                            frame.save(args.output_dir / (label + '_frame%04d.png' % frames))
                        frames += 1
                        trace.append({'t': t, 'xyz': xyz.tolist(), 'q': snap['q'][0].cpu().tolist(),
                                      'feet_force': snap['force'][0].cpu().tolist(),
                                      'clearance_m': snap['clearance'][0].cpu().tolist()})
                        if frames % 100 == 0:
                            print('RECORDING', json.dumps({'speed': speed, 'frames': frames, 'sim_time': t, 'dx': dx}), flush=True)
                        if args.probe:
                            break
            finally:
                ffmpeg.stdin.close()
                if ffmpeg.wait() != 0:
                    raise RuntimeError('ffmpeg failed')
            report = {'video': output.name, 'frames': frames, 'fps': fps, 'resolution': [width, height],
                      'command_m_s': speed, 'fell': fell, 'simulated_seconds': (k + 1) * env.dt,
                      'forward_displacement_m': float(xyz[0] - start[0]), 'trace': trace,
                      'sha256': digest(output)}
            all_reports.append(report)
            print('VIDEO_DONE', json.dumps({k: v for k, v in report.items() if k != 'trace'}), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)
    receipt = {'schema': 'sable-remote-side-camera/1', 'checkpoint': str(args.checkpoint),
               'checkpoint_sha256': cp_hash, 'iteration': cp['iter'], 'runtime_sha256': cp['infos']['runtime_sha256'],
               'recorder_sha256': digest(__file__), 'created_unix': time.time(), 'wall_seconds': time.time() - started,
               'training': False, 'seed': 947, 'reset_noise_rad': .015, 'friction': 1.,
               'mirror_policy': True, 'reference_strength': cfg.reference.strength,
               'camera': 'world-side follow, right side (-Y), fixed height and yaw; +X is screen right',
               'sample_scope': 'One unselected rollout per command; not a population success-rate evaluation.',
               'videos': all_reports}
    (args.output_dir / 'recording_receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
