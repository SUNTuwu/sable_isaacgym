"""Unselected actual student replay, including encoder and both mirror branches."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from isaacgym import gymapi
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import migration_env
from migration_config import student_config
from history_policy import HistoryActorCritic
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--stage', default='S2')
    a = p.parse_args()
    teacher = verify_teacher_manifest(a.teacher)
    class CameraRobot(migration_env.MigrationRobot):
        def create_sim(self):
            self.graphics_device_id = 0
            return super().create_sim()
    migration_env.MigrationRobot = CameraRobot
    cfg = student_config(teacher['run'], a.stage, 1)
    cfg.env.episode_length_s = a.seconds+2
    cfg.commands.resampling_time = 1000.
    cfg.reference.file = '/missing_student_reference.json'
    torch.manual_seed(947)
    np.random.seed(947)
    env = migration_env.create_migration_env(cfg, noise=False)
    env.reset_noise = .015
    env.capture_eval = True
    cp = torch.load(str(a.checkpoint), map_location=env.device)
    policy = HistoryActorCritic().to(env.device).eval()
    policy.load_state_dict(cp['model_state_dict'], strict=True)
    camera_spec = gymapi.CameraProperties()
    camera_spec.width, camera_spec.height = 640, 360
    camera_spec.horizontal_fov = 48.
    camera_spec.near_plane, camera_spec.far_plane = .01, 30.
    camera = env.gym.create_camera_sensor(env.envs[0], camera_spec)
    assert camera >= 0
    a.output_dir.mkdir(parents=True, exist_ok=True)
    output = a.output_dir / ('student_%06d.mp4' % cp['iter'])
    process = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s','640x360','-r','50','-i','-','-an','-vf','fps=24','-c:v','libx264','-preset','fast',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)], stdin=subprocess.PIPE)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    checkpoint_sha = sha256(a.checkpoint)
    obs, _ = env.reset()
    commands = env.commands16.clone()
    commands[:, :3] = commands.new_tensor([.12, 0., 0.])
    env.set_commands(commands)
    frames, fell, duration = 0, False, 0.
    try:
        with torch.no_grad():
            for step in range(round(a.seconds/env.dt)):
                obs, _, _, done, _ = env.step(policy.act_inference(env.get_observations()))
                s = env.eval_snapshot
                duration = (step+1)*env.dt
                fell = bool(s['fall'][0])
                if bool(done[0]):
                    break
                xyz = s['xyz'][0].cpu().tolist()
                env.gym.fetch_results(env.sim, True)
                env.gym.set_camera_location(camera, env.envs[0], gymapi.Vec3(xyz[0]+.02, xyz[1]-.52, .16),
                                            gymapi.Vec3(xyz[0]+.02, xyz[1], .075))
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                pixels = env.gym.get_camera_image(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR)
                frame = Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(360, 640, 4)[:, :, :3])
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 45), fill=(17, 24, 32))
                draw.text((12, 6), 'SABLE CMD16 / HIST6 | student %d | actual replay / 1x' % cp['iter'], font=font, fill='white')
                draw.text((12, 24), 'No teacher or reference execution | seed 947 | front/rear friction 1.0', font=font, fill='#bacadb')
                draw.rectangle((0, 320, 640, 360), fill=(17, 24, 32))
                draw.text((12, 325), 'vx 0.120 m/s | yaw 0 rad/s | 1.0 Hz | t %.2f s' % duration, font=font, fill='white')
                draw.text((12, 343), 'Checkpoint SHA256 '+checkpoint_sha[:24], font=font, fill='#bacadb')
                process.stdin.write(np.asarray(frame).tobytes())
                frames += 1
    finally:
        process.stdin.close()
        code = process.wait()
        env.gym.destroy_sim(env.sim)
    if code or frames == 0:
        raise RuntimeError('No decodable replay frames, exit=%d' % code)
    atomic_json(a.output_dir/'recording_receipt.json', {'checkpoint': str(a.checkpoint),
        'checkpoint_sha256': checkpoint_sha, 'iteration': cp['iter'], 'video': str(output),
        'video_sha256': sha256(output), 'fell': fell, 'simulated_seconds': duration,
        'frames_50hz': frames, 'reference_execution_calls': env.reference_execution_calls,
        'seed': 947, 'created_unix': time.time(), 'scope': 'One unselected deterministic student replay; not population acceptance'})


if __name__ == '__main__':
    main()
