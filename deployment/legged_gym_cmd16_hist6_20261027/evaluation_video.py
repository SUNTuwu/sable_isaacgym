"""Record environment zero during the same independent evaluation simulation."""
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from isaacgym import gymapi
from run_io import atomic_json, sha256


class EvaluationVideo:
    def __init__(self, env, directory, checkpoint, iteration, seed):
        self.env, self.directory = env, Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.checkpoint, self.iteration, self.seed = str(checkpoint), iteration, seed
        self.checkpoint_sha = sha256(checkpoint)
        spec = gymapi.CameraProperties()
        spec.width, spec.height, spec.horizontal_fov = 640, 360, 48.
        spec.near_plane, spec.far_plane = .01, 30.
        self.camera = env.gym.create_camera_sensor(env.envs[0],spec)
        assert self.camera >= 0
        self.path = self.directory/('student_%06d.mp4'%iteration)
        self.process = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s','640x360','-r','25','-i','-','-an','-vf','fps=24','-c:v','libx264','-preset','fast',
            '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
        self.frames, self.fell, self.seconds, self.closed = 0, False, 0., False

    def sample(self, step, done, snapshot, command):
        if self.closed:return
        self.seconds = (step+1)*self.env.dt
        self.fell = bool(snapshot['fall'][0])
        if bool(done[0]) or self.seconds > 8.:
            self.close()
            return
        if step%2:return
        env = self.env
        xyz = snapshot['xyz'][0].cpu().tolist()
        env.gym.fetch_results(env.sim,True)
        env.gym.set_camera_location(self.camera,env.envs[0],gymapi.Vec3(xyz[0]+.02,xyz[1]-.52,.16),
                                    gymapi.Vec3(xyz[0]+.02,xyz[1],.075))
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        pixels = env.gym.get_camera_image(env.sim,env.envs[0],self.camera,gymapi.IMAGE_COLOR)
        frame = Image.fromarray(np.asarray(pixels,dtype=np.uint8).reshape(360,640,4)[:,:,:3])
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0,0,640,45),fill=(17,24,32))
        draw.text((12,6),'SABLE CMD16 / HIST6 | student %d | actual evaluation / 1x'%self.iteration,font=self.font,fill='white')
        draw.text((12,24),'No teacher/reference execution | seed %d | environment 0 (unselected)'%self.seed,font=self.font,fill='#bacadb')
        draw.rectangle((0,320,640,360),fill=(17,24,32))
        c = command.cpu().tolist()
        draw.text((12,325),'vx %.3f vy %.3f m/s | yaw %.2f rad/s | %.2f Hz | t %.2f s'%(c[0],c[1],c[2],c[5],self.seconds),font=self.font,fill='white')
        draw.text((12,343),'Checkpoint SHA256 '+self.checkpoint_sha[:24],font=self.font,fill='#bacadb')
        self.process.stdin.write(np.asarray(frame).tobytes())
        self.frames += 1

    def close(self):
        if self.closed:return
        self.closed = True
        self.process.stdin.close()
        code = self.process.wait()
        if code or not self.frames:
            atomic_json(self.directory/'recording_failure.json',{'exit_code':code,'frames':self.frames,'fell':self.fell})
            return
        atomic_json(self.directory/'recording_receipt.json',{'checkpoint':self.checkpoint,
            'checkpoint_sha256':self.checkpoint_sha,'iteration':self.iteration,'video':str(self.path),
            'video_sha256':sha256(self.path),'fell':self.fell,'simulated_seconds':self.frames/25,
            'frames_25hz':self.frames,'reference_execution_calls':self.env.reference_execution_calls,
            'seed':self.seed,'environment':0,'created_unix':time.time(),
            'scope':'Environment zero of the same independent evaluation; no extra rollout or selection'})
