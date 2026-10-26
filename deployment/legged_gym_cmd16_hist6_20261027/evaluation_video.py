"""Predetermined scenario views from the same independent evaluation."""
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from isaacgym import gymapi
from run_io import atomic_json, sha256


class EvaluationVideo:
    def __init__(self, env, directory, checkpoint, iteration, seed, groups=None, envs_per_scenario=1):
        self.env,self.directory=env,Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.checkpoint,self.iteration,self.seed=str(checkpoint),iteration,seed
        self.checkpoint_sha=sha256(checkpoint)
        names=[name for name,_ in groups] if groups else ['forward09']
        self.max_seconds=8.
        if 'frequency_switch' in names:
            chosen=['frequency_switch'];self.max_seconds=28.
        elif 'left' in names:
            chosen=['forward105','reverse','left','turn_left']
        else:chosen=[names[0]]
        self.views=[]
        spec=gymapi.CameraProperties()
        spec.width,spec.height,spec.horizontal_fov=640,360,48.
        spec.near_plane,spec.far_plane=.01,30.
        for name in chosen:
            index=names.index(name)*envs_per_scenario
            camera=env.gym.create_camera_sensor(env.envs[index],spec)
            assert camera>=0
            origin=env.gym.get_env_origin(env.envs[index])
            self.views.append({'name':name,'index':index,'camera':camera,'origin':origin,
                'fell':False,'finished':False,'last_frame':None,'last_time':0.,'last_command':None})
        self.width,self.height=(1280,720) if len(chosen)>1 else (640,360)
        self.path=self.directory/('student_%06d.mp4'%iteration)
        self.process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s','%dx%d'%(self.width,self.height),'-r','25','-i','-','-an','-vf','fps=24',
            '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',
            str(self.path)],stdin=subprocess.PIPE)
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
        self.frames,self.seconds,self.closed=0,0.,False

    def sample(self,step,done,snapshot,commands):
        if self.closed:return
        self.seconds=(step+1)*self.env.dt
        if self.seconds>self.max_seconds:
            self.close();return
        for view in self.views:
            if bool(done[view['index']]):
                view['fell'] |= bool(snapshot['fall'][view['index']])
                view['finished']=True
        if step%2:return
        env=self.env
        env.gym.fetch_results(env.sim,True)
        for view in self.views:
            if view['finished']:continue
            xyz=snapshot['xyz'][view['index']].cpu().tolist();origin=view['origin']
            x,y=xyz[0]-origin.x,xyz[1]-origin.y
            env.gym.set_camera_location(view['camera'],env.envs[view['index']],
                gymapi.Vec3(x+.02,y-.62,.17-origin.z),gymapi.Vec3(x+.02,y,.085-origin.z))
        env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
        output=Image.new('RGB',(self.width,self.height),(17,24,32))
        for i,view in enumerate(self.views):
            index=view['index']
            if not view['finished']:
                pixels=env.gym.get_camera_image(env.sim,env.envs[index],view['camera'],gymapi.IMAGE_COLOR)
                view['last_frame']=Image.fromarray(np.asarray(pixels,dtype=np.uint8).reshape(360,640,4)[:,:,:3])
                view['last_time']=self.seconds
                view['last_command']=commands[index].cpu().tolist()
            frame=view['last_frame'].copy() if view['last_frame'] is not None else Image.new('RGB',(640,360))
            draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,45),fill=(17,24,32))
            draw.text((12,6),'SABLE CMD16 / HIST6 | student %d | %s | 1x'%(self.iteration,view['name']),font=self.font,fill='white')
            label='FELL - frozen last original-episode frame' if view['fell'] else 'No teacher/reference | seed %d | environment %d (predetermined)'%(self.seed,index)
            draw.text((12,24),label,font=self.font,fill='#ff9999' if view['fell'] else '#bacadb')
            draw.rectangle((0,320,640,360),fill=(17,24,32))
            c=view['last_command'] or commands[index].cpu().tolist()
            draw.text((12,325),'vx %.3f vy %.3f m/s | yaw %.2f rad/s | %.2f Hz | t %.2f s'%(c[0],c[1],c[2],c[5],view['last_time']),font=self.font,fill='white')
            draw.text((12,343),'Checkpoint SHA256 '+self.checkpoint_sha[:24],font=self.font,fill='#bacadb')
            output.paste(frame,((i%2)*640,(i//2)*360) if len(self.views)>1 else (0,0))
        self.process.stdin.write(np.asarray(output).tobytes());self.frames+=1

    def close(self):
        if self.closed:return
        self.closed=True;self.process.stdin.close();code=self.process.wait()
        fell=any(v['fell'] for v in self.views)
        if code or not self.frames:
            atomic_json(self.directory/'recording_failure.json',{'exit_code':code,'frames':self.frames,'fell':fell});return
        atomic_json(self.directory/'recording_receipt.json',{'checkpoint':self.checkpoint,
            'checkpoint_sha256':self.checkpoint_sha,'iteration':self.iteration,'video':str(self.path),
            'video_sha256':sha256(self.path),'fell':fell,'simulated_seconds':self.frames/25,
            'frames_25hz':self.frames,'reference_execution_calls':self.env.reference_execution_calls,
            'seed':self.seed,'views':[{'scenario':v['name'],'environment':v['index'],
                'fell':v['fell'],'last_original_time_s':v['last_time']} for v in self.views],
            'created_unix':time.time(),'scope':'First environment of each predetermined scenario; same independent evaluation, no extra rollout or selection; failed originals freeze, resets never replace them'})
