"""Publish real unselected checkpoint replays at the requested interval (default180s) and once at final save; no PPO."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def atomic(p,d):
 t=p.with_name(p.name+'.video.partial');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--speed',type=float,default=.12);p.add_argument('--interval',type=int,default=180);a=p.parse_args();run=a.run.resolve()
statusfile=OUT/(run.name+'_video_watch.json');start=read(run/'status.json')['started_unix'];previous=read(OUT/'side_video_preview.json',{});last_hash=previous.get('checkpoint_sha256') if previous.get('run')==run.name else None;next_due=previous.get('published_unix',start)+a.interval-120 if last_hash else start;final_done=False
while time.time()<start+3600:
 if read(ROOT/'runs/current.json',{}).get('run_dir')!=str(run):break
 state=read(run/'status.json',{});stopped=state.get('status') in ['stopped_requested','stopped_at_deadline','iterations_complete','failed'];cp=read(run/'checkpoint.json',{})
 if not cp or not (stopped or time.time()>=next_due):time.sleep(5);continue
 if not stopped and cp.get('sha256')==last_hash:time.sleep(5);continue
 directory=OUT/'videos'/('%s_%06d_%s'%(run.name,cp['iteration'],time.strftime('%H%M%S')));directory.mkdir(parents=True)
 marker={'status':'recording','run':run.name,'iteration':cp['iteration'],'started_unix':time.time(),'interval_s':a.interval,'final':stopped,'directory':str(directory)};atomic(statusfile,marker)
 try:
  with (directory/'record.log').open('w') as log:
   subprocess.run([sys.executable,str(OUT/'record_powered_ankles_video_360p.py'),'--checkpoint',cp['path'],'--output-dir',str(directory),'--seconds','8','--speeds',str(a.speed)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
  receipt=read(directory/'recording_receipt.json');assert receipt['iteration']==cp['iteration'] and receipt['checkpoint_sha256']==cp['sha256']
  v=receipt['videos'][0];video=directory/v['video'];assert sha(video)==v['sha256']
  facts=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames:format=duration','-of','json',str(video)]));probe=facts['streams'][0];probe['duration']=facts['format']['duration'];assert probe['width']==640 and probe['height']==360 and probe['avg_frame_rate']=='24/1'
  subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  label='最终权重回放' if stopped else '迁移初值回放，尚非新奖励学习结果' if cp['iteration']==read(run/'status.json')['parent_checkpoint']['iteration'] else '训练检查点回放'
  published={'status':'ready','path':str(video),'run':run.name,'iteration':receipt['iteration'],'created_unix':receipt['created_unix'],'published_unix':time.time(),'sha256':v['sha256'],'bytes':video.stat().st_size,'checkpoint_sha256':receipt['checkpoint_sha256'],'description':'%s。指令%.0fcm/s，%.2f秒，%s；一个固定种子的未筛选样本，正常速度。每%d分钟更新，群体效果以独立评估为准。'%(label,a.speed*100,v['simulated_seconds'],'跌倒并保留片段' if v['fell'] else '未跌倒',a.interval//60),'recording_receipt':str(directory/'recording_receipt.json'),'probe':probe,'mounting_status':'corrected_inline','scope':'One unselected deterministic replay, not population evaluation. No PPO.','material_friction':receipt['friction']}
  atomic(directory/'published_receipt.json',published)
  if read(ROOT/'runs/current.json',{}).get('run_dir')==str(run):atomic(OUT/'side_video_preview.json',published)
  marker.update(status='published',finished_unix=time.time(),video=published);atomic(statusfile,marker);last_hash=receipt['checkpoint_sha256']
  if stopped:final_done=True;break
 except Exception as exc:
  marker.update(status='failed',finished_unix=time.time(),error=type(exc).__name__+': '+str(exc));atomic(statusfile,marker)
  if stopped:break
 next_due=max(marker['started_unix']+a.interval,time.time()+5)
 time.sleep(5)
