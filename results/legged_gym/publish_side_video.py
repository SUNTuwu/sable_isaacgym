"""Package actual camera segments, decode-check, then atomically publish metadata."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
folder=a.directory.resolve();receipt=json.loads((folder/'recording_receipt.json').read_text())
assert receipt['training'] is False
parts=[];durations=[]
for video in receipt['videos']:
    path=folder/video['video'];assert path.parent==folder
    assert hashlib.sha256(path.read_bytes()).hexdigest()==video['sha256']
    parts.append("file '"+path.name+"'\n")
    durations.append(video['source_frames']/video['source_fps'])
listing=folder/'concat.txt';listing.write_text(''.join(parts))
output=folder/('sable_%s_%d_side_360p24.mp4'%(receipt['run'].split('_')[2],receipt['iteration']))
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','concat','-safe','1','-i',str(listing),'-c','copy','-movflags','+faststart',str(output)],check=True)
probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(output)],capture_output=True,text=True,check=True)
assert not probe.stderr,probe.stderr
stream=json.loads(probe.stdout)['streams'][0]
assert stream['width']==640 and stream['height']==360 and stream['avg_frame_rate']=='24/1'
assert int(stream['nb_read_frames'])==round(float(stream['duration'])*24)
assert abs(float(stream['duration'])-sum(durations))<=len(parts)/24+.001
description='；'.join('%.3fm/s：%.2f秒，%s'%(v['command_m_s'],seconds,'跌倒后结束' if v['fell'] else '未跌倒') for v,seconds in zip(receipt['videos'],durations))+'。各一个未筛选样本，正常速度。'
metadata={'status':'ready','path':str(output),'run':receipt['run'],'iteration':receipt['iteration'],
          'created_unix':receipt['created_unix'],'published_unix':time.time(),
          'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size,
          'checkpoint_sha256':receipt['checkpoint_sha256'],'description':description,
          'recording_receipt':str(folder/'recording_receipt.json'),'probe':stream,
          'packager_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'Unselected single rollouts, not a population evaluation; 50Hz physical frames resampled to24fps at1x.'}
(folder/'video_receipt.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
target=Path(__file__).resolve().parent/'side_video_preview.json';temp=target.with_suffix('.partial')
temp.write_text(json.dumps(metadata,ensure_ascii=False,indent=2));temp.replace(target)
print(json.dumps(metadata,ensure_ascii=False))
