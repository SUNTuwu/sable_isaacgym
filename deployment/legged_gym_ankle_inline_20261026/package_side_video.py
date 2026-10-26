"""Package the unselected side-camera replays into a checked 720p review clip."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,codec_name,pix_fmt,avg_frame_rate,nb_read_frames:format=duration',
        '-of', 'json', str(path)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    a = parser.parse_args()
    receipt = json.loads((a.directory / 'recording_receipt.json').read_text())
    entries = []
    for video in receipt['videos']:
        path = a.directory / video['video']
        assert sha(path) == video['sha256']
        facts = probe(path)
        stream = facts['streams'][0]
        assert (stream['width'], stream['height'], stream['avg_frame_rate']) == (1280, 720, '25/1')
        assert int(stream['nb_read_frames']) == video['frames']
        entries.append("file '%s'" % path.name)
    source_list = a.directory / 'concat.txt'
    source_list.write_text('\n'.join(entries) + '\n')
    caption = a.directory / 'caption.txt'
    caption.write_text('Isaac Gym / RTX 4090  |  Fixed reference + learned residual  |  Blue: right / Orange: left')
    output = a.directory / 'sable_v11_2674_side_720p.mp4'
    filters = ('drawbox=x=0:y=47:w=iw:h=47:color=0x111820:t=fill,'
               'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:'
               'textfile=%s:x=25:y=54:fontsize=20:fontcolor=0xB6CADB' % caption)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n',
        '-f', 'concat', '-safe', '0', '-i', str(source_list), '-vf', filters,
        '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output)], check=True)
    facts = probe(output)
    assert int(facts['streams'][0]['nb_read_frames']) == sum(v['frames'] for v in receipt['videos'])
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
    for sec in [2, 8, 18, 24, 31]:
        if sec >= float(facts['format']['duration']):
            continue
        subprocess.run(['ffmpeg', '-v', 'error', '-n', '-ss', str(sec), '-i', str(output),
                        '-frames:v', '1', str(a.directory / ('review_%02ds.png' % sec))], check=True)
    result = {'video': output.name, 'sha256': sha(output), 'bytes': output.stat().st_size,
              'probe': facts, 'checkpoint_sha256': receipt['checkpoint_sha256'],
              'runtime_sha256': receipt['runtime_sha256'], 'source_receipt_sha256': sha(a.directory / 'recording_receipt.json'),
              'packager_sha256': sha(Path(__file__)), 'order_m_s': [v['command_m_s'] for v in receipt['videos']],
              'edits': 'Concatenation and accurate reference/residual caption only; original 1x timing, no frame selection.',
              'samples': [{k: v[k] for k in ['command_m_s', 'fell', 'frames', 'forward_displacement_m', 'simulated_seconds']} for v in receipt['videos']]}
    (a.directory / 'review_video_receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
