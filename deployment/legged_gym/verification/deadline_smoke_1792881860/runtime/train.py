import argparse,signal,time,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--deadline');p.add_argument('--phase');a=p.parse_args()
stop=False
def finish(*args):
 global stop
 stop=True
signal.signal(signal.SIGTERM,finish)
while not stop:time.sleep(.05)
(a.out/'final_saved.txt').write_text('graceful save')
(a.out/'status.json').write_text(json.dumps({'status':'deadline_test_complete'}))
