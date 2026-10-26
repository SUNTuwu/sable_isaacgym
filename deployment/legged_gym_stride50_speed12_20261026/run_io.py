"""Atomic metadata shared by training, independent evaluation and the dashboard."""
import hashlib,json,os,time
from pathlib import Path

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(str(tmp),str(path))

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def append_json(path,value):
    with Path(path).open('a',encoding='utf8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()

def read_json(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf8'))
    except (OSError,ValueError):return default
