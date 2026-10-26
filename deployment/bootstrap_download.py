
import urllib.request,concurrent.futures,time,tarfile,json,subprocess
from pathlib import Path
root=Path('/root/gpufree-data/projects/sable-isaacgym')
tmp=Path('/root/gpufree-data/tmp/sable-isaacgym')
url='https://micro.mamba.pm/api/micromamba/linux-64/latest'
req=urllib.request.Request(url,headers={'Range':'bytes=0-1023'})
with urllib.request.urlopen(req,timeout=30) as r:
 assert r.status==206
 total=int(r.headers['Content-Range'].split('/')[-1]); final=r.url
 r.read()
print('MICROMAMBA_TOTAL',total,flush=True)
def download(i):
 start=total*i//8; end=total*(i+1)//8-1
 path=tmp/('mamba-part-%d'%i)
 for trial in range(4):
  try:
   request=urllib.request.Request(final,headers={'Range':'bytes=%d-%d'%(start,end)})
   with urllib.request.urlopen(request,timeout=90) as r:
    assert r.status==206
    data=r.read()
   assert len(data)==end-start+1
   path.write_bytes(data)
   print('PART',i,len(data),flush=True)
   return path
  except Exception:
   if trial==3: raise
   time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: parts=list(pool.map(download,range(8)))
archive=tmp/'micromamba-complete.tar.bz2'
with archive.open('wb') as out:
 for p in parts:out.write(p.read_bytes())
with tarfile.open(archive) as tar:
 member=tar.getmember('bin/micromamba')
 tar.extract(member,'/root/gpufree-data/toolchains/micromamba-sable',filter='data')
for p in parts:p.unlink()
old=root/'logs/install.exit'
if old.exists():old.unlink()
with (root/'logs/install.log').open('a') as log:
 p=subprocess.Popen(['bash',str(root/'deployment/install.sh')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=root)
(root/'logs/install.pid').write_text(str(p.pid))
print('INSTALL_RESTARTED',p.pid,flush=True)
