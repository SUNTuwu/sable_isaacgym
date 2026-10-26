"""Resume a verified PyTorch wheel download from the official index's CDN."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

URL='https://download-r2.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp38-cp38-linux_x86_64.whl'
SHA256='5f0a8085343b55935052f85447f4649641b45cd07fe940023aef4d8f6a7c4c65'
ROOT=Path('/root/gpufree-data/cache/sable-isaacgym/wheels')
PARTS=ROOT/'torch-download-parts'
NAME='torch-2.1.2+cu118-cp38-cp38-linux_x86_64.whl'

def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
    return digest.hexdigest()

def main():
    ROOT.mkdir(parents=True,exist_ok=True); PARTS.mkdir(exist_ok=True)
    target=ROOT/NAME
    if target.is_file() and checksum(target)==SHA256:
        print('VERIFIED_WHEEL_ALREADY_PRESENT',flush=True); return
    with urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':'bytes=0-0'}),timeout=30) as response:
        assert response.status==206
        total=int(response.headers['Content-Range'].split('/')[-1])
        response.read()
    block=8*1024*1024; count=(total+block-1)//block
    print(json.dumps({'total_bytes':total,'parts':count,'connections':16}),flush=True)
    started=time.monotonic(); finished=0
    def get_part(index):
        start=index*block; end=min(total,start+block)-1
        dest=PARTS/('%04d.part'%index)
        if dest.is_file() and dest.stat().st_size==end-start+1:return dest
        for attempt in range(6):
            try:
                req=urllib.request.Request(URL,headers={'Range':'bytes=%d-%d'%(start,end)})
                now=time.monotonic()
                with urllib.request.urlopen(req,timeout=30) as response, dest.with_suffix('.pending').open('wb') as out:
                    assert response.status==206
                    assert response.headers['Content-Range'].startswith('bytes %d-%d/'%(start,end))
                    while True:
                        chunk=response.read(65536)
                        if not chunk:break
                        out.write(chunk)
                        if time.monotonic()-now>180:raise TimeoutError('Segment transfer deadline')
                assert dest.with_suffix('.pending').stat().st_size==end-start+1
                dest.with_suffix('.pending').replace(dest)
                return dest
            except Exception:
                if attempt==5:raise
                time.sleep(2+attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(get_part,i) for i in range(count)]
        for future in concurrent.futures.as_completed(futures):
            future.result(); finished+=1
            if finished%10==0 or finished==count:
                print(json.dumps({'completed_parts':finished,'total_parts':count,'elapsed_s':round(time.monotonic()-started)}),flush=True)
    temporary=ROOT/(NAME+'.pending')
    with temporary.open('wb') as out:
        for index in range(count):
            with (PARTS/('%04d.part'%index)).open('rb') as stream:
                for data in iter(lambda:stream.read(1024*1024),b''):out.write(data)
    actual=checksum(temporary)
    assert actual==SHA256, 'Downloaded wheel hash does not match the official PyTorch index'
    temporary.replace(target)
    for path in PARTS.iterdir():
        if path.is_file():path.unlink()
    PARTS.rmdir()
    (ROOT/'torch-source.json').write_text(json.dumps({'url':URL,'sha256':actual,'bytes':total},indent=2))
    print('TORCH_DOWNLOAD_VERIFIED',total,actual,flush=True)

if __name__=='__main__':main()
