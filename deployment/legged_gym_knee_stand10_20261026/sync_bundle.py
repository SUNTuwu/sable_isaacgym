"""Remote evidence snapshot over one SSH stdout stream; no credentials or RL."""
import hashlib,io,json,sys,tarfile,time
from pathlib import Path

ROOT=Path('/root/gpufree-data/projects/sable-isaacgym/runs')
FILES=['status.json','checkpoint.json','metrics.jsonl','milestones.jsonl','latest_evaluation.json','best_evaluation.json','config.json','runtime_manifest.json','completion.json','backup_receipt.json','supervisor.json','events.jsonl','stage_intent.json']

def main():
    known=json.load(sys.stdin)
    current=json.loads((ROOT/'current.json').read_text());run=Path(current['run_dir'])
    assert run.parent==ROOT and run.is_dir()
    name=run.name;data={f:(run/f).read_bytes() for f in FILES if (run/f).is_file()}
    cp=json.loads(data['checkpoint.json']);wanted={Path(cp['path']).name}
    for line in data.get('milestones.jsonl',b'').splitlines():
        try:wanted.add(json.loads(line)['checkpoint'])
        except (ValueError,KeyError):pass
    runtime=run/'runtime.tar.gz'
    if not runtime.exists():
        tmp=run/'runtime.bundle.partial'
        with tarfile.open(str(tmp),'w:gz',compresslevel=1) as t:
            for f in ['runtime','preflight','config.json','runtime_manifest.json']:t.add(str(run/f),arcname=f)
        tmp.replace(runtime)
    send=[(runtime,name+'/runtime.tar.gz')]
    # Preserve each stage's actual evaluation and late-added preflight proof,
    # even after latest_evaluation.json and best_evaluation.json move forward.
    for folder,pattern in [('evaluations','*.json'),('preflight','*.json'),('preflight','*.tar.gz'),('diagnostics','*.py'),('diagnostics','*.json')]:
        for path in sorted((run/folder).glob(pattern)):
            if path.is_file():send.append((path,name+'/'+folder+'/'+path.name))
    for n in sorted(wanted):
        assert Path(n).name==n and n.startswith('model_') and n.endswith('.pt')
        send.append((run/'checkpoints'/n,name+'/checkpoints/'+n))
    if 'backup_receipt.json' in data:
        receipt=json.loads(data['backup_receipt.json']);archive=Path(receipt['archive'])
        assert archive.parent==ROOT and archive.name==name+'_backup.tar.gz'
        send.append((archive,name+'/'+archive.name))
    evidence={}
    for path,n in send:
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            evidence[n]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
        def add(n,b):
            info=tarfile.TarInfo(n);info.size=len(b);info.mtime=int(time.time());t.addfile(info,io.BytesIO(b))
        add('current.json',json.dumps(current).encode())
        add(name+'/bundle_receipt.json',json.dumps({'created_unix':time.time(),'files':evidence}).encode())
        for n,b in data.items():add(name+'/'+n,b)
        for path,n in send:
            if not path.is_file():continue
            # Frozen files are immutable except a checkpoint rewritten in the same update.
            if known.get(n)==evidence[n]['sha256']:continue
            t.add(str(path),arcname=n,recursive=False)

if __name__=='__main__':main()
