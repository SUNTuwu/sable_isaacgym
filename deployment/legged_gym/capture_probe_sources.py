"""Reconstruct only hash-verifiable diagnostic sources; never infer missing files."""
from pathlib import Path
from run_io import sha256,read_json
code=Path(__file__).resolve().parent;root=code.parent.parent
names=['train.py','sable_env.py','sable_config.py','gait_support.py','guided_reference.py','evaluate.py','assets/gait_reference.json','assets/sable.xml','assets/physical_manifest.json']
for name in ['guided_transfer_smoke_v1','guided_transfer_smoke_v2']:
    run=root/'runs'/name;manifest=read_json(run/'runtime_manifest.json')['files'];dest=run/'tested_sources';dest.mkdir(exist_ok=True)
    for n in names:
        source=code/n;data=source.read_bytes();expected=manifest[n]
        if sha256(source)!=expected and name.endswith('v1'):
            text=data.decode('utf8')
            if n=='train.py':text=text.replace('cfg.reference.enabled=True;cfg.reference.strength=.25;', 'cfg.reference.enabled=True;')
            if n=='sable_config.py':text=text.replace(';strength=1.', '')
            if n=='guided_reference.py':text=text.replace('return x*x*(3-2*x)*self.env.cfg.reference.strength', 'return x*x*(3-2*x)')
            data=text.encode('utf8')
        import hashlib
        assert hashlib.sha256(data).hexdigest()==expected,(name,n,'Cannot recover exact diagnostic source')
        target=dest/n;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
    print('EXACT_PROBE_SOURCES_VERIFIED',name,flush=True)
