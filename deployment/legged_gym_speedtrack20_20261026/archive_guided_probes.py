from pathlib import Path
import tarfile,time
from run_io import sha256,atomic_json
code=Path(__file__).resolve().parent;root=code.parent.parent
out=code/'verification/guided_prototype_evidence.tar.gz'
assert not out.exists(),'Refusing overwrite of diagnostic archive'
with tarfile.open(out,'w:gz',compresslevel=1) as tar:
    for name in ['guided_transfer_smoke_v1','guided_transfer_smoke_v2']:
        tar.add(root/'runs'/name,arcname=name)
    tar.add(code/'probe_reference_strength.py',arcname='probe_reference_strength.py')
    for name in ['guided_reference_only.json','guided_parent_reference.json','guided_quarter_reference_only.json','guided_quarter_parent.json','reference_strength_probe.json']:
        tar.add(code/'verification'/name,arcname=name)
atomic_json(code/'verification/guided_prototype_receipt.json',{'archive':str(out),'bytes':out.stat().st_size,'sha256':sha256(out),'created_unix':time.time()})
print('PROTOTYPE_ARCHIVE',out.stat().st_size,sha256(out))
