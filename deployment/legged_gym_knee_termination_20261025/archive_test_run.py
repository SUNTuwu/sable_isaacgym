"""Freeze hash-matched test code/assets and archive a completed remote test."""
import argparse,time,tarfile
from pathlib import Path
from run_io import sha256,read_json,atomic_json
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
p.add_argument('--reports',type=Path,nargs='*',default=[]);a=p.parse_args()
code=Path(__file__).resolve().parent;root=code.parent.parent;run=a.run.resolve();out=a.archive.resolve()
assert run.parent==root/'runs' and out.parent==code/'verification'
assert not out.exists(),'Refusing overwrite of existing evidence archive'
assert read_json(run/'status.json')['status']=='iterations_complete'
manifest=read_json(run/'runtime_manifest.json');dest=run/'tested_sources';dest.mkdir(exist_ok=True);count=0
for name,expected in manifest['files'].items():
    if not (name.startswith('assets/') or ('/' not in name and name.endswith('.py'))):continue
    source=code/name;assert sha256(source)==expected,(name,'Actual tested source has changed')
    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha256(target)==expected
    else:target.write_bytes(source.read_bytes())
    count+=1
for report in a.reports:assert report.resolve().parent==code/'verification' and report.is_file()
with tarfile.open(out,'w:gz',compresslevel=1) as tar:
    tar.add(run,arcname=run.name)
    for report in a.reports:tar.add(report,arcname=report.name)
receipt={'archive':str(out),'sha256':sha256(out),'bytes':out.stat().st_size,'exact_source_files':count,'created_unix':time.time()}
atomic_json(out.with_name(out.name+'.receipt.json'),receipt);print('TEST_RUN_ARCHIVE',receipt)
