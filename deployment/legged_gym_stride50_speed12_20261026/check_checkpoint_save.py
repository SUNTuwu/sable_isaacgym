"""Verify evaluated checkpoint bytes survive a redundant upstream save."""
import ast,hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from run_io import atomic_json,sha256
HERE=Path(__file__).resolve().parent
tree=ast.parse((HERE/'train.py').read_text())
runner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AuditedRunner')
method=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='save')
module=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(module);namespace={};exec(compile(module,str(HERE/'train.py'),'exec'),namespace)
with tempfile.TemporaryDirectory(dir='/root/gpufree-data/tmp',prefix='checkpoint_preservation_') as folder:
    root=Path(folder);(root/'checkpoints').mkdir();path=root/'checkpoints/model_000051.pt'
    path.write_bytes(b'checkpoint-already-evaluated');before=sha256(path);calls=[]
    def save_new(reason):
        calls.append(reason);target=root/'checkpoints/model_000052.pt';target.write_bytes(b'new-checkpoint');return target
    owner=SimpleNamespace(out=root,completed=51,save_checkpoint=save_new)
    assert namespace['save'](owner,'upstream-ignored')==path and sha256(path)==before and not calls
    owner.completed=52;created=namespace['save'](owner,'upstream-ignored');assert created.read_bytes()==b'new-checkpoint' and calls==['upstream']
report={'passed':True,'existing_evaluated_checkpoint_bytes_preserved':True,'new_iteration_saved':True,'source_sha256':{'train.py':sha256(HERE/'train.py'),'check_checkpoint_save.py':sha256(__file__)},'scope':'File preservation regression only; one real PPO and migration tests still required by launcher.'}
atomic_json(HERE/'verification/checkpoint_preservation.json',report);print(json.dumps(report))
