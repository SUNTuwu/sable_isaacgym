"""Validate file hashes and numeric data using NumPy; no simulation or training."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]).resolve()
inventory=json.loads((root/'FILES_SHA256.json').read_text(encoding='utf8'))
for name,record in inventory['files'].items():
    p=(root/name).resolve()
    assert root in p.parents and p.is_file(),name
    assert p.stat().st_size==record['bytes'],name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],name
catalog=json.loads((root/'motion_catalog.json').read_text(encoding='utf8'))
assert len(catalog['clips'])==10
for c in catalog['clips']:
    d=np.load(root/'motions'/(c['id']+'.npz'),allow_pickle=False)
    assert d['joint_position'].shape==(c['samples'],15)
    assert d['root_transform_world'].shape==(c['samples'],4,4)
    assert d['toe_actual_world'].shape==(c['samples'],4,3)
    assert np.all(np.diff(d['time_s'])>0)
    assert all(np.isfinite(d[n]).all() for n in d.files)
    assert np.allclose(d['joint_position_policy_order'],d['joint_position'][:,c['policy_from_motion_indices']])
    assert np.allclose(np.linalg.norm(d['root_quaternion_wxyz'],axis=1),1,atol=1e-6)
print(json.dumps({'passed':True,'files_verified':len(inventory['files']),'clips':len(catalog['clips']),
                  'training_ready':False,'scope':'Hash, shape, finite-value, quaternion and joint-order checks only'}))
