"""CPU-only capsule FK/limits and applied-height check for the new reference."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from guided_reference import CapsuleKinematics,PAIRS

HERE=Path(__file__).resolve().parent
asset=HERE/'assets/sable.xml';manifest=HERE/'assets/physical_manifest.json'
table_path=HERE/'assets/gait_reference.json';data=json.loads(table_path.read_text())
parent=json.loads((HERE/'assets/gait_reference_v23_parent.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert data['source_asset_sha256']==sha(asset) and data['source_manifest_sha256']==sha(manifest)
assert data['builder_sha256']==sha(HERE/'guided_reference.py')
kin=CapsuleKinematics(asset,json.loads(manifest.read_text()));table=np.array(data['deltas_rad'])
assert np.max(np.abs(table[:,:,:2]-np.array(parent['deltas_rad'])[:,:,:2]))<1e-10
applied_strength=np.array([.75,.75,1.,1.]);min_margin=math.inf;rows=[]
for si,speed in enumerate(data['speeds_m_s']):
    peaks=np.zeros(4)
    for pi in range(table.shape[1]):
        q=dict(kin.q0)
        for foot,pair in enumerate(PAIRS):
            for j,name in enumerate(pair):
                q[name]+=float(table[si,pi,foot,j]*applied_strength[foot])
                min_margin=min(min_margin,q[name]-kin.limits[name][0],kin.limits[name][1]-q[name])
        peaks=np.maximum(peaks,kin.fk(q)[:,2]-kin.base[:,2])
    if speed>0:assert (peaks[2:]>.020).all(),(speed,peaks)
    rows.append({'command_m_s':speed,'applied_capsule_peak_m':peaks.tolist()})
assert min_margin>0
report={'passed':True,'created_unix':time.time(),'source_sha256':sha(Path(__file__)),
        'reference_sha256':sha(table_path),'front_reference_preserved_exactly':True,
        'applied_strength_per_foot':applied_strength.tolist(),'rows':rows,
        'minimum_applied_limit_margin_deg':math.degrees(min_margin),
        'scope':'Fixed-root kinematic feasibility only, not free-root dynamic stability or learning.'}
(HERE/'verification/hind20_reference_static.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
