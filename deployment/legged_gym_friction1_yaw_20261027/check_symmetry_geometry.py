"""Static FK/inertial reflection check only. No simulator or training."""
import argparse,json,time
from pathlib import Path
import numpy as np
from guided_reference import CapsuleKinematics,quaternion,rotation
from policy_symmetry import NAMES,mirrored_coordinates,numpy_involution_error
from run_io import atomic_json,sha256
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
code=Path(__file__).resolve().parent;asset=code/'assets/sable.xml';manifest=code/'assets/physical_manifest.json'
m=json.loads(manifest.read_text());assert m['policy_joint_names']==NAMES;k=CapsuleKinematics(asset,m)
S=np.diag([1.,-1.,1.]);foot_index=[1,0,3,2]
def mass_properties(q):
    mass=0.;first=np.zeros(3);inertia=np.zeros((3,3))
    def visit(b,pos,r):
        nonlocal mass,first,inertia
        pos=pos+r@np.fromstring(b.get('pos','0 0 0'),sep=' ')
        if b.get('quat'):
            w,x,y,z=np.fromstring(b.get('quat'),sep=' ');r=r@quaternion([x,y,z,w])
        for j in b.findall('joint'):
            if j.get('name') not in q:continue
            pivot=np.fromstring(j.get('pos','0 0 0'),sep=' ');rr=rotation(np.fromstring(j.get('axis'),sep=' '),q[j.get('name')])
            pos=pos+r@(pivot-rr@pivot);r=r@rr
        prop=m['masses'][b.get('name')];w=prop['mass_kg'];c=pos+r@np.asarray(prop['com_local_m'])
        mass+=w;first+=w*c;inertia+=r@np.asarray(prop['inertia'])@r.T+w*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        for child in b.findall('body'):visit(child,pos.copy(),r.copy())
    visit(k.root,np.asarray(m['init_position']),quaternion(m['init_quat_xyzw']))
    return first/mass,inertia
rng=np.random.default_rng(947);feet_error=0.;com_error=0.;inertia_error=0.;limits_ok=True
for sample in range(100):
    q={n:float(np.clip(v+rng.uniform(-.15,.15),*k.limits[n])) for n,v in k.q0.items()}
    q['R_hip_roll_mimic']=-q['hip_roll'];q['spine_pitch']=k.q0['spine_pitch'];mq=mirrored_coordinates(q)
    feet_error=max(feet_error,float(np.max(np.abs(k.fk(mq)-k.fk(q)[foot_index]@S))))
    c,i=mass_properties(q);mc,mi=mass_properties(mq)
    com_error=max(com_error,float(np.max(np.abs(mc-S@c))));inertia_error=max(inertia_error,float(np.max(np.abs(mi-S@i@S))))
    limits_ok &= all(k.limits[n][0]-1e-7<=v<=k.limits[n][1]+1e-7 for n,v in mq.items())
table=np.asarray(json.loads((code/'assets/gait_reference.json').read_text())['deltas_rad'])
table_error=float(np.max(np.abs(table-table[:,:,foot_index,:])))
involutions={str(n):numpy_involution_error(n) for n in [54,56,58]}
passed=feet_error<1e-6 and com_error<1e-6 and inertia_error<1e-8 and limits_ok and table_error<1e-5 and max(involutions.values())==0
report={'passed':passed,'samples':100,'seed':947,'max_capsule_point_reflection_error_m':feet_error,
 'max_total_com_reflection_error_m':com_error,'max_world_inertia_reflection_error_kg_m2':inertia_error,
 'reference_table_pair_error_rad':table_error,'mirrored_joint_limits_passed':bool(limits_ok),'observation_involution_errors':involutions,
 'asset_sha256':sha256(asset),'manifest_sha256':sha256(manifest),'mirror_source_sha256':sha256(code/'policy_symmetry.py'),
 'checker_sha256':sha256(__file__),'created_unix':time.time(),
 'scope':'Static kinematic/inertial symmetry for current model and coupled coordinates only. Does not prove exact PhysX dynamics/tendon symmetry, stability or policy benefit.'}
atomic_json(a.output,report);print(json.dumps(report,indent=2))
if not passed:raise SystemExit(2)
