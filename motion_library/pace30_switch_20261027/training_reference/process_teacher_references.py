"""Create periodic, ground-clear kinematic candidates without changing authored data.

This is static reference preparation, not physical validation. Ground contacts
remain geometric estimates. Each output retains a provenance/processing record.
"""
from pathlib import Path
import json,sys,hashlib,importlib.util,time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline

ROOT=Path(sys.argv[1]).resolve()
OUT=ROOT/'processed_v1';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('sable_kinematics',ROOT/'editor/scripts/sable_kinematics.py')
km=importlib.util.module_from_spec(spec);spec.loader.exec_module(km)
MODEL=km.Kinematics(json.loads((ROOT/'editor/source/rig_snapshot.json').read_text(encoding='utf8')),
                    json.loads((ROOT/'editor/source/authored_model.json').read_text(encoding='utf8')))
catalog=json.loads((ROOT/'motion_catalog.json').read_text(encoding='utf8'))
POLICY=catalog['clips'][0]['policy_joint_names'];MAPPING=[MODEL.names.index(n) for n in POLICY]
N=50;T=1.;K=5
t=np.arange(N+1)/N


def periodic(values, trend=None):
    """Remove endpoint drift, retain 5 harmonics; analytic periodic derivative."""
    values=np.asarray(values,float)
    y=values.copy()
    if trend is None:trend=np.zeros_like(y[0])
    trend=np.asarray(trend)
    y=y-t.reshape((-1,)+(1,)*(y.ndim-1))*(y[-1]-y[0])
    c=np.fft.rfft(y[:-1],axis=0)/N;c[K+1:]=0
    # Fourier coefficients generated from the original series sample at n/N.
    angles=2*np.pi*t[:,None]*np.arange(len(c))[None,:]
    e=np.exp(1j*angles)
    shape=(len(t),len(c))+(1,)*(y.ndim-1)
    weighted=c.copy();weighted[1:-1]*=2
    if N%2:weighted[-1]*=2
    out=np.sum(e.reshape(shape)*weighted[None,...],axis=1).real
    derivative=np.sum((e*(2j*np.pi*np.arange(len(c))[None,:])).reshape(shape)*weighted[None,...],axis=1).real
    out+=t.reshape((-1,)+(1,)*(y.ndim-1))*trend
    derivative+=trend
    return out,derivative


def roots_from(euler,position):
    roots=np.repeat(np.eye(4)[None,:,:],len(t),axis=0)
    roots[:,:3,:3]=Rotation.from_euler('xyz',euler).as_matrix();roots[:,:3,3]=position
    return roots


def omega_world(roots):
    # Extend across cycle via its proper rotation, to avoid one-sided derivatives.
    rotations=roots[:,:3,:3]
    g=rotations[-1]@rotations[0].T
    extended=np.concatenate([(g.T@rotations[-2])[None],rotations,(g@rotations[1])[None]],axis=0)
    omega=[]
    for i in range(len(rotations)):
        delta=extended[i+2]@extended[i].T
        omega.append(Rotation.from_matrix(delta).as_rotvec()/(2/N))
    return np.array(omega)


def save_index(items):
    manifest={'schema':'sable-processed-reference/1','created_local':time.strftime('%Y-%m-%dT%H:%M:%S'),
              'fps':50,'sample_count':51,'period_s':1.,'joint_names':MODEL.names,
              'policy_joint_names':POLICY,'foot_order':['FL','FR','HL','HR'],
              'clips':items,'ready_ids':[c['id'] for c in items],
              'physical_validation_passed':False,
              'contact_labels':'Geometric estimates only; no simulated contact forces have been fabricated.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'quality.json').write_text(json.dumps({c['id']:c['quality'] for c in items},ensure_ascii=False,indent=2),encoding='utf8')


items=[]
order=[c for c in catalog['clips'] if c['id'] not in ('04_TURN_LEFT','05_TURN_RIGHT')]+[c for c in catalog['clips'] if c['id'] in ('04_TURN_LEFT','05_TURN_RIGHT')]
for meta in order:
    ident=meta['id'];raw=np.load(ROOT/'motions'/(ident+'.npz'),allow_pickle=False)
    q,qd=periodic(raw['joint_position'])
    # Clip authoring limits, then reproject into a periodic smooth trajectory.
    q=np.clip(q,MODEL.limits[:,0]+1e-5,MODEL.limits[:,1]-1e-5)
    q[-1]=q[0];qd=CubicSpline(t,q,bc_type='periodic')(t,1)
    assert np.max(np.maximum(MODEL.limits[:,0]-q,0)+np.maximum(q-MODEL.limits[:,1],0))<1e-4
    oldroot=raw['root_transform_world']
    euler=Rotation.from_matrix(oldroot[:,:3,:3]).as_euler('xyz');euler=np.unwrap(euler,axis=0)
    yaw_delta=euler[-1,2]-euler[0,2]
    pos_trend=oldroot[-1,:3,3]-oldroot[0,:3,3];pos_trend[2]=0.
    euler_fit,euler_dot=periodic(euler,trend=np.array([0.,0.,yaw_delta]))
    pos,vel=periodic(oldroot[:,:3,3],trend=pos_trend)
    roots=roots_from(euler_fit,pos)
    processing={'periodic_filter':'Endpoint-drift removal, five Fourier harmonics; analytic periodic q derivatives',
                'harmonics':K,'source_npz_sha256':hashlib.sha256((ROOT/'motions'/(ident+'.npz')).read_bytes()).hexdigest(),
                'root_cycle_translation_m':pos_trend.tolist(),'root_cycle_yaw_rad':float(yaw_delta)}
    target_error=None
    if ident in ('04_TURN_LEFT','05_TURN_RIGHT'):
        # Turn only ten degrees per one-second gait cycle. Compress horizontal
        # foot motion relative to the body coherently with root turning, then IK.
        ratio=np.deg2rad(10)/abs(yaw_delta)
        original_feet=np.array([MODEL.fk(qq,rr)[0] for qq,rr in zip(q,roots)])
        local=np.einsum('nki,nij->nkj',original_feet-roots[:,None,:3,3],roots[:,:3,:3])
        center=local[:-1,:,:2].mean(axis=0)
        compressed=local.copy();compressed[:,:,:2]=center[None]+ratio*(local[:,:,:2]-center[None])
        euler_fit[:,2]=euler_fit[0,2]+ratio*(euler_fit[:,2]-euler_fit[0,2])
        pos[:,:2]=pos[0,:2]+ratio*(pos[:,:2]-pos[0,:2]-t[:,None]*pos_trend[None,:2])
        # The new turn is in place; suppress the source's incidental drift.
        pos_trend[:2]=0.;roots=roots_from(euler_fit,pos)
        desired=np.einsum('nkj,nij->nki',compressed,roots[:,:3,:3])+roots[:,None,:3,3]
        solved=[];errors=[]
        for i in range(N):
            answer,error=MODEL.solve(desired[i],roots[i],q[i],q[i],iterations=35,regularization=.00025)
            solved.append(answer);errors.append(error)
        solved=np.array(solved+[solved[0]])
        q,qd=periodic(solved)
        q=np.clip(q,MODEL.limits[:,0]+1e-5,MODEL.limits[:,1]-1e-5)
        q[-1]=q[0];qd=CubicSpline(t,q,bc_type='periodic')(t,1)
        target_error=float(np.max(np.linalg.norm(np.array([MODEL.fk(qq,rr)[0] for qq,rr in zip(q,roots)])-desired,axis=2))*1000)
        processing['turn_variant']={'original_yaw_rad':float(yaw_delta),'desired_yaw_rad':float(np.sign(yaw_delta)*np.deg2rad(10)),
                                    'horizontal_body_relative_foot_scale':float(ratio),'root_translation':'in_place',
                                    'max_reprojection_error_mm':target_error}
    # Raise the complete body, preserving FK. A smooth upper envelope of the
    # penetration correction avoids discrete root jumps at support-leg changes.
    surface=np.array([MODEL.surface_z(qq,rr) for qq,rr in zip(q,roots)])
    lift_needed=np.maximum(.00025-surface.min(axis=1),0.)
    lift,liftdot=periodic(lift_needed[:,None]);lift=lift[:,0];liftdot=liftdot[:,0]
    lift+=max(float(np.max(lift_needed-lift)),0.)+.00005
    roots[:,2,3]+=lift
    # Derive root velocities from the final periodic position path, not raw data.
    final_pos=roots[:,:3,3]
    _,rootvel=periodic(final_pos,trend=final_pos[-1]-final_pos[0])
    feet=np.array([MODEL.fk(qq,rr)[0] for qq,rr in zip(q,roots)])
    surface=np.array([MODEL.surface_z(qq,rr) for qq,rr in zip(q,roots)])
    quat=Rotation.from_matrix(roots[:,:3,:3]).as_quat()[:,[3,0,1,2]]
    for i in range(1,len(quat)):
        if np.dot(quat[i-1],quat[i])<0:quat[i]*=-1
    omega=omega_world(roots)
    contact=surface<=.003
    processing['ground_correction']={'method':'Smooth periodic vertical-root upper envelope; joint FK preserved',
                                     'margin_m':.00025,'extra_margin_m':.00005,'maximum_root_raise_m':float(lift.max()),
                                     'estimated_contact_height_threshold_m':.003}
    # The pose fit is exactly periodic in joint space; world endpoints retain
    # translation/yaw cycles. Report velocity closure analytically as exported.
    quality={'minimum_capsule_z_mm':float(surface.min()*1000),
             'max_joint_pose_seam_deg':float(np.rad2deg(np.max(abs(q[-1]-q[0])))),
             'max_joint_velocity_seam_rad_s':float(np.max(abs(qd[-1]-qd[0]))),
             'max_joint_limit_violation_deg':float(np.rad2deg(np.max(np.maximum(MODEL.limits[:,0]-q,0)+np.maximum(q-MODEL.limits[:,1],0)))),
             'max_joint_change_from_authored_deg':float(np.rad2deg(np.max(abs(q-raw['joint_position'])))),
             'max_joint_velocity_rad_s':float(abs(qd).max()),'turn_target_reprojection_error_mm':target_error,
             'max_root_raise_mm':float(lift.max()*1000),
             'mean_geometric_contact_count':float(contact.mean(axis=0).sum()),
             'physical_validation_passed':False}
    assert quality['minimum_capsule_z_mm']>=.249-1e-6
    assert quality['max_joint_pose_seam_deg']<1e-5 and quality['max_joint_velocity_seam_rad_s']<1e-5
    assert quality['max_joint_limit_violation_deg']<.01
    np.savez_compressed(OUT/(ident+'.npz'),time_s=t,joint_position=q,joint_velocity=qd,
                        joint_position_policy_order=q[:,MAPPING],joint_velocity_policy_order=qd[:,MAPPING],
                        root_transform_world=roots,root_position_world=roots[:,:3,3],root_quaternion_wxyz=quat,
                        root_linear_velocity_world=rootvel,root_angular_velocity_world=omega,
                        toe_actual_world=feet,toe_target_world=feet,foot_capsule_min_z=surface,
                        estimated_contact_mask=contact)
    record={'id':ident,'file':ident+'.npz','title':meta['title'],'fps':50,'samples':51,'period_s':1.,
            'joint_names':MODEL.names,'policy_joint_names':POLICY,'foot_order':['FL','FR','HL','HR'],
            'source_metadata':'../motions/'+ident+'.json','processing':processing,'quality':quality,
            'native_command_world_mean':(roots[-1,:3,3]-roots[0,:3,3]).tolist(),
            'native_yaw_rate_rad_s':float(np.arctan2((roots[-1,:3,:3]@roots[0,:3,:3].T)[1,0],(roots[-1,:3,:3]@roots[0,:3,:3].T)[0,0]))}
    (OUT/(ident+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    items.append(record);save_index(items)
    print('READY',ident,json.dumps(quality),flush=True)
print('PROCESSED_COMPLETE',len(items),str(OUT),flush=True)
