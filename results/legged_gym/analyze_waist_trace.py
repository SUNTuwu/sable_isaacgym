"""Offline geometry audit of saved first-environment traces; no simulator/RL."""
import json, math, hashlib, time
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
RUN=ROOT/'runs/stand5_20261025_v17_hip_posture_weight'
ASSETS=RUN/'runtime/assets'
model=json.loads((ASSETS/'physical_manifest.json').read_text())
xml=ET.parse(ASSETS/'sable.xml')
def rotation(axis, angle):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
def basis(a,y):
    x=a/np.linalg.norm(a);y=y-x*np.dot(x,y);y/=np.linalg.norm(y)
    return np.stack([x,y,np.cross(x,y)],1)
def point(name):
    return np.fromstring(xml.find(".//body[@name='%s']"%name).get('pos','0 0 0'),sep=' ')
def axis(name):
    return np.fromstring(xml.find(".//joint[@name='%s']"%name).get('axis'),sep=' ')
def pitch(R):return math.degrees(math.asin(np.clip(-R[2,0],-1,1)))
def gravity_error(R, reference):
    # Heading-invariant comparison of world gravity expressed in each body.
    return math.degrees(math.acos(np.clip(np.dot(R[2],reference[2]),-1,1)))
quat=model['init_quat_xyzw'];root_standard=rotation([0,1,0],2*math.atan2(quat[1],quat[3]))
rear_standard=root_standard@rotation(axis('spine_pitch'),model['default_angles']['spine_pitch'])@rotation(axis('spine_roll'),model['default_angles']['spine_roll'])
local_basis=basis((point('L_shoulder_roll')+point('R_shoulder_roll'))/2,point('L_shoulder_roll')-point('R_shoulder_roll'))
results=[]
for iteration in [4000,4186]:
    path=RUN/'evaluations'/('%s_%06d.json'%('final' if iteration==4186 else 'eval',iteration))
    ev=json.loads(path.read_text());names=ev['body_names'];dofs=ev['dof_names'];rows=[]
    for sample in ev['trace']:
        if sample['t']<1:continue
        positions=np.asarray(sample['body_xyz']);q=np.asarray(sample['q'])
        pos=lambda name:positions[names.index(name)]
        angle=lambda name:q[dofs.index(name)]
        R=basis((pos('L_shoulder_roll')+pos('R_shoulder_roll'))/2-pos('ROOT'),pos('L_shoulder_roll')-pos('R_shoulder_roll'))@local_basis.T
        Rwaist=R@rotation(axis('spine_pitch'),angle('spine_pitch'))
        rear=Rwaist@rotation(axis('spine_roll'),angle('spine_roll'))
        predicted=pos('ROOT')+Rwaist@point('spine_roll')
        closure=np.linalg.norm(predicted-pos('spine_roll'))
        thighs=[]
        for side in ['L','R']:
            v=pos(side+'_knee_pitch')-pos(side+'_hip_pitch')
            thighs.append(math.degrees(math.atan2(v[0],-v[2])))
        rows.append([sample['t'],pitch(R),math.degrees(angle('spine_pitch')),pitch(rear),gravity_error(R,root_standard),gravity_error(rear,rear_standard),closure,*thighs])
    values=np.asarray(rows)
    results.append({'iteration':iteration,'checkpoint_sha256':ev['checkpoint_sha256'],'trace_report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sample_count':len(rows),'median':np.median(values,axis=0).tolist(),'maximum_waist_chain_position_error_m':float(values[:,6].max())})
report={'created_unix':time.time(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'standard_root_pitch_deg':pitch(root_standard),'standard_waist_joint_deg':math.degrees(model['default_angles']['spine_pitch']),'standard_rear_pitch_deg':pitch(rear_standard),'columns':['time_s','root_pitch_deg','actual_waist_joint_deg','rear_pitch_deg','root_gravity_error_deg','rear_gravity_error_deg','waist_chain_position_error_m','left_thigh_sideview_deg','right_thigh_sideview_deg'],'results':results,'scope':'Existing first-environment (0.03m/s) traces after first1s only; independent24-environment aggregate not claimed. ROOT rotation reconstructed from fixed shoulder-roll pivot locations; rear rotation composes actual waist pitch/roll around XML axes. Position closure checks transform sign. World-X thigh sideview angles are diagnostic and include heading influence.'}
out=ROOT/'results/legged_gym/waist_trace_audit.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
