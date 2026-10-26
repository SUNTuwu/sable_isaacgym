"""Construct a fresh MJCF physical asset from the current authored rig only."""
import json,math,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
OUT=HERE/'assets';MODEL=json.loads((OUT/'authored_model.json').read_text(encoding='utf8'))
JOINTS={r['name']:r for r in MODEL['joints']};POS={n:np.array(r['zero_position_world_m']) for n,r in JOINTS.items()}
def fmt(x):return ' '.join('%.10g'%v for v in np.atleast_1d(x))
def quat_z(axis):
    axis=np.array(axis)/np.linalg.norm(axis);z=np.array([0,0,1.]);dot=np.dot(z,axis)
    if dot<-1+1e-9:q=np.array([1,0,0,0.])
    else:q=np.r_[np.cross(z,axis),1+dot];q/=np.linalg.norm(q)
    return fmt(q[[3,0,1,2]])
root=ET.Element('mujoco',model='sable_stand5_v1')
ET.SubElement(root,'compiler',angle='radian',meshdir='meshes',inertiafromgeom='false',fusestatic='false')
ET.SubElement(root,'option',timestep='.001',gravity='0 0 -9.81')
d=ET.SubElement(root,'default');ET.SubElement(d,'joint',limited='true',damping='.001',armature='.000015')
ET.SubElement(d,'geom',density='0',contype='1',conaffinity='1',friction='1 .005 .0001',margin='.0002')
asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');bodies={};massparts={n:[] for n in JOINTS}
for row in MODEL['joints']:
    n=row['name'];parent=row['parent'];p=POS[n]-(POS[parent] if parent else POS[n])
    body=ET.SubElement(bodies[parent] if parent else world,'body',name=n,pos=fmt(p));bodies[n]=body
    if n=='ROOT':ET.SubElement(body,'freejoint',name='floating_base')
    elif row['kind']=='revolute':ET.SubElement(body,'joint',name=n,type='hinge',axis=fmt(row['zero_axis_world']),range=fmt(row['limits_rad']))

def add_mass(bone,mass,center,size):
    size=np.maximum(np.asarray(size),.001)
    inertia=mass/12*np.array([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
    massparts[bone].append((mass,np.asarray(center)-POS[bone],np.diag(inertia)))

shape_metadata={};motor_specs={}
srcpath=HERE.parents[1]/'robot_authoring/inspection/source_inventory.json'
source={o['name']:o for o in json.loads(srcpath.read_text(encoding='utf8'))['objects']}
(OUT/'meshes').mkdir(exist_ok=True)
for mesh in MODEL['meshes']:
    n,bone,key=mesh['object'],mesh['bone'],mesh['source_key'];vv=np.array(mesh['vertices']);local=vv-POS[bone]
    if key in ['hand','foot']:mass=.00165*(1.375 if key=='foot' else 1.)
    elif key in ['wrist_fixed','ankle_fixed']:mass=.010
    elif key in ['shoulder_pitch','hip_pitch']:mass=.097
    elif key=='hip_roll':mass=.015 # two visual outputs, ONE 30 g motor
    else:mass=.030
    add_mass(bone,mass,vv.mean(0),np.ptp(vv,axis=0))
    objpath=OUT/'meshes'/f'{n}.obj'
    with objpath.open('w',encoding='ascii') as f:
        for v in local:f.write('v '+fmt(v)+'\n')
        for face in mesh['faces']:
            for i in range(1,len(face)-1):f.write('f %d %d %d\n'%(face[0]+1,face[i]+1,face[i+1]+1))
    ET.SubElement(asset,'mesh',name=n,file=objpath.name)
    color='.75 .49 .22 1' if n.startswith('L_') or not n.startswith('R_') else '.35 .52 .65 1'
    ET.SubElement(bodies[bone],'geom',name=n,type='mesh',mesh=n,contype='0',conaffinity='0',group='1',rgba=color)
    # Initial flat-ground curriculum uses the terminal capsules as the only
    # wrist/ankle collision shapes; block visuals and their inertia are retained.
    if key in ['hand','foot','wrist_fixed','ankle_fixed']:continue
    # Convex primitives fitted to named components, with exact visual meshes separate.
    for i,comp in enumerate(source[mesh['source_object']]['components']):
        pts=vv[comp['indices']];center=pts.mean(0);cov=np.cov(pts.T);_,basis=np.linalg.eigh(cov)
        name=n+'_collision_'+str(i)
        if comp['count'] in [36,64,144]:
            axis=basis[:,0] if comp['count'] in [36,64] else basis[:,-1]
            rel=pts-center;projection=rel@axis
            radius=float(np.max(np.linalg.norm(rel-projection[:,None]*axis,axis=1)))
            half=float(np.ptp(projection)/2)
            geom=dict(type='cylinder',pos=fmt(center-POS[bone]),quat=quat_z(axis),size=fmt([radius,half]))
        else:
            # One oriented box per named block component, not duplicate hulls
            # of the complete multi-component object.
            if np.linalg.det(basis)<0:basis[:,0]*=-1
            projection=(pts-center)@basis
            q=Rotation.from_matrix(basis).as_quat()
            geom=dict(type='box',pos=fmt(center-POS[bone]),quat=fmt(q[[3,0,1,2]]),size=fmt(np.maximum(np.ptp(projection,axis=0)/2,.0001)))
        ET.SubElement(bodies[bone],'geom',name=name,rgba='.2 .5 .5 0',group='3',**geom)
        shape_metadata[name]={'bone':bone,'foot':False,'floor':key not in ['wrist_fixed','ankle_fixed']}

for c in MODEL['capsules']:
    b=c['bone'];center=np.array(c['center_world_zero_m']);axis=np.array(c['axis_world_zero'])
    a=center-axis*c['cylinder_length_m']/2-POS[b];z=center+axis*c['cylinder_length_m']/2-POS[b]
    ET.SubElement(bodies[b],'geom',name=c['name'],type='capsule',fromto=fmt(np.r_[a,z]),size=str(c['radius_m']),rgba='.1 .6 .5 1',priority='1')
    shape_metadata[c['name']]={'bone':b,'foot':True,'floor':True}

# User-requested fixed head/tail payloads; no head/tail collision or actuator.
payloads=[]
for label,bone,mass,reference,dx in [('head','ROOT',.050,'shoulder_pitch',.030),('tail','spine_roll',.030,'hip_pitch',-.030)]:
    center=(POS['L_'+reference]+POS['R_'+reference])/2+np.array([dx,0,0])
    add_mass(bone,mass,center,[.010,.010,.010])
    payloads.append({'name':label,'bone':bone,'mass_kg':mass,'center_world_zero_m':center.tolist(),
        'reference':reference,'offset_forward_m':dx,'collision':False,'inertia_assumption':'10 mm cube about specified mass center'})
# Fresh declared mass/collision approximation; no historical body, head, tail or policy is imported.
# Total 1.5 kg is provisional, not measured; motor masses retain their named hardware values.
fixed_mass=sum(p[0] for parts in massparts.values() for p in parts)
for n in JOINTS:add_mass(n,.001,POS[n],[.004,.004,.004])
remaining=1.5412375-fixed_mass-.001*len(JOINTS)
for bone,mass,offset,size in [('ROOT',remaining*.65,[.045,0,-.012],[.09,.035,.025]),('spine_pitch',remaining*.35,[-.030,0,-.008],[.06,.035,.022])]:
    center=POS[bone]+offset;add_mass(bone,mass,center,size)
    name=bone+'_torso_collision'
    ET.SubElement(bodies[bone],'geom',name=name,type='box',pos=fmt(offset),size=fmt(np.array(size)/2),rgba='.25 .3 .35 1')
    shape_metadata[name]={'bone':bone,'foot':False,'floor':True}
budget={}
for n,parts in massparts.items():
    mass=sum(x[0] for x in parts);center=sum(m*p for m,p,_ in parts)/mass
    inertia=sum(I+m*((np.dot(p-center,p-center))*np.eye(3)-np.outer(p-center,p-center)) for m,p,I in parts)
    ev=np.linalg.eigvalsh(inertia);assert np.min(ev)>0 and ev.max()<=ev.sum()/2+1e-10
    inert=ET.Element('inertial',mass=str(mass),pos=fmt(center),fullinertia=fmt([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
    bodies[n].insert(0,inert);budget[n]={'mass_kg':mass,'com_local_m':center.tolist(),'inertia':inertia.tolist()}
tendon=ET.SubElement(ET.SubElement(root,'tendon'),'fixed',name='opposed_hip_roll',limited='true',range='-.0001 .0001',stiffness='1000',damping='1')
ET.SubElement(tendon,'joint',joint='hip_roll',coef='1');ET.SubElement(tendon,'joint',joint='R_hip_roll_mimic',coef='1')
act=ET.SubElement(root,'actuator')
for row in MODEL['joints']:
    if not row['actuated']:continue
    n=row['name'];large=n.endswith(('shoulder_pitch','hip_pitch'));ratio=38/15 if n=='spine_roll' else 1
    cap=1.95 if large else ratio*(.85 if ratio!=1 else 1);speed=(421.3 if large else 120)/ratio
    motor_specs[n]={'model':'3505' if large else 'dc1628','stall_torque_nm':cap,'no_load_rpm':speed,
        'rated_rpm':245.1 if large else None,'rated_torque_nm':.78 if large else None,'ratio':ratio,
        'parameter_status':'Prior named-hardware values reused; gearbox inertia and thermal limits unmeasured'}
    ET.SubElement(act,'motor',name=n,joint=n,gear='1',ctrllimited='true',ctrlrange=fmt([-cap,cap]))
if hasattr(ET,'indent'):ET.indent(root)
ET.ElementTree(root).write(OUT/'sable.xml',encoding='utf-8',xml_declaration=True)

# Independent analytic FK, compared to evaluated Blender transforms at the standard pose.
transforms={};errors={}
for row in MODEL['joints']:
    n,p=row['name'],row['parent']
    if n=='ROOT':T=np.array(MODEL['root_matrix_world'])
    else:
        T=np.eye(4);T[:3,3]=POS[n]-POS[p]
        if row['kind']=='revolute':T[:3,:3]=Rotation.from_rotvec(np.array(row['zero_axis_world'])*MODEL['angles_rad'][n]).as_matrix()
        T=transforms[p]@T
    transforms[n]=T
    errors[n]=float(np.linalg.norm(T[:3,3]-np.array(MODEL['bone_world_at_stand'][n])[:3,3]))
assert max(errors.values())<1e-6,errors
footlow=[]
for c in MODEL['capsules']:
    b=c['bone'];a=np.array(c['center_world_zero_m'])-np.array(c['axis_world_zero'])*c['cylinder_length_m']/2-POS[b]
    z=np.array(c['center_world_zero_m'])+np.array(c['axis_world_zero'])*c['cylinder_length_m']/2-POS[b]
    T=transforms[b];worldpoints=np.array([a,z])@T[:3,:3].T+T[:3,3]
    footlow.append(float(worldpoints[:,2].min()-c['radius_m']))
rootstate=np.array(MODEL['root_matrix_world']);basepos=rootstate[:3,3].copy();basepos[2]-=min(footlow)-.001
manifest={'schema':'sable-isaacgym-physical/1','authored_sha256':MODEL['rig_sha256'],'frame':5,
    'default_angles':MODEL['angles_rad'],'init_position':basepos.tolist(),'init_quat_xyzw':Rotation.from_matrix(rootstate[:3,:3]).as_quat().tolist(),
    'foot_min_z_at_authored_pose':footlow,'foot_height_spread_m':max(footlow)-min(footlow),
    'independent_fk_max_position_error_m':max(errors.values()),'masses':budget,'total_mass_kg':sum(v['mass_kg'] for v in budget.values()),
    'motors':motor_specs,'payloads':payloads,'shape_metadata':shape_metadata,
    'collision_shapes_by_body':{n:[g.attrib['name'] for g in b.findall('geom') if g.attrib.get('contype','1')!='0'] for n,b in bodies.items()},
    'foot_bodies':['L_wrist_fixed','R_wrist_fixed','L_ankle_pitch','R_ankle_pitch'],
    'policy_joint_names':[r['name'] for r in MODEL['joints'] if r['actuated'] and r['name']!='spine_pitch' and 'ankle' not in r['name']]+['L_ankle_pitch','R_ankle_pitch'],
    'held_joint':'spine_pitch','held_joint_angle_rad':math.radians(-40),'physical_independent_actuators':15,
    'dependent_joint':'R_hip_roll_mimic','transmission':'q_right=-q_left; one hip actuator, passive fixed tendon; requires Isaac Gym validation',
    'assumptions':['1.5412375 kg provisional total: prior body mass unchanged, two 10g ankle blocks replaced by 30g motors and rear pad volume scaled 1.375x','Named motor mass: 3505 97g, DC1628 30g; hip-roll motor counted once',
       'User head 50g and tail 30g included within 1.5kg total, at shoulder +30mm and hip -30mm; no head/tail collider or actuator','Motor torque-speed interpolation retained from named hardware records; no measured thermal calibration',
       'Rubber-pad dry-floor friction nominal 1.0; foot-only 0.8..1.2 randomization','Sim-only model; physical import validation pending']}
manifest['assumptions'].append('Initial flat-ground curriculum: wrist/ankle blocks retain visual geometry and inertia but no collision; four capsules (new rear feet 40mm) provide terminal contact. Other named motor collisions retained.')
(OUT/'physical_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:manifest[k] for k in ['total_mass_kg','foot_height_spread_m','independent_fk_max_position_error_m','init_position','init_quat_xyzw']},indent=2))
