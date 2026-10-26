"""Build a new robot-authoring .blend. Never saves over the input source.

Run: blender --background --factory-startup --disable-autoexec --threads 2
     blender/draft.blend --python robot_authoring/build_rig.py
"""
import bpy
import hashlib
import json
import math
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'robot_authoring'))
from capsule_geometry import TERMINALS, replace_terminal
OUT = ROOT / 'blender' / 'sable_rig.blend'
AUDIT = ROOT / 'robot_authoring' / 'inspection'
SOURCE = ROOT / 'robot_authoring/source/original_draft_2026-10-24.blend'
ALLOWLIST = {
    'shoulder_roll': '肩膀roll-dc1628', 'shoulder_pitch': '肩膀pitch-3505',
    'elbow_pitch': '肘关节pitch-dc1628', 'wrist_fixed': '手腕pitch-步进丝杆-仿真暂时固定',
    'hand': '手', 'spine_pitch': '腰pitch-dc1628', 'spine_roll': '腰roll-dc1628-速比38比15',
    'hip_roll': '左右髋联动roll-dc1628', 'hip_pitch': '髋关节pitch-3505',
    'knee_pitch': '膝关节pitch-dc1628', 'ankle_fixed': '踝关节pitch-步进丝杆-仿真暂时固定',
    'foot': '脚板',
}
AXIS_COUNTS = {'shoulder_roll': 36, **{k:64 for k in
               ['shoulder_pitch','elbow_pitch','spine_pitch','spine_roll','hip_roll','hip_pitch','knee_pitch']}}
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
inventory = json.loads((AUDIT/'source_inventory.json').read_text(encoding='utf8'))
assert inventory['sha256'] == source_hash, 'Source changed: inspect again before building'
assert Path(bpy.data.filepath).resolve() == SOURCE.resolve()
assert not OUT.exists(), 'Refusing to overwrite an existing authored rig'
rows = {x['name']: x for x in inventory['objects']}
geo = {k: rows[v] for k,v in ALLOWLIST.items()}

def unit(x):
    x=np.asarray(x,dtype=float); return x/np.linalg.norm(x)

axes={}; points={}
for key,count in AXIS_COUNTS.items():
    candidates=[c for c in geo[key]['components'] if c['count']==count]
    assert len(candidates)==1, key
    c=candidates[0]; a=unit(c['axes'][0]); a*=1 if a[np.argmax(abs(a))]>0 else -1
    axes[key]=a; points[key]=np.array(c['center'])
for key in ['wrist_fixed','ankle_fixed','hand','foot']:
    points[key]=np.array(geo[key]['matrix'])[:3,3]

# Each derived mesh undergoes a rigid rotation/translation only. The user
# explicitly permits resetting assembly orientations to create a common zero.
def rotation_axis_to_y(axis):
    return np.array(Vector(axis).rotation_difference(Vector((0,1,0))).to_matrix())

def limb_frame(axis, downward):
    y=unit(axis); z=unit(-np.asarray(downward)+np.dot(downward,y)*y)
    x=unit(np.cross(y,z)); z=unit(np.cross(x,y))
    return np.column_stack((x,y,z)).T

zero={k:v.copy() for k,v in points.items()}
rot={k:np.eye(3) for k in ALLOWLIST}
for proximal,distal,terminal,paw in [('shoulder_pitch','elbow_pitch','wrist_fixed','hand'),
                                     ('hip_pitch','knee_pitch','ankle_fixed','foot')]:
    a,b,c=[points[k] for k in (proximal,distal,terminal)]
    zero[distal]=a+np.array([0.,0.,-np.linalg.norm(b-a)])
    zero[terminal]=zero[distal]+np.array([0.,0.,-np.linalg.norm(c-b)])
    rot[proximal]=rotation_axis_to_y(axes[proximal])
    rot[distal]=limb_frame(axes[distal], b-a)
    rot[terminal]=limb_frame(axes[distal], c-b)
    rot[paw]=rot[terminal]
    zero[paw]=zero[terminal]+rot[terminal]@(points[paw]-c)

# User's annotated correction supersedes the draft's raised waist. The rear
# waist-roll and hip-roll mounts stay fixed; only waist pitch / front roll move.
for key in ['spine_pitch','shoulder_roll']:
    zero[key][2]=points['spine_roll'][2]
zero['spine_pitch'][0]-=.005  # 2026-10-25: user requested a 5 mm rearward waist pivot shift.

transformed={}
for k,g in geo.items():
    anchor='wrist_fixed' if k=='hand' else 'ankle_fixed' if k=='foot' else k
    transformed[k]=(np.array(g['vertices'])-points[anchor])@rot[k].T+zero[anchor]

origin=points['spine_pitch'].copy()
min_z=min(v[:,2].min() for v in transformed.values())
shift=np.array([-origin[0],0.,-min_z])
# Keep the original global reference height, then apply the requested link length.
for proximal,distal,terminal,paw in [('shoulder_pitch','elbow_pitch','wrist_fixed','hand'),
                                     ('hip_pitch','knee_pitch','ankle_fixed','foot')]:
    dz=zero[proximal][2]-.044-zero[distal][2]
    for key in [distal,terminal,paw]:
        zero[key][2]+=dz
        transformed[key][:,2]+=dz
# Match the knee housing side-view direction to the front elbow, keeping its
# existing named-cylinder output shaft fixed. Do not mirror or scale its shape.
housing_vectors=[]
for key in ['elbow_pitch','knee_pitch']:
    ids=next(c['indices'] for c in geo[key]['components'] if c['count']==144)
    housing_vectors.append(transformed[key][ids].mean(0)-zero[key])
knee_mount_angle=math.atan2(housing_vectors[0][0],housing_vectors[0][2])-math.atan2(housing_vectors[1][0],housing_vectors[1][2])
knee_mount_rotation=np.array(Matrix.Rotation(knee_mount_angle,3,'Y'))
transformed['knee_pitch']=(transformed['knee_pitch']-zero['knee_pitch'])@knee_mount_rotation.T+zero['knee_pitch']
rot['knee_pitch']=knee_mount_rotation@rot['knee_pitch']
# Later user revision: elbow housings move rearward; knees retain the prior
# forward mounting based on the historical front elbow layout above.
elbow_mount_angle=-2*math.atan2(housing_vectors[0][0],housing_vectors[0][2])
elbow_mount_rotation=np.array(Matrix.Rotation(elbow_mount_angle,3,'Y'))
transformed['elbow_pitch']=(transformed['elbow_pitch']-zero['elbow_pitch'])@elbow_mount_rotation.T+zero['elbow_pitch']
rot['elbow_pitch']=elbow_mount_rotation@rot['elbow_pitch']
shank_dz=zero['knee_pitch'][2]-.050-zero['ankle_fixed'][2]
for key in ['ankle_fixed','foot']:
    zero[key][2]+=shank_dz
    transformed[key][:,2]+=shank_dz
# Lower the one opposed hip-roll assembly while leaving hip-pitch/leg centers fixed.
zero['hip_roll'][2]-=.010
transformed['hip_roll'][:,2]-=.010
zero['shoulder_roll'][2]-=.010
transformed['shoulder_roll'][:,2]-=.010
base=zero['spine_pitch']+shift
def pos(k,side=1): return zero[k]*[1,side,1]+shift

# Start the derived document from an empty scene; no unnamed cylinder is copied.
for obj in list(bpy.data.objects): bpy.data.objects.remove(obj,do_unlink=True)
for col in list(bpy.data.collections): bpy.data.collections.remove(col)
scene=bpy.context.scene
scene.unit_settings.system='METRIC'; scene.unit_settings.scale_length=1.0
scene.unit_settings.length_unit='MILLIMETERS'; scene.unit_settings.system_rotation='DEGREES'
scene.render.fps=50; scene.frame_start=1; scene.frame_end=100; scene.frame_set(1)
scene['robot_source_sha256']=source_hash
scene['robot_authoring_schema']='sable-rig/1'
scene['coordinate_convention']='+X forward, +Y left, +Z up; meters; radians in exported joint tables'
scene['limits_status']='PROVISIONAL_USER_EDITABLE_NOT_HARDWARE_VALIDATED'
scene['zero_pose_note']='Authored straight limbs; source preserved; rigid part reassembly, not a measured encoder zero.'
scene['source_allowlist_json']=json.dumps(ALLOWLIST,ensure_ascii=False)
scene['export_scope']='USD skeleton/visuals + joint sidecar; no validated physics articulation yet'
scene['waist_alignment_revision']='straight_waist_v1: use rear spine-roll height, preserve rear assembly'
scene['waist_rearward_offset_m']=.005
scene['hip_roll_lowering_revision']='opposed_hip_roll_down_10mm_v1'
scene['hip_roll_lowering_m']=.010
scene['shoulder_roll_lowering_revision']='shoulder_roll_down_10mm_v1'
scene['shoulder_roll_lowering_m']=.010
scene['upper_limb_length_revision']='upper_arm_and_thigh_44mm_v1'
scene['upper_limb_target_length_m']=.044
scene['hind_assembly_revision']='knee_motor_front_side_shank_50mm_v1'
scene['shank_target_length_m']=.050
scene['elbow_motor_mount_revision']='elbow_housings_rear_side_v1'
scene['elbow_motor_mounting_json']=json.dumps({side:{'rotation_world_y_deg':math.degrees(elbow_mount_angle),
    'pivot_world_m':(zero['elbow_pitch']*[1,sign,1]+shift).tolist(),
    'housing_offset_before_m':(housing_vectors[0]*[1,sign,1]).tolist(),
    'housing_offset_after_m':((elbow_mount_rotation@housing_vectors[0])*[1,sign,1]).tolist(),
    'method':'rigid rotation about unchanged output axis; mirrored side-view center placement'}
    for side,sign in [('L',1),('R',-1)]})
scene['knee_motor_mounting_json']=json.dumps({side:{'rotation_world_y_deg':math.degrees(knee_mount_angle),
    'pivot_world_m':(zero['knee_pitch']*[1,sign,1]+shift).tolist(),
    'reference_elbow_housing_offset_m':(housing_vectors[0]*[1,sign,1]).tolist(),
    'knee_housing_offset_before_m':(housing_vectors[1]*[1,sign,1]).tolist(),
    'knee_housing_offset_after_m':((knee_mount_rotation@housing_vectors[1])*[1,sign,1]).tolist()}
    for side,sign in [('L',1),('R',-1)]})

def collection(name):
    c=bpy.data.collections.new(name); scene.collection.children.link(c); return c
robot_col=collection('ROBOT_export')
guide_col=collection('GUIDES_nonexport')
shape_col=collection('CONTROL_SHAPES_nonexport')
render_col=collection('PREVIEW_nonexport')
shape_col.hide_render=True

arm=bpy.data.armatures.new('SableRobot_Skeleton')
rig=bpy.data.objects.new('SableRobot',arm); robot_col.objects.link(rig)
rig.show_in_front=True; arm.display_type='STICK'
bpy.context.view_layer.objects.active=rig; rig.select_set(True)
joint_rows=[]

def joint(name,parent,p,axis,limits,actuated=True,source_key=None,kind='revolute'):
    joint_rows.append(dict(name=name,parent=parent,position=np.asarray(p).tolist(),
                           axis=np.asarray(axis).tolist(),limits_deg=list(limits),actuated=actuated,
                           source_key=source_key,kind=kind))

joint('ROOT',None,base,[0,1,0],[-180,180],False,None,'floating')
joint('spine_pitch','ROOT',pos('spine_pitch'),[0,1,0],[-90,90],True,'spine_pitch')
joint('spine_roll','spine_pitch',pos('spine_roll'),[1,0,0],[-30,30],True,'spine_roll')
for side,prefix in [(1,'L'),(-1,'R')]:
    hr='hip_roll' if side==1 else 'R_hip_roll_mimic'
    joint(hr,'spine_roll',pos('hip_roll',side),[1,0,0],[-30,30],side==1,'hip_roll')
    joint(prefix+'_shoulder_roll','ROOT',pos('shoulder_roll',side),[1,0,0],[-45,45],True,'shoulder_roll')
    joint(prefix+'_shoulder_pitch',prefix+'_shoulder_roll',pos('shoulder_pitch',side),[0,1,0],[-120,120],True,'shoulder_pitch')
    joint(prefix+'_elbow_pitch',prefix+'_shoulder_pitch',pos('elbow_pitch',side),[0,1,0],[-150,150],True,'elbow_pitch')
    joint(prefix+'_wrist_fixed',prefix+'_elbow_pitch',pos('wrist_fixed',side),[0,1,0],[0,0],False,'wrist_fixed','fixed')
    joint(prefix+'_hip_pitch',hr,pos('hip_pitch',side),[0,1,0],[-120,120],True,'hip_pitch')
    joint(prefix+'_knee_pitch',prefix+'_hip_pitch',pos('knee_pitch',side),[0,1,0],[-150,150],True,'knee_pitch')
    joint(prefix+'_ankle_fixed',prefix+'_knee_pitch',pos('ankle_fixed',side),[0,1,0],[0,0],False,'ankle_fixed','fixed')

bpy.ops.object.mode_set(mode='EDIT')
for row in joint_rows:
    b=arm.edit_bones.new(row['name']); b.head=row['position']
    b.tail=Vector(row['position'])+Vector(row['axis'])*(.035 if row['name']=='ROOT' else .014)
    b.align_roll(Vector((0,0,1))); b.use_deform=True
    if row['parent']: b.parent=arm.edit_bones[row['parent']]
bpy.ops.object.mode_set(mode='OBJECT')
bc={name:arm.collections.new(name) for name in ['Body','Front legs','Hind legs','Fixed terminals']}
for row in joint_rows:
    pb=rig.pose.bones[row['name']]; pb.rotation_mode='XYZ'
    pb.lock_scale=(True,True,True)
    pb['joint_type']=row['kind']; pb['actuated']=row['actuated']
    pb['limit_status']='temporary; edit Joint limits constraint'
    pb['rotation_control']='Local Y only; R Y Y, or Rotation Y in Item panel'
    if row['source_key']: pb['source_object']=ALLOWLIST[row['source_key']]
    if row['name']=='ROOT': group='Body'
    elif row['kind']=='fixed': group='Fixed terminals'
    elif 'shoulder' in row['name'] or 'elbow' in row['name']: group='Front legs'
    elif 'spine' in row['name']: group='Body'
    else: group='Hind legs'
    bc[group].assign(arm.bones[row['name']])
    if row['kind']!='floating':
        pb.lock_location=(True,True,True)
        pb.lock_rotation=(True,row['kind']=='fixed' or row['name']=='R_hip_roll_mimic',True)
        c=pb.constraints.new('LIMIT_ROTATION'); c.name='Joint limits (temporary)'
        c.owner_space='LOCAL'; c.use_transform_limit=True
        c.use_limit_x=c.use_limit_y=c.use_limit_z=True
        c.min_x=c.max_x=c.min_z=c.max_z=0
        c.min_y,c.max_y=map(math.radians,row['limits_deg'])
    pb.color.palette='THEME03' if group=='Front legs' else 'THEME04' if group=='Hind legs' else 'THEME09'

dep=rig.pose.bones['R_hip_roll_mimic']
c=dep.constraints.new('COPY_ROTATION'); c.name='One motor: right = -left'
c.target=rig; c.subtarget='hip_roll'; c.owner_space=c.target_space='LOCAL'
c.use_x=False; c.use_y=True; c.use_z=False; c.invert_y=True; c.mix_mode='REPLACE'
dep.constraints.move(len(dep.constraints)-1,0)
for dst,src in [('min_y','max_y'),('max_y','min_y')]:
    fc=dep.constraints['Joint limits (temporary)'].driver_add(dst)
    var=fc.driver.variables.new(); var.name='bound'; var.type='SINGLE_PROP'
    var.targets[0].id=rig
    var.targets[0].data_path='pose.bones["hip_roll"].constraints["Joint limits (temporary)"].'+src
    fc.driver.expression='-bound'
dep['coupling']='q_R = -q_hip_roll; one actuator only'

def material(name,color):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); return m
mats={'motor':material('Motor • warm metal',(0.82,.46,.16)),
      'terminal':material('Hands and feet • teal',(.08,.53,.49)),
      'rear':material('Rear motor • slate',(.24,.39,.57)),
      'guide':material('Assembly guides only',(.36,.43,.49))}
mesh_rows=[]
def mesh_part(key,side,bone):
    prefix='' if side==0 else 'L_' if side==1 else 'R_'
    name=prefix+key+'_visual'
    side=side or 1
    vv=transformed[key]*[1,side,1]+shift
    faces=geo[key]['faces'] if side==1 else [list(reversed(f)) for f in geo[key]['faces']]
    src=bpy.data.meshes.new(name+'_mesh'); src.from_pydata(vv.tolist(),[],faces); src.update()
    obj=bpy.data.objects.new(name,src); robot_col.objects.link(obj); obj.parent=rig
    vg=obj.vertex_groups.new(name=bone); vg.add(list(range(len(vv))),1.,'REPLACE')
    mod=obj.modifiers.new('Rigid link binding','ARMATURE'); mod.object=rig
    obj['robot_export']=True; obj['source_object']=ALLOWLIST[key]; obj['rigid_link_bone']=bone
    obj['mirrored_from_source']=side<0; obj['part_geometry_changed']=False
    src.materials.append(mats['terminal'] if key in ['wrist_fixed','ankle_fixed','hand','foot']
                         else mats['rear'] if key in ['hip_pitch','knee_pitch'] else mats['motor'])
    row={'object':name,'source_object':ALLOWLIST[key],'source_key':key,'side':side,'bone':bone,
         'source_vertices':len(vv),'faces':len(faces)}
    mesh_rows.append(row)
    return obj

mesh_part('spine_pitch',0,'ROOT'); mesh_part('spine_roll',0,'spine_pitch')
for side,prefix in [(1,'L'),(-1,'R')]:
    hr='hip_roll' if side==1 else 'R_hip_roll_mimic'
    for key,bone in [('hip_roll','spine_roll'),('shoulder_roll','ROOT'),
                     ('shoulder_pitch',prefix+'_shoulder_roll'),('elbow_pitch',prefix+'_shoulder_pitch'),
                     ('wrist_fixed',prefix+'_wrist_fixed'),('hand',prefix+'_wrist_fixed'),
                     ('hip_pitch',hr),('knee_pitch',prefix+'_hip_pitch'),
                     ('ankle_fixed',prefix+'_ankle_fixed'),('foot',prefix+'_ankle_fixed')]:
        mesh_part(key,side,bone)

# Explicit user revision: only the four terminal boxes become capsules.
colliders=[]
for row in mesh_rows:
    if row['object'] in TERMINALS:
        obj=bpy.data.objects[row['object']]
        colliders.append(replace_terminal(obj,rig))
        row['authored_vertex_count']=len(obj.data.vertices)
        row['authored_face_count']=len(obj.data.polygons)
        row['geometry_revision']='terminal capsule; original center/axis/overall length retained'
rig['collision_shapes_json']=json.dumps(colliders)
scene['terminal_capsule_revision']='terminal_capsules_v1'

# Display-only links make the motor-only source easy to pose; excluded from USD.
def guide(name,a,b,bone,radius=.0014):
    a,b=Vector(a),Vector(b); delta=b-a
    bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=radius,depth=delta.length,location=(a+b)*.5)
    obj=bpy.context.object; obj.name='GUIDE_'+name
    obj.rotation_mode='QUATERNION'; obj.rotation_quaternion=delta.to_track_quat('Z','Y')
    bpy.context.view_layer.update(); matrix=obj.matrix_world.copy()
    for v in obj.data.vertices: v.co=matrix@v.co
    obj.matrix_world=Matrix.Identity(4)
    for col in list(obj.users_collection): col.objects.unlink(obj)
    guide_col.objects.link(obj); obj.parent=rig
    vg=obj.vertex_groups.new(name=bone); vg.add(list(range(len(obj.data.vertices))),1.,'REPLACE')
    obj.modifiers.new('Follow link','ARMATURE').object=rig
    obj.data.materials.append(mats['guide']); obj.hide_select=True; obj['robot_export']=False

for side,pre in [(1,'L'),(-1,'R')]:
    for a,b,bone in [('shoulder_roll','shoulder_pitch',pre+'_shoulder_roll'),
                      ('shoulder_pitch','elbow_pitch',pre+'_shoulder_pitch'),
                      ('elbow_pitch','wrist_fixed',pre+'_elbow_pitch'),
                      ('hip_roll','hip_pitch','hip_roll' if side==1 else 'R_hip_roll_mimic'),
                      ('hip_pitch','knee_pitch',pre+'_hip_pitch'),
                      ('knee_pitch','ankle_fixed',pre+'_knee_pitch')]:
        guide(pre+'_'+a,pos(a,side),pos(b,side),bone)
    guide(pre+'_chest',pos('spine_pitch'),pos('shoulder_roll',side),'ROOT',.002)
    guide(pre+'_pelvis',pos('spine_roll'),pos('hip_roll',side),'spine_roll',.002)
guide('waist',pos('spine_pitch'),pos('spine_roll'),'spine_pitch',.002)

# Pose controls use the same local axis for every hinge, with all other DOFs locked.
verts=[(.012*math.cos(t),0,.012*math.sin(t)) for t in np.linspace(0,2*math.pi,49)[:-1]]
shape_mesh=bpy.data.meshes.new('Joint_control_ring'); shape_mesh.from_pydata(verts,[(i,(i+1)%48) for i in range(48)],[])
shape=bpy.data.objects.new('Joint_control_ring',shape_mesh); shape_col.objects.link(shape)
shape.hide_render=True; shape.hide_set(True)
for row in joint_rows:
    pb=rig.pose.bones[row['name']]
    if row['kind']=='revolute':
        pb.custom_shape=shape; pb.use_custom_shape_bone_size=False
        pb.custom_shape_scale_xyz=(.8,.8,.8) if 'roll' in row['name'] else (1,1,1)

# A neutral editable keyframe, not an automatically accepted standing policy.
for pb in rig.pose.bones:
    if pb.name=='ROOT':
        pb.keyframe_insert(data_path='location',frame=1,group='ROOT')
        pb.keyframe_insert(data_path='rotation_euler',frame=1,group='ROOT')
    elif pb['actuated']:
        pb.keyframe_insert(data_path='rotation_euler',index=1,frame=1,group=pb.name)
rig.animation_data.action.name='StandPose_EDIT_ME'
rig.animation_data.action.use_fake_user=True
scene.timeline_markers.new('START • edit standing pose',frame=1)
scene.timeline_markers.new('Next pose / transition',frame=50)
rig['joint_table_json']=json.dumps(joint_rows,ensure_ascii=False)
rig['mesh_table_json']=json.dumps(mesh_rows,ensure_ascii=False)
rig['independent_motor_count']=13
rig['coupling_note']='hip_roll and R_hip_roll_mimic share ONE actuator; do not create a second rear-roll motor'

# Same final dimensions/mounting used by the incremental saved-rig revision.
from layout_geometry import apply_layout, update_build_report
layout_audit=apply_layout(rig,scene,shoulder_distance=.090)
from pitch_mount_geometry import apply_pitch_mount
pitch_mount_audit=apply_pitch_mount(rig,scene)
from motor_offset_geometry import apply_motor_offsets,configure_chest_root
motor_offset_audit=apply_motor_offsets(rig,scene)
motor_offset_audit['chest_root']=configure_chest_root(rig,scene,ROOT/'blender/draft.blend')
from straight_limb_geometry import apply_straight_limbs
straight_limb_audit=apply_straight_limbs(rig,scene)
scene['robot_geometry_source_file']='robot_authoring/source/original_draft_2026-10-24.blend'
scene['robot_pose_reference_file']='blender/draft.blend'
scene['robot_pose_reference_sha256']=hashlib.sha256((ROOT/'blender/draft.blend').read_bytes()).hexdigest()
joint_rows=json.loads(rig['joint_table_json']);mesh_rows=json.loads(rig['mesh_table_json'])
colliders=json.loads(rig['collision_shapes_json'])

notes=(ROOT/'robot_authoring'/'README.md').read_text(encoding='utf8')
bpy.data.texts.new('START_HERE_使用说明').write(notes)
bpy.data.texts.new('EXPORT_ROBOT_DESCRIPTION.py').write(
    'import bpy, runpy\nfrom pathlib import Path\n'
    'path=Path(bpy.path.abspath("//"))/"../robot_authoring/export_description.py"\n'
    'runpy.run_path(str(path.resolve()),run_name="__main__")\n')

scene.render.engine='BLENDER_WORKBENCH'; scene.render.resolution_x=1200; scene.render.resolution_y=900
scene.render.resolution_percentage=100; scene.render.image_settings.file_format='PNG'
scene.display.shading.light='STUDIO'; scene.display.shading.studio_light='paint.sl'
scene.display.shading.color_type='MATERIAL'; scene.display.shading.show_shadows=True
scene.display.shading.show_cavity=True; scene.display.shading.cavity_type='BOTH'
scene.display.shading.background_type='WORLD'; scene.world.color=(.045,.055,.075)
scene.view_settings.view_transform='Standard'
bounds=np.array([v.co for row in mesh_rows for v in bpy.data.objects[row['object']].data.vertices])
center=Vector(((bounds[:,0].min()+bounds[:,0].max())/2,0,(bounds[:,2].min()+bounds[:,2].max())/2))
camera_data=bpy.data.cameras.new('Authoring_preview'); camera=bpy.data.objects.new('Authoring_preview',camera_data)
render_col.objects.link(camera); scene.camera=camera; camera_data.type='ORTHO'; camera_data.ortho_scale=.29
camera.location=center+Vector((.28,-.40,.23)); camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
camera.hide_set(True)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_distance=.34
            area.spaces.active.region_3d.view_location=center
            area.spaces.active.region_3d.view_rotation=camera.rotation_euler.to_quaternion()
            area.spaces.active.clip_start=.001
            area.spaces.active.shading.color_type='MATERIAL'
            area.spaces.active.overlay.show_floor=True

for obj in bpy.context.selected_objects: obj.select_set(False)
rig.select_set(True); bpy.context.view_layer.objects.active=rig
bpy.ops.object.mode_set(mode='POSE')
arm.bones.active=arm.bones['L_shoulder_pitch']
rig.pose.bones['L_shoulder_pitch'].select=True
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='PROPERTIES':
            try: area.spaces.active.context='BONE_CONSTRAINT'
            except TypeError: pass  # Background contexts may not expose pose tabs until UI redraw.
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT),check_existing=False)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
report={'source_sha256':source_hash,'output':str(OUT),'allowlist':ALLOWLIST,
        'excluded':[o['name'] for o in inventory['objects'] if o['name'] not in ALLOWLIST.values()],
        'coordinates':scene['coordinate_convention'],'source_preserved':True,
        'joint_rows':joint_rows,'mesh_rows':mesh_rows,'base_height_m':float(base[2]),
        'terminal_capsule_revision':'terminal_capsules_v1','collision_shapes':colliders,
        'upper_limb_length_revision':'upper_arm_and_thigh_44mm_v1','upper_limb_target_length_m':.044,
        'hind_assembly_revision':scene['hind_assembly_revision'],'shank_target_length_m':.050,
        'knee_motor_mounting':json.loads(scene['knee_motor_mounting_json']),
        'elbow_motor_mount_revision':scene['elbow_motor_mount_revision'],
        'elbow_motor_mounting':json.loads(scene['elbow_motor_mounting_json']),
        'hip_roll_lowering_revision':scene['hip_roll_lowering_revision'],'hip_roll_lowering_m':.010,
        'shoulder_roll_lowering_revision':scene['shoulder_roll_lowering_revision'],'shoulder_roll_lowering_m':.010,
        'assembly_policy':'User-authorized rigid reassembly; align pitch shaft lines to +Y and limb joint centers vertically; no mesh scaling.',
        'world_translation_m':shift.tolist(),
        'parts':{k:{'rotation_source_to_zero':rot[k].tolist(),'source_pivot_m':points[k].tolist(),
                    'zero_pivot_before_global_shift_m':zero[k].tolist()} for k in ALLOWLIST},
        'front_shoulder_world_z_correction_deg':math.degrees(math.atan2(rot['shoulder_pitch'][1,0],rot['shoulder_pitch'][0,0])),
        'independent_actuators':13,'revolute_coordinates':14,'fixed_terminals':4,
        'limits':'temporary, editable; not transferred from old policies',
        'validation_scope':'Geometry and Blender rig only; mass, inertia, collision and torque-speed calibration pending'}
update_build_report(report,rig,layout_audit)
report.update(shoulder_pitch_mount_revision=pitch_mount_audit['revision'],shoulder_pitch_mounting=pitch_mount_audit)
report.update(motor_offset_revision=motor_offset_audit['revision'],motor_offsets=motor_offset_audit)
report.update(straight_limb_revision=straight_limb_audit['revision'],straight_limbs=straight_limb_audit)
(AUDIT/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('RIG_SAVED',OUT,'JOINTS',len(joint_rows),'MESHES',len(mesh_rows))

sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import export_robot
export_robot()
bpy.ops.object.mode_set(mode='OBJECT')
for label,delta in [('perspective',(.28,-.40,.23)),('side',(0,-.45,0)),('front',(.45,0,0))]:
    camera.hide_set(False); camera.location=center+Vector(delta)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('DONE_SOURCE_UNCHANGED',hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash)
