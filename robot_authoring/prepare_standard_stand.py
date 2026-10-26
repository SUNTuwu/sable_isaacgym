"""Use the user's full frame-6 pose as the mirrored frame-5 standard stand."""
import bpy,json,math,hashlib,sys
import numpy as np
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import effective_angle,export_robot
target=ROOT/'blender/sable_rig.blend';backup=ROOT/'待删除/2026-10-25_before_standard_stand_training/blender/sable_rig.blend'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
input_hash=digest(target);assert input_hash==digest(backup)
protected={p:digest(ROOT/p) for p in ['blender/draft.blend','robot_authoring/source/original_draft_2026-10-24.blend']}
rig=bpy.data.objects['SableRobot'];s=bpy.context.scene
s.frame_set(6);bpy.context.view_layer.update()
assert abs(math.degrees(rig.pose.bones['spine_pitch'].rotation_euler.y)+40)<1e-4
source_pose={p.name:list(p.rotation_euler) for p in rig.pose.bones}
for part in ['shoulder_roll','shoulder_pitch','elbow_pitch','hip_pitch','knee_pitch']:
    a,b=rig.pose.bones['L_'+part],rig.pose.bones['R_'+part]
    ca=next(c for c in a.constraints if c.type=='LIMIT_ROTATION');cb=next(c for c in b.constraints if c.type=='LIMIT_ROTATION')
    sign=-1 if part=='shoulder_roll' else 1
    cb.min_y,cb.max_y=(ca.min_y,ca.max_y) if sign==1 else (-ca.max_y,-ca.min_y)
    source_pose[b.name]=[0,sign*source_pose[a.name][1],0]
source_pose['spine_pitch']=[0,math.radians(-40),0]
for frame in [5,6]:
    s.frame_set(frame)
    for name,q in source_pose.items():
        p=rig.pose.bones[name];p.rotation_euler=q
        p.keyframe_insert(data_path='rotation_euler',frame=frame)
s.frame_set(5);bpy.context.view_layer.update()
s['standard_stand_frame']=5;s['standard_stand_source_frame']=6
s['standard_stand_waist_pitch_deg']=-40.
s['standard_stand_note']='User left-side pose/limits mirrored; source complete keys at frame 6, copied to requested frame 5.'
q={r['name']:effective_angle(rig,r['name']) for r in json.loads(rig['joint_table_json']) if r['kind']=='revolute'}
assert abs(q['spine_pitch']-math.radians(-40))<1e-6
for part in ['shoulder_pitch','elbow_pitch','hip_pitch','knee_pitch']:
    assert abs(q['L_'+part]-q['R_'+part])<1e-6,part
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='POSE')
bpy.context.preferences.filepaths.save_version=0
assert digest(target)==input_hash
bpy.ops.wm.save_as_mainfile(filepath=str(target),check_existing=False)
assert all(digest(ROOT/p)==v for p,v in protected.items())
manifest=export_robot()
root_matrix=np.array(rig.matrix_world@rig.pose.bones['ROOT'].matrix)
geometries=[]
for row in json.loads(rig['mesh_table_json']):
    obj=bpy.data.objects[row['object']]
    assert np.max(abs(np.array(obj.matrix_world)-np.eye(4)))<1e-8
    geometries.append(dict(row,vertices=[list(v.co) for v in obj.data.vertices],faces=[list(f.vertices) for f in obj.data.polygons],bone=obj.vertex_groups[0].name))
payload={'rig_sha256':digest(target),'source_rig_sha256':input_hash,'frame':5,'source_frame':6,
    'angles_rad':q,'angles_deg':{k:math.degrees(v) for k,v in q.items()},'root_matrix_world':root_matrix.tolist(),
    'joints':manifest['joints'],'meshes':geometries,'capsules':manifest['collision_shapes'],
    'bone_world_at_stand':{p.name:[list(r) for r in rig.matrix_world@p.matrix] for p in rig.pose.bones},
    'protected_files_sha256':protected,'scope':'Blender standard-pose symmetry/limits verified; physical import and standing validation pending'}
out=ROOT/'deployment/legged_gym/assets';out.mkdir(parents=True,exist_ok=True)
(out/'authored_model.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'robot_authoring/inspection/standard_stand.json').write_text(json.dumps({k:v for k,v in payload.items() if k not in ['meshes','joints','capsules','bone_world_at_stand']},ensure_ascii=False,indent=2),encoding='utf8')
print('STANDARD_STAND',json.dumps(payload['angles_deg']))
