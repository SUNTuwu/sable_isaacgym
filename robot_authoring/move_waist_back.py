"""Apply the user's 5 mm rearward waist-pitch rest-geometry correction once."""
import bpy
import hashlib
import json
import math
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'blender/sable_rig.blend'
SOURCE=ROOT/'blender/draft.blend'
BACKUP=ROOT/'待删除/2026-10-25_before_waist_back_5mm/blender/sable_rig.blend'
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
before_hash=digest(TARGET)
assert before_hash==digest(BACKUP), 'Saved rig changed since backup'
source_hash=digest(SOURCE)
rig=bpy.data.objects['SableRobot']; scene=bpy.context.scene
assert not scene.get('waist_rearward_offset_m'), 'Waist correction already applied'
old_frame=scene.frame_current; old_mode=rig.mode
delta=Vector((-.005,0,0)); moved={'ROOT','spine_pitch'}
rows=json.loads(rig['joint_table_json']); meshes=json.loads(rig['mesh_table_json'])

def mesh_hash(name):
    return hashlib.sha256(np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32).tobytes()).hexdigest()

def actions_snapshot():
    return [{'name':a.name,'curves':[
        {'path':fc.data_path,'index':fc.array_index,'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in fc.keyframe_points]}
        for layer in a.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]}
        for a in bpy.data.actions]

def limits_snapshot():
    return {p.name:[[c.name,c.type]+[getattr(c,k,None) for k in ['min_x','max_x','min_y','max_y','min_z','max_z']]
                    for c in p.constraints] for p in rig.pose.bones}

protected={r['object']:mesh_hash(r['object']) for r in meshes if r['object']!='spine_pitch_visual'}
bones_before={b.name:[list(x) for x in b.matrix_local] for b in rig.data.bones}
mesh=bpy.data.objects['spine_pitch_visual'].data
vertices_before=[list(v.co) for v in mesh.vertices]
actions_before=actions_snapshot(); limits_before=limits_snapshot()
scene.frame_set(1)
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active=rig
bpy.ops.object.mode_set(mode='EDIT')
for name in moved:
    bone=rig.data.edit_bones[name]; bone.head+=delta; bone.tail+=delta
bpy.ops.object.mode_set(mode='OBJECT')
for v in mesh.vertices:v.co+=delta
mesh.update()

for name,a,b,radius in [('GUIDE_waist','spine_pitch','spine_roll',.002),
    ('GUIDE_L_chest','spine_pitch','L_shoulder_roll',.002),
    ('GUIDE_R_chest','spine_pitch','R_shoulder_roll',.002)]:
    obj=bpy.data.objects[name]
    start=rig.data.bones[a].head_local.copy(); end=rig.data.bones[b].head_local.copy()
    rotation=(end-start).to_track_quat('Z','Y')
    vertices=[point+rotation@Vector((radius*math.cos(2*math.pi*i/12),radius*math.sin(2*math.pi*i/12),0))
              for point in (start,end) for i in range(12)]
    assert len(obj.data.vertices)==len(vertices)
    for v,p in zip(obj.data.vertices,vertices):v.co=p
    obj.data.update()

for row in rows:row['position']=list(rig.data.bones[row['name']].head_local)
rig['joint_table_json']=json.dumps(rows,ensure_ascii=False)
scene['waist_rearward_offset_m']=.005
scene['waist_backshift_revision']='waist_pitch_back_5mm_v1'
notes=bpy.data.texts.get('START_HERE_使用说明')
notes.clear(); notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
scene.frame_set(old_frame); bpy.context.view_layer.update()
assert all(mesh_hash(name)==expected for name,expected in protected.items())
for name,values in bones_before.items():
    expected=np.array(values)
    if name in moved:expected[:3,3]+=np.array(delta)
    error=abs(np.array(rig.data.bones[name].matrix_local)-expected)
    # Entering Edit Mode can round the normalized 3x3 basis by a float32 ULP.
    assert np.max(error[:3,3])<1e-8 and np.max(error[:3,:3])<5e-7,(name,error.tolist())
assert actions_before==actions_snapshot() and limits_before==limits_snapshot()
assert np.max(abs(np.array([v.co for v in mesh.vertices])-np.array(vertices_before)-np.array(delta)))<1e-8
bpy.context.view_layer.objects.active=rig
if old_mode!='OBJECT':bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash, 'Concurrent save detected; refusing overwrite'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert digest(SOURCE)==source_hash
audit={'revision':'waist_pitch_back_5mm_v1','translation_world_m':list(delta),
    'input_rig_sha256':before_hash,'output_rig_sha256':digest(TARGET),'source_sha256':source_hash,
    'bone_rest_matrices_before':bones_before,'moved_bones':sorted(moved),
    'waist_vertices_before':vertices_before,'protected_mesh_vertex_sha256':protected,
    'actions_before':actions_before,'limits_before':limits_before,
    'keyframes_preserved':True,'limits_preserved':True,'source_unchanged':True}
(ROOT/'robot_authoring/inspection/waist_backshift.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
p=ROOT/'robot_authoring/inspection/build_report.json'
build=json.loads(p.read_text(encoding='utf8'))
build['joint_rows']=rows; build['waist_rearward_offset_m']=.005
build['parts']['spine_pitch']['zero_pivot_before_global_shift_m'][0]-=.005
p.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
print('WAIST_MOVED_WORLD_MM',list(delta*1000),'OTHER_MESHES_PRESERVED',len(protected))

sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import export_robot
export_robot()
bpy.ops.object.mode_set(mode='OBJECT'); scene.frame_set(1)
camera=scene.camera
allverts=np.array([v.co for row in meshes for v in bpy.data.objects[row['object']].data.vertices])
center=Vector(((allverts[:,0].min()+allverts[:,0].max())/2,0,(allverts[:,2].min()+allverts[:,2].max())/2))
for label,offset in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0))]:
    camera.hide_set(False); camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('WAIST_BACKSHIFT_EXPORT_COMPLETE')
