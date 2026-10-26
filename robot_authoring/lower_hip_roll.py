"""Lower roll mounts 10 mm; defaults to hip, also called by shoulder wrapper."""
import bpy
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from mathutils import Vector

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'blender/sable_rig.blend'
SOURCE=ROOT/'blender/draft.blend'
MOUNT=globals().get('ROLL_MOUNT','hip')
assert MOUNT in ['hip','shoulder']
BACKUP=ROOT/('待删除/2026-10-25_before_'+MOUNT+'_roll_down_10mm/blender/sable_rig.blend')
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
before_hash,source_hash=digest(TARGET),digest(SOURCE)
assert before_hash==digest(BACKUP),'Concurrent change since backup'
rig,scene=bpy.data.objects['SableRobot'],bpy.context.scene
assert not scene.get(MOUNT+'_roll_lowering_revision'),'Already applied'
assert np.max(abs(np.array(rig.matrix_world)-np.eye(4)))<1e-8
old_frame,old_mode=scene.frame_current,rig.mode
rows,meshes=json.loads(rig['joint_table_json']),json.loads(rig['mesh_table_json'])
delta=Vector((0,0,-.010))
moved_bones=['hip_roll','R_hip_roll_mimic'] if MOUNT=='hip' else ['L_shoulder_roll','R_shoulder_roll']
moved_meshes=[side+'_'+MOUNT+'_roll_visual' for side in ['L','R']]


def mesh_hash(name):
    return hashlib.sha256(np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32).tobytes()).hexdigest()


def actions_snapshot():
    return [{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
        'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points]}
        for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]}
        for a in bpy.data.actions]


def limits_snapshot():
    return {p.name:[[c.name,c.type]+[getattr(c,k,None) for k in
        ['min_x','max_x','min_y','max_y','min_z','max_z']] for c in p.constraints] for p in rig.pose.bones}


protected={r['object']:mesh_hash(r['object']) for r in meshes if r['object'] not in moved_meshes}
bones_before={b.name:[list(row) for row in b.matrix_local] for b in rig.data.bones}
vertices_before={n:[list(v.co) for v in bpy.data.objects[n].data.vertices] for n in moved_meshes}
actions_before,limits_before=actions_snapshot(),limits_snapshot()
scene.frame_set(1)
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active=rig
bpy.ops.object.mode_set(mode='EDIT')
for name in moved_bones:
    b=rig.data.edit_bones[name];b.head+=delta;b.tail+=delta
bpy.ops.object.mode_set(mode='OBJECT')
for name in moved_meshes:
    obj=bpy.data.objects[name]
    for v in obj.data.vertices:v.co+=delta
    obj.data.update()

# Recreate the two upstream and two downstream display guides, without changing
# the output axes or moving hip-pitch joints with their new parent rest frames.
for side in ['L','R']:
    roll=('hip_roll' if side=='L' else 'R_hip_roll_mimic') if MOUNT=='hip' else side+'_shoulder_roll'
    upstream='spine_roll' if MOUNT=='hip' else 'spine_pitch'
    guide='pelvis' if MOUNT=='hip' else 'chest'
    for name,a,b,radius in [('GUIDE_'+side+'_'+guide,upstream,roll,.002),
                           ('GUIDE_'+side+'_'+MOUNT+'_roll',roll,side+'_'+MOUNT+'_pitch',.0014)]:
        obj=bpy.data.objects[name]
        start,end=rig.data.bones[a].head_local,rig.data.bones[b].head_local
        q=(end-start).to_track_quat('Z','Y')
        vertices=[p+q@Vector((radius*math.cos(2*math.pi*i/12),radius*math.sin(2*math.pi*i/12),0))
                  for p in [start,end] for i in range(12)]
        faces=[tuple(reversed(range(12))),tuple(range(12,24))]
        faces += [(i,(i+1)%12,(i+1)%12+12,i+12) for i in range(12)]
        group_name=obj.vertex_groups[0].name;old=obj.data
        mesh=bpy.data.meshes.new(name+'_hip_roll_lowered');mesh.from_pydata(vertices,[],faces);mesh.update()
        for mat in old.materials:mesh.materials.append(mat)
        obj.data=mesh
        group=obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
        group.add(list(range(24)),1.,'REPLACE')
        if old.users==0:bpy.data.meshes.remove(old)

for row in rows:row['position']=list(rig.data.bones[row['name']].head_local)
rig['joint_table_json']=json.dumps(rows,ensure_ascii=False)
scene[MOUNT+'_roll_lowering_revision']=('opposed_' if MOUNT=='hip' else '')+MOUNT+'_roll_down_10mm_v1'
scene[MOUNT+'_roll_lowering_m']=.010
notes=bpy.data.texts['START_HERE_使用说明'];notes.clear()
notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
assert all(mesh_hash(n)==h for n,h in protected.items())
for name,values in bones_before.items():
    expected=np.array(values)
    if name in moved_bones:expected[:3,3]+=np.array(delta)
    error=abs(np.array(rig.data.bones[name].matrix_local)-expected)
    assert error[:3,3].max()<1e-8 and error[:3,:3].max()<5e-7,name
assert actions_snapshot()==actions_before and limits_snapshot()==limits_before
scene.frame_set(old_frame);bpy.context.view_layer.update()
if old_mode!='OBJECT':bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash,'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert digest(SOURCE)==source_hash
audit={'revision':scene[MOUNT+'_roll_lowering_revision'],'input_rig_sha256':before_hash,'output_rig_sha256':digest(TARGET),
    'source_sha256':source_hash,'source_unchanged':True,'translation_world_m':list(delta),
    'moved_bones':moved_bones,'moved_meshes':moved_meshes,
    'bone_rest_matrices_before':bones_before,'mesh_vertices_before':vertices_before,
    'protected_mesh_vertex_sha256':protected,'actions_before':actions_before,'limits_before':limits_before,
    'keyframes_preserved':True,'limits_preserved':True}
(ROOT/('robot_authoring/inspection/'+MOUNT+'_roll_lowering.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'robot_authoring/inspection/build_report.json';build=json.loads(path.read_text(encoding='utf8'))
build.update(joint_rows=rows)
build[MOUNT+'_roll_lowering_revision']=scene[MOUNT+'_roll_lowering_revision']
build[MOUNT+'_roll_lowering_m']=.010
build['parts'][MOUNT+'_roll']['zero_pivot_before_global_shift_m'][2]-=.010
path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import export_robot
export_robot()
print(MOUNT.upper()+'_ROLL_LOWERED_10MM',json.dumps({n:list(rig.data.bones[n].head_local) for n in moved_bones}))
bpy.ops.object.mode_set(mode='OBJECT');scene.frame_set(1)
camera=scene.camera
vv=np.array([v.co for row in meshes for v in bpy.data.objects[row['object']].data.vertices])
center=Vector(((vv[:,0].min()+vv[:,0].max())/2,0,(vv[:,2].min()+vv[:,2].max())/2))
for label,offset in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0))]:
    camera.hide_set(False);camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print(MOUNT.upper()+'_ROLL_LOWERING_EXPORT_COMPLETE')
