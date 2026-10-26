"""Set saved-rig waist-to-shoulder spacing to 90 mm without changing poses."""
import bpy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from mathutils import Vector

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'robot_authoring'))
from layout_geometry import FRONT_PARTS, rebuild_guides
TARGET=ROOT/'blender/sable_rig.blend'
SOURCE=ROOT/'blender/draft.blend'
BACKUP=ROOT/'待删除/2026-10-25_before_waist_shoulder_90mm'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before_hash,source_hash=digest(TARGET),digest(SOURCE)
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
assert before_hash==digest(BACKUP/'blender/sable_rig.blend')
rig,scene=bpy.data.objects['SableRobot'],bpy.context.scene
assert not scene.get('shoulder_distance_revision'),'Already applied'
old_frame,old_mode=scene.frame_current,rig.mode
rows,meshes=json.loads(rig['joint_table_json']),json.loads(rig['mesh_table_json'])
bone_before={b.name:np.array(b.matrix_local) for b in rig.data.bones}
vertices_before={r['object']:np.array([v.co for v in bpy.data.objects[r['object']].data.vertices],dtype=np.float64) for r in meshes}

def actions():
    return [{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
             'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points]}
             for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]}
             for a in bpy.data.actions]

def limits():
    return {p.name:[[c.name,c.type]+[getattr(c,k,None) for k in
             ['min_x','max_x','min_y','max_y','min_z','max_z']] for c in p.constraints] for p in rig.pose.bones}

actions_before,limits_before=actions(),limits()
dx=rig.data.bones['spine_pitch'].head_local.x+.090-rig.data.bones['L_shoulder_pitch'].head_local.x
assert abs(dx+.010)<1e-8
delta=np.array([dx,0.,0.])
moved_bones=[n for n in bone_before if n[:2] in ['L_','R_'] and n[2:] in FRONT_PARTS]
moved_meshes=[r['object'] for r in meshes if r['source_key'] in FRONT_PARTS]
scene.frame_set(1)
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active=rig
bpy.ops.object.mode_set(mode='EDIT')
for name in moved_bones:
    b=rig.data.edit_bones[name];b.head+=Vector(delta);b.tail+=Vector(delta)
bpy.ops.object.mode_set(mode='OBJECT')
for name in moved_meshes:
    obj=bpy.data.objects[name]
    for v,co in zip(obj.data.vertices,vertices_before[name]+delta):v.co=co
    obj.data.update()
    if obj.get('collision_description_json'):
        c=json.loads(obj['collision_description_json'])
        for key in ['center_world_zero_m','proximal_sphere_center_world_zero_m']:
            c[key]=(np.array(c[key])+delta).tolist()
        # The whole fixed bone and its capsule translated together: local data is unchanged.
        obj['collision_description_json']=json.dumps(c)
rebuild_guides(rig)
for r in rows:r['position']=list(rig.data.bones[r['name']].head_local)
colliders=[json.loads(bpy.data.objects[r['object']]['collision_description_json'])
           for r in meshes if bpy.data.objects[r['object']].get('collision_description_json')]
rig['joint_table_json']=json.dumps(rows,ensure_ascii=False)
rig['collision_shapes_json']=json.dumps(colliders)
scene['waist_shoulder_horizontal_m']=.090
scene['shoulder_distance_revision']='waist_to_shoulder_90mm_v1'
scene['layout_revision']='waist_90_50_forearm40_wrist20_ankle70_v2'
for n,before in bone_before.items():
    expected=before.copy()
    if n in moved_bones:expected[:3,3]+=delta
    error=abs(np.array(rig.data.bones[n].matrix_local)-expected)
    assert error[:3,3].max()<1e-8 and error[:3,:3].max()<5e-7,n
for n,before in vertices_before.items():
    expected=before+delta if n in moved_meshes else before
    actual=np.array([v.co for v in bpy.data.objects[n].data.vertices],dtype=np.float64)
    assert np.max(abs(actual-expected))<1e-8,n
assert actions()==actions_before and limits()==limits_before
notes=bpy.data.texts['START_HERE_使用说明'];notes.clear()
notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
scene.frame_set(old_frame);bpy.context.view_layer.update()
if old_mode!='OBJECT':bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash,'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert digest(SOURCE)==source_hash

# Compose this translation after the previous layout. Keep its pre-layout
# vertices/matrices as the shared audit baseline; the prior audit is backed up.
path=ROOT/'robot_authoring/inspection/layout_revision.json'
audit=json.loads(path.read_text(encoding='utf8'))
for n in moved_bones:audit['bone_translations_m'][n]=(np.array(audit['bone_translations_m'][n])+delta).tolist()
for n in moved_meshes:
    m=np.array(audit['mesh_transforms_world'][n]);m[:3,3]+=delta
    audit['mesh_transforms_world'][n]=m.tolist()
audit['front_translation_x_m']+=dx
audit.update(revision=scene['layout_revision'],output_rig_sha256=digest(TARGET),collision_shapes=colliders)
audit.setdefault('amendments',[]).append({'revision':scene['shoulder_distance_revision'],
    'input_rig_sha256':before_hash,'output_rig_sha256':digest(TARGET),'translation_m':delta.tolist(),
    'moved_bones':moved_bones,'moved_meshes':moved_meshes,'actions_preserved':True,'limits_preserved':True})
path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'robot_authoring/inspection/build_report.json';build=json.loads(path.read_text(encoding='utf8'))
build.update(layout_revision=scene['layout_revision'],layout_adjustment=audit,joint_rows=rows,collision_shapes=colliders)
for key,part in build['parts'].items():
    if key in FRONT_PARTS:part['zero_pivot_before_global_shift_m'][0]+=dx
path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
from export_description import export_robot
export_robot()
bpy.ops.object.mode_set(mode='OBJECT');scene.frame_set(1)
vv=np.array([v.co for r in meshes for v in bpy.data.objects[r['object']].data.vertices])
center=Vector(((vv[:,0].min()+vv[:,0].max())/2,0,(vv[:,2].min()+vv[:,2].max())/2))
camera=scene.camera
for label,offset in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0))]:
    camera.hide_set(False);camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('WAIST_TO_SHOULDER_90MM_SAVED_AND_EXPORTED')
