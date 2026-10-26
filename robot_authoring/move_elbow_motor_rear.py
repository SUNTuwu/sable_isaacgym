"""Rear-mount both elbow motor housings, preserving the output pivots and rig."""
import bpy
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from mathutils import Matrix, Vector

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'blender/sable_rig.blend'
SOURCE=ROOT/'blender/draft.blend'
BACKUP=ROOT/'待删除/2026-10-25_before_elbow_motor_rear/blender/sable_rig.blend'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
before_hash,source_hash=digest(TARGET),digest(SOURCE)
assert before_hash==digest(BACKUP),'Concurrent change since backup'
rig,scene=bpy.data.objects['SableRobot'],bpy.context.scene
assert not scene.get('elbow_motor_mount_revision'),'Already applied'
assert np.max(abs(np.array(rig.matrix_world)-np.eye(4)))<1e-8
old_frame,old_mode=scene.frame_current,rig.mode
meshes=json.loads(rig['mesh_table_json'])
inventory=json.loads((ROOT/'robot_authoring/inspection/source_inventory.json').read_text(encoding='utf8'))
sources={r['name']:r for r in inventory['objects']}


def actions_snapshot():
    return [{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
        'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points]}
        for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]}
        for a in bpy.data.actions]


def limits_snapshot():
    return {p.name:[[c.name,c.type]+[getattr(c,k,None) for k in
        ['min_x','max_x','min_y','max_y','min_z','max_z']] for c in p.constraints] for p in rig.pose.bones}


def mesh_hash(name):
    return hashlib.sha256(np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32).tobytes()).hexdigest()


changed={'L_elbow_pitch_visual','R_elbow_pitch_visual'}
protected={r['object']:mesh_hash(r['object']) for r in meshes if r['object'] not in changed}
bones_before={b.name:[list(row) for row in b.matrix_local] for b in rig.data.bones}
actions_before,limits_before=actions_snapshot(),limits_snapshot()
vertices_before={n:[list(v.co) for v in bpy.data.objects[n].data.vertices] for n in changed}
transforms,mounting={},{}
scene.frame_set(1)
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
for side in ['L','R']:
    name=side+'_elbow_pitch_visual';obj=bpy.data.objects[name]
    pivot=rig.data.bones[side+'_elbow_pitch'].head_local.copy()
    ids=next(c['indices'] for c in sources[obj['source_object']]['components'] if c['count']==144)
    offset=np.array(vertices_before[name])[ids].mean(0)-np.array(pivot)
    assert offset[0]>0 and offset[2]>0,'Expected existing front-upper motor layout'
    # A proper rotation about the existing output axis makes the side-view
    # center offset (-x, z), keeping the complete housing a rigid solid.
    angle=-2*math.atan2(offset[0],offset[2])
    rotation=Matrix.Rotation(angle,4,'Y')
    transform=Matrix.Translation(pivot)@rotation@Matrix.Translation(-pivot)
    for v in obj.data.vertices:v.co=transform@v.co
    obj.data.update()
    transforms[name]=[list(row) for row in transform]
    mounting[side]={'rotation_world_y_deg':math.degrees(angle),'pivot_world_m':list(pivot),
        'housing_offset_before_m':offset.tolist(),
        'housing_offset_after_m':list(rotation.to_3x3()@Vector(offset)),
        'method':'rigid rotation about unchanged output axis; mirrored side-view center placement'}
assert actions_snapshot()==actions_before and limits_snapshot()==limits_before
assert all(mesh_hash(n)==h for n,h in protected.items())
assert all([list(row) for row in rig.data.bones[n].matrix_local]==m for n,m in bones_before.items())
scene['elbow_motor_mount_revision']='elbow_housings_rear_side_v1'
scene['elbow_motor_mounting_json']=json.dumps(mounting)
notes=bpy.data.texts['START_HERE_使用说明'];notes.clear()
notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
scene.frame_set(old_frame);bpy.context.view_layer.update()
bpy.context.view_layer.objects.active=rig
if old_mode!='OBJECT':bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash,'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert digest(SOURCE)==source_hash
audit={'revision':scene['elbow_motor_mount_revision'],'input_rig_sha256':before_hash,'output_rig_sha256':digest(TARGET),
    'source_sha256':source_hash,'source_unchanged':True,'mounting':mounting,'mesh_transforms_world':transforms,
    'mesh_vertices_before':vertices_before,'protected_mesh_vertex_sha256':protected,
    'bone_rest_matrices_before':bones_before,'actions_before':actions_before,'limits_before':limits_before,
    'keyframes_preserved':True,'limits_preserved':True}
(ROOT/'robot_authoring/inspection/elbow_motor_rear.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'robot_authoring/inspection/build_report.json'
build=json.loads(path.read_text(encoding='utf8'))
rotation=np.array(Matrix.Rotation(math.radians(mounting['L']['rotation_world_y_deg']),3,'Y'))
build['parts']['elbow_pitch']['rotation_source_to_zero']=(rotation@np.array(build['parts']['elbow_pitch']['rotation_source_to_zero'])).tolist()
build.update(elbow_motor_mount_revision=scene['elbow_motor_mount_revision'],elbow_motor_mounting=mounting)
path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import export_robot
export_robot()
print('ELBOW_MOTORS_REAR',json.dumps(mounting))
bpy.ops.object.mode_set(mode='OBJECT');scene.frame_set(1)
camera=scene.camera
vv=np.array([v.co for row in meshes for v in bpy.data.objects[row['object']].data.vertices])
center=Vector(((vv[:,0].min()+vv[:,0].max())/2,0,(vv[:,2].min()+vv[:,2].max())/2))
for label,offset in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0))]:
    camera.hide_set(False);camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('ELBOW_REAR_EXPORT_COMPLETE')
