"""Move knee motor housings to the front-side layout and set shanks to 50 mm.

Apply once to the saved authored rig in an isolated background Blender process.
The knee output shafts and all upstream joints remain fixed.
"""
import bpy
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from mathutils import Matrix, Vector

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'blender/sable_rig.blend'
SOURCE = ROOT / 'blender/draft.blend'
BACKUP = ROOT / '待删除/2026-10-25_before_hind_motor_and_50mm_shank/blender/sable_rig.blend'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve() == TARGET.resolve()
before_hash, source_hash = digest(TARGET), digest(SOURCE)
assert before_hash == digest(BACKUP), 'Concurrent change since backup'
rig, scene = bpy.data.objects['SableRobot'], bpy.context.scene
assert not scene.get('hind_assembly_revision'), 'Already applied'
assert np.max(abs(np.array(rig.matrix_world)-np.eye(4))) < 1e-8
old_frame, old_mode = scene.frame_current, rig.mode
rows, meshes = json.loads(rig['joint_table_json']), json.loads(rig['mesh_table_json'])
inventory = json.loads((ROOT/'robot_authoring/inspection/source_inventory.json').read_text(encoding='utf8'))
source_objects = {r['name']: r for r in inventory['objects']}


def actions_snapshot():
    return [{'name': a.name, 'curves': [
        {'path': f.data_path, 'index': f.array_index,
         'keys': [[list(k.co), list(k.handle_left), list(k.handle_right), k.interpolation]
                  for k in f.keyframe_points]}
        for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]}
        for a in bpy.data.actions]


def limits_snapshot():
    return {p.name: [[c.name,c.type]+[getattr(c,k,None) for k in
        ['min_x','max_x','min_y','max_y','min_z','max_z']] for c in p.constraints] for p in rig.pose.bones}


bone_before = {b.name: [list(x) for x in b.matrix_local] for b in rig.data.bones}
mesh_before = {r['object']: [list(v.co) for v in bpy.data.objects[r['object']].data.vertices] for r in meshes}
actions_before, limits_before = actions_snapshot(), limits_snapshot()
mesh_transforms, bone_deltas, mounting, lengths_before = {}, {}, {}, {}
for side in ['L','R']:
    a, b = side+'_elbow_pitch_visual', side+'_knee_pitch_visual'
    pa = np.array(rig.data.bones[side+'_elbow_pitch'].head_local)
    pb = np.array(rig.data.bones[side+'_knee_pitch'].head_local)
    offsets = []
    for name, pivot in [(a,pa),(b,pb)]:
        obj = bpy.data.objects[name]
        # Source object is explicitly named; shaft axes were already derived from
        # its 64-vertex cylinder. The 144-vertex housing supplies only a layout vector.
        components = source_objects[obj['source_object']]['components']
        ids = next(c['indices'] for c in components if c['count']==144)
        offsets.append(np.array(mesh_before[name])[ids].mean(0)-pivot)
    angle = math.atan2(offsets[0][0], offsets[0][2])-math.atan2(offsets[1][0], offsets[1][2])
    rotation = Matrix.Rotation(angle,4,'Y')
    transform = Matrix.Translation(Vector(pb)) @ rotation @ Matrix.Translation(-Vector(pb))
    mesh_transforms[b] = transform
    mounting[side] = {'rotation_world_y_deg': math.degrees(angle),
        'pivot_world_m': pb.tolist(), 'reference_elbow_housing_offset_m': offsets[0].tolist(),
        'knee_housing_offset_before_m': offsets[1].tolist(),
        'knee_housing_offset_after_m': list(rotation.to_3x3()@Vector(offsets[1]))}
    ankle = side+'_ankle_fixed'
    old = rig.data.bones[ankle].head_local.copy()
    knee = rig.data.bones[side+'_knee_pitch'].head_local
    lengths_before[side] = (old-knee).length
    delta = Vector((0,0,knee.z-.050-old.z))
    bone_deltas[ankle] = delta
    for name in [side+'_ankle_fixed_visual',side+'_foot_visual']:
        mesh_transforms[name] = Matrix.Translation(delta)

scene.frame_set(1)
if bpy.context.object and bpy.context.object.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active=rig
bpy.ops.object.mode_set(mode='EDIT')
for name,delta in bone_deltas.items():
    bone=rig.data.edit_bones[name]; bone.head+=delta; bone.tail+=delta
bpy.ops.object.mode_set(mode='OBJECT')
for name,transform in mesh_transforms.items():
    obj=bpy.data.objects[name]
    for v in obj.data.vertices: v.co=transform@v.co
    obj.data.update()

# Lengthen only the two display guide rods; preserve their binding/materials.
for side in ['L','R']:
    obj=bpy.data.objects['GUIDE_'+side+'_knee_pitch']
    start=rig.data.bones[side+'_knee_pitch'].head_local
    end=rig.data.bones[side+'_ankle_fixed'].head_local
    q=(end-start).to_track_quat('Z','Y')
    vertices=[p+q@Vector((.0014*math.cos(2*math.pi*i/12),.0014*math.sin(2*math.pi*i/12),0))
              for p in [start,end] for i in range(12)]
    assert len(obj.data.vertices)==24
    for v,p in zip(obj.data.vertices,vertices): v.co=p
    obj.data.update()

colliders=[]
for row in meshes:
    obj=bpy.data.objects[row['object']]
    if not obj.get('collision_description_json'): continue
    c=json.loads(obj['collision_description_json'])
    if obj.name in mesh_transforms:
        c['center_world_zero_m']=list(mesh_transforms[obj.name]@Vector(c['center_world_zero_m']))
        obj['collision_description_json']=json.dumps(c)
    colliders.append(c)
rig['collision_shapes_json']=json.dumps(colliders)
for row in rows: row['position']=list(rig.data.bones[row['name']].head_local)
rig['joint_table_json']=json.dumps(rows,ensure_ascii=False)
scene['hind_assembly_revision']='knee_motor_front_side_shank_50mm_v1'
scene['shank_target_length_m']=.050
scene['knee_motor_mounting_json']=json.dumps(mounting)
notes=bpy.data.texts['START_HERE_使用说明']; notes.clear()
notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
assert actions_snapshot()==actions_before and limits_snapshot()==limits_before
for name,values in bone_before.items():
    expected=np.array(values); expected[:3,3]+=np.array(bone_deltas.get(name,(0,0,0)))
    error=abs(np.array(rig.data.bones[name].matrix_local)-expected)
    assert error[:3,3].max()<1e-8 and error[:3,:3].max()<5e-7,name
for name,values in mesh_before.items():
    expected=np.array([mesh_transforms[name]@Vector(v) for v in values]) if name in mesh_transforms else np.array(values)
    assert np.max(abs(np.array([v.co for v in bpy.data.objects[name].data.vertices])-expected))<1e-8,name
for side in ['L','R']:
    assert abs((rig.data.bones[side+'_knee_pitch'].head_local-rig.data.bones[side+'_ankle_fixed'].head_local).length-.050)<1e-8
scene.frame_set(old_frame); bpy.context.view_layer.update()
if old_mode!='OBJECT': bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash,'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert digest(SOURCE)==source_hash
audit={'revision':scene['hind_assembly_revision'],'input_rig_sha256':before_hash,'output_rig_sha256':digest(TARGET),
       'source_sha256':source_hash,'source_unchanged':True,'target_shank_length_m':.050,
       'shank_lengths_before_m':lengths_before,'mounting':mounting,
       'mesh_transforms_world':{n:[list(x) for x in m] for n,m in mesh_transforms.items()},
       'bone_translations_m':{n:list(d) for n,d in bone_deltas.items()},
       'bone_rest_matrices_before':bone_before,'mesh_vertices_before':mesh_before,
       'actions_before':actions_before,'limits_before':limits_before,'keyframes_preserved':True,'limits_preserved':True}
(ROOT/'robot_authoring/inspection/hind_assembly.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'robot_authoring/inspection/build_report.json'
build=json.loads(path.read_text(encoding='utf8'))
build.update(joint_rows=rows,collision_shapes=colliders,hind_assembly_revision=scene['hind_assembly_revision'],
             shank_target_length_m=.050,knee_motor_mounting=mounting)
for key in ['ankle_fixed','foot']: build['parts'][key]['zero_pivot_before_global_shift_m'][2]+=bone_deltas['L_ankle_fixed'].z
rot=np.array(Matrix.Rotation(math.radians(mounting['L']['rotation_world_y_deg']),3,'Y'))
build['parts']['knee_pitch']['rotation_source_to_zero']=(rot@np.array(build['parts']['knee_pitch']['rotation_source_to_zero'])).tolist()
path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import export_robot
export_robot()
print('HIND_ASSEMBLY_UPDATED',json.dumps(mounting),'SHANK_LENGTH_M',.050)
bpy.ops.object.mode_set(mode='OBJECT'); scene.frame_set(1)
camera=scene.camera
vv=np.array([v.co for row in meshes for v in bpy.data.objects[row['object']].data.vertices])
center=Vector(((vv[:,0].min()+vv[:,0].max())/2,0,(vv[:,2].min()+vv[:,2].max())/2))
for label,offset in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0))]:
    camera.hide_set(False); camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('HIND_ASSEMBLY_EXPORT_COMPLETE')
