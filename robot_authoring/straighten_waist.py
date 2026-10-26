"""One-time in-place rest-geometry correction, preserving action and limits."""
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
BACKUP=ROOT/'待删除/2026-10-24_waist_before_correction/blender/sable_rig.blend'
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
before_hash=hashlib.sha256(TARGET.read_bytes()).hexdigest()
assert before_hash==hashlib.sha256(BACKUP.read_bytes()).hexdigest(), 'Saved rig changed since revision backup'
rig=bpy.data.objects['SableRobot']; scene=bpy.context.scene
assert not scene.get('waist_alignment_revision'), 'Correction already applied'
source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
old_frame=scene.frame_current
scene.frame_set(1); bpy.context.view_layer.update()

def mesh_hash(obj):
    vertices=np.array([v.co for v in obj.data.vertices],dtype=np.float32)
    return hashlib.sha256(vertices.tobytes()).hexdigest()

def action_snapshot():
    result=[]
    for action in bpy.data.actions:
        curves=[]
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        curves.append({'path':fc.data_path,'index':fc.array_index,
                            'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in fc.keyframe_points]})
        result.append({'name':action.name,'curves':curves})
    return result

def limit_snapshot():
    return {p.name:[[c.name,c.type]+[getattr(c,k,None) for k in ['min_x','max_x','min_y','max_y','min_z','max_z']]
                    for c in p.constraints] for p in rig.pose.bones}

mesh_rows=json.loads(rig['mesh_table_json'])
changed_meshes={'spine_pitch_visual','L_shoulder_roll_visual','R_shoulder_roll_visual'}
protected_meshes={r['object']:mesh_hash(bpy.data.objects[r['object']]) for r in mesh_rows if r['object'] not in changed_meshes}
before_bones={b.name:{'head':list(b.head_local),'tail':list(b.tail_local)} for b in rig.data.bones}
before_actions=action_snapshot(); before_limits=limit_snapshot()
height=float(rig.data.bones['spine_roll'].head_local.z)
moved=['ROOT','spine_pitch','L_shoulder_roll','R_shoulder_roll']
deltas={name:height-float(rig.data.bones[name].head_local.z) for name in moved}

if bpy.context.object and bpy.context.object.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
for obj in bpy.context.selected_objects: obj.select_set(False)
rig.select_set(True); bpy.context.view_layer.objects.active=rig
bpy.ops.object.mode_set(mode='EDIT')
for name in moved:
    b=rig.data.edit_bones[name]; delta=Vector((0,0,deltas[name]))
    b.head+=delta; b.tail+=delta
bpy.ops.object.mode_set(mode='OBJECT')
for objname,bone in [('spine_pitch_visual','spine_pitch'),('L_shoulder_roll_visual','L_shoulder_roll'),('R_shoulder_roll_visual','R_shoulder_roll')]:
    obj=bpy.data.objects[objname]
    for v in obj.data.vertices: v.co.z+=deltas[bone]
    obj.data.update()

def replace_guide(name,bone_a,bone_b,radius):
    obj=bpy.data.objects[name]
    assert len(obj.vertex_groups)==1
    group_name=obj.vertex_groups[0].name
    a=rig.data.bones[bone_a].head_local.copy(); b=rig.data.bones[bone_b].head_local.copy()
    direction=b-a; q=direction.to_track_quat('Z','Y')
    verts=[]
    for endpoint in (a,b):
        for i in range(12):
            angle=2*math.pi*i/12
            verts.append(endpoint+q@Vector((radius*math.cos(angle),radius*math.sin(angle),0)))
    faces=[tuple(reversed(range(12))),tuple(range(12,24))]
    faces += [(i,(i+1)%12,(i+1)%12+12,i+12) for i in range(12)]
    old=obj.data; mesh=bpy.data.meshes.new(name+'_straight_waist')
    mesh.from_pydata(verts,[],faces); mesh.update()
    for mat in old.materials: mesh.materials.append(mat)
    obj.data=mesh
    group=obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
    group.add(list(range(24)),1.,'REPLACE')
    if old.users==0: bpy.data.meshes.remove(old)

replace_guide('GUIDE_waist','spine_pitch','spine_roll',.002)
for side in ['L','R']:
    replace_guide('GUIDE_'+side+'_chest','spine_pitch',side+'_shoulder_roll',.002)
    replace_guide('GUIDE_'+side+'_shoulder_roll',side+'_shoulder_roll',side+'_shoulder_pitch',.0014)

rows=json.loads(rig['joint_table_json'])
for row in rows: row['position']=list(rig.data.bones[row['name']].head_local)
rig['joint_table_json']=json.dumps(rows,ensure_ascii=False)
scene['waist_alignment_revision']='straight_waist_v1: user annotation overrides draft; rear spine-roll height is fixed'
scene['zero_pose_note']='Straight waist and limbs, rest-geometry correction; original draft preserved; rear assembly unchanged.'
text=bpy.data.texts.get('START_HERE_使用说明')
text.clear(); text.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
bpy.context.view_layer.update()

for name,digest in protected_meshes.items(): assert mesh_hash(bpy.data.objects[name])==digest,name
for name,before in before_bones.items():
    if name not in moved:
        b=rig.data.bones[name]
        assert (b.head_local-Vector(before['head'])).length<1e-8,name
        assert (b.tail_local-Vector(before['tail'])).length<1e-8,name
assert action_snapshot()==before_actions, 'Animation changed unexpectedly'
assert limit_snapshot()==before_limits, 'Limits changed unexpectedly'
assert max(abs(rig.data.bones[name].head_local.z-height) for name in moved)<1e-7

scene.frame_set(old_frame)
bpy.ops.object.mode_set(mode='POSE')
for pb in rig.pose.bones: pb.select=False
rig.pose.bones['spine_pitch'].select=True; rig.data.bones.active=rig.data.bones['spine_pitch']
bpy.context.preferences.filepaths.save_version=0
assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==before_hash, 'Another process saved this file; refusing to overwrite'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash

report_path=ROOT/'robot_authoring/inspection/build_report.json'
build=json.loads(report_path.read_text(encoding='utf8'))
build['joint_rows']=rows; build['base_height_m']=height
for key in ['spine_pitch','shoulder_roll']:
    build['parts'][key]['zero_pivot_before_global_shift_m'][2]=height-build['world_translation_m'][2]
build['waist_alignment_revision']=scene['waist_alignment_revision']
report_path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
correction={'revision':'straight_waist_v1','reference_bone':'spine_roll','reference_height_m':height,
    'input_rig_sha256':before_hash,'output_rig_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),
    'source_sha256':source_hash,'source_unchanged':True,'translation_z_mm':{k:v*1000 for k,v in deltas.items()},
    'protected_mesh_vertex_sha256':protected_meshes,
    'protected_bones':{k:v for k,v in before_bones.items() if k not in moved},
    'keyframes_preserved':True,'limits_preserved':True,'before_actions':before_actions,
    'instruction':'Use user annotated horizontal waist instead of raised waist in draft; rear motors unchanged'}
(ROOT/'robot_authoring/inspection/waist_correction.json').write_text(json.dumps(correction,ensure_ascii=False,indent=2),encoding='utf8')
print('WAIST_CORRECTED_MM',json.dumps(correction['translation_z_mm']))
print('PROTECTED_MESHES',len(protected_meshes),'KEYFRAMES_AND_LIMITS_PRESERVED')

sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import export_robot
export_robot()
bpy.ops.object.mode_set(mode='OBJECT')
camera=scene.camera
allverts=np.array([v.co for row in mesh_rows for v in bpy.data.objects[row['object']].data.vertices])
center=Vector(((allverts[:,0].min()+allverts[:,0].max())/2,0,(allverts[:,2].min()+allverts[:,2].max())/2))
for label,delta in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0))]:
    camera.hide_set(False); camera.location=center+Vector(delta)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('CORRECTION_AND_EXPORT_COMPLETE')
