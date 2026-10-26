"""One-time terminal geometry revision; preserves the saved rig and its action."""
import bpy
import hashlib
import json
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'robot_authoring'))
from capsule_geometry import TERMINALS, replace_terminal

TARGET=ROOT/'blender/sable_rig.blend'
SOURCE=ROOT/'blender/draft.blend'
BACKUP=ROOT/'待删除/2026-10-24_before_foot_capsules/blender/sable_rig.blend'
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before_hash=digest(TARGET)
assert before_hash==digest(BACKUP), 'Saved rig changed since revision backup'
rig=bpy.data.objects['SableRobot']; scene=bpy.context.scene
assert not scene.get('terminal_capsule_revision'), 'Capsule correction already applied'
source_hash=digest(SOURCE)
old_frame=scene.frame_current; old_mode=rig.mode

def mesh_hash(obj):
    return hashlib.sha256(np.array([v.co for v in obj.data.vertices],dtype=np.float32).tobytes()).hexdigest()

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
protected={r['object']:mesh_hash(bpy.data.objects[r['object']]) for r in mesh_rows if r['object'] not in TERMINALS}
bones={b.name:[list(v) for v in b.matrix_local] for b in rig.data.bones}
actions=action_snapshot(); limits=limit_snapshot()
scene.frame_set(1); bpy.context.view_layer.update()
if bpy.context.object and bpy.context.object.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
colliders=[]
for row in mesh_rows:
    if row['object'] not in TERMINALS: continue
    obj=bpy.data.objects[row['object']]
    colliders.append(replace_terminal(obj,rig))
    row['authored_vertex_count']=len(obj.data.vertices)
    row['authored_face_count']=len(obj.data.polygons)
    row['geometry_revision']='terminal capsule; original center/axis/overall length retained'
assert len(colliders)==4
rig['mesh_table_json']=json.dumps(mesh_rows,ensure_ascii=False)
rig['collision_shapes_json']=json.dumps(colliders)
scene['terminal_capsule_revision']='terminal_capsules_v1'
notes=bpy.data.texts.get('START_HERE_使用说明')
notes.clear(); notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
scene.frame_set(old_frame); bpy.context.view_layer.update()
assert all(mesh_hash(bpy.data.objects[n])==h for n,h in protected.items())
assert bones=={b.name:[list(v) for v in b.matrix_local] for b in rig.data.bones}
assert actions==action_snapshot() and limits==limit_snapshot()
bpy.context.view_layer.objects.active=rig
if old_mode!='OBJECT': bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash, 'Another process saved the rig; refusing overwrite'
assert digest(SOURCE)==source_hash
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert digest(SOURCE)==source_hash
audit={'revision':'terminal_capsules_v1','source_sha256':source_hash,
    'input_rig_sha256':before_hash,'output_rig_sha256':digest(TARGET),'collision_shapes':colliders,
    'protected_mesh_vertex_sha256':protected,'protected_bone_rest_matrices':bones,
    'actions':actions,'limits':limits,'keyframes_preserved':True,'limits_preserved':True,
    'instruction':'User requested the four terminal boxes to become capsule collision shapes; preserve the rest of the rig.'}
(ROOT/'robot_authoring/inspection/terminal_capsules.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
report_path=ROOT/'robot_authoring/inspection/build_report.json'
report=json.loads(report_path.read_text(encoding='utf8'))
report.update(mesh_rows=mesh_rows,terminal_capsule_revision='terminal_capsules_v1',collision_shapes=colliders)
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('CAPSULES_SAVED',json.dumps({c['name']:{'length_mm':c['total_length_m']*1000,'radius_mm':c['radius_m']*1000} for c in colliders}))
from export_description import export_robot
export_robot()
bpy.ops.object.mode_set(mode='OBJECT')
scene.frame_set(1); bpy.context.view_layer.update()
camera=scene.camera
allverts=np.array([v.co for row in mesh_rows for v in bpy.data.objects[row['object']].data.vertices])
center=Vector(((allverts[:,0].min()+allverts[:,0].max())/2,0,(allverts[:,2].min()+allverts[:,2].max())/2))
for label,delta in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0))]:
    camera.hide_set(False); camera.location=center+Vector(delta)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('CAPSULE_EXPORT_AND_PREVIEWS_COMPLETE')
