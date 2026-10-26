"""Apply the approved latest changes to the saved rig, preserving source files."""
import bpy,json,hashlib,sys
import numpy as np
from pathlib import Path
from mathutils import Vector
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'robot_authoring'))
from straight_limb_geometry import apply_straight_limbs
from mount_collision import components,interference
TARGET=ROOT/'blender/sable_rig.blend'
BACKUP=ROOT/'待删除/2026-10-25_before_straight_limbs_manual_roll_sync'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
before_hash=digest(TARGET)
assert before_hash==digest(BACKUP/'blender/sable_rig.blend'),'Concurrent edit since backup'
protected={p:digest(ROOT/p) for p in ['blender/draft.blend','blender/sable_rig_final_2026-10-25.blend','robot_authoring/source/original_draft_2026-10-24.blend']}
rig,scene=bpy.data.objects['SableRobot'],bpy.context.scene
assert np.max(abs(np.array(rig.matrix_world)-np.eye(4)))<1e-8
old_frame=scene.frame_current
def actions():
    return [{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
      'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points]}
      for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]} for a in bpy.data.actions]
def limits():
    return {p.name:[[c.name,c.type]+[getattr(c,k,None) for k in ['min_x','max_x','min_y','max_y','min_z','max_z']] for c in p.constraints] for p in rig.pose.bones}
ab,lb=actions(),limits()
scene.frame_set(1);bpy.context.view_layer.update()
assert all(np.linalg.norm(p.rotation_euler)<1e-7 and np.linalg.norm(p.location)<1e-7 for p in rig.pose.bones),'Nonzero authored frame 1'
# Confirm the only changes since the previous revision are the two manual
# object translations. Any additional mesh/rest-frame edits require inspection.
previous=json.loads((ROOT/'robot_authoring/inspection/motor_offsets.json').read_text(encoding='utf8'))
manual_bones={'L_shoulder_roll':[.008,0,0],'spine_roll':[-.005,0,0]}
for name,values in previous['mesh_vertices_before'].items():
    expected=np.array(values)+np.array(previous['mesh_translations_m'].get(name,[0,0,0]))
    assert np.max(abs(np.array([v.co for v in bpy.data.objects[name].data.vertices])-expected))<1e-8,name
for name,values in previous['bone_rest_matrices_before'].items():
    expected=np.array(values);expected[:3,3]+=np.array(previous['bone_translations_m'].get(name,[0,0,0]))
    expected[:3,3]+=np.array(manual_bones.get(name,[0,0,0]))
    error=abs(np.array(rig.data.bones[name].matrix_local)-expected)
    assert error[:3,3].max()<1e-8 and error[:3,:3].max()<5e-7,name
audit=apply_straight_limbs(rig,scene,existing_roll_bone_deltas=manual_bones)
for name,values in audit['bone_rest_matrices_before'].items():
    expected=np.array(values);expected[:3,3]+=np.array(audit['bone_translations_m'].get(name,[0,0,0]))
    error=abs(np.array(rig.data.bones[name].matrix_local)-expected)
    assert error[:3,3].max()<1e-8 and error[:3,:3].max()<5e-7,name
for name,values in audit['mesh_vertices_before'].items():
    expected=np.array(values)+np.array(audit['mesh_translations_m'].get(name,[0,0,0]))
    assert np.max(abs(np.array([v.co for v in bpy.data.objects[name].data.vertices])-expected))<1e-8,name
assert actions()==ab and limits()==lb
audit['shoulder_zero_clearance']={s:interference(components(bpy.data.objects[s+'_shoulder_roll_visual']),
    components(bpy.data.objects[s+'_shoulder_pitch_visual'])) for s in ['L','R']}
notes=bpy.data.texts['START_HERE_使用说明'];notes.clear();notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
scene.frame_set(old_frame);bpy.context.view_layer.update()
for o in bpy.context.selected_objects:o.select_set(False)
rig.select_set(True);bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='POSE')
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash,'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert all(digest(ROOT/p)==h for p,h in protected.items())
audit.update(input_rig_sha256=before_hash,output_rig_sha256=digest(TARGET),protected_files_sha256=protected,actions_before=ab,limits_before=lb)
(ROOT/'robot_authoring/inspection/straight_limbs.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'robot_authoring/inspection/build_report.json';build=json.loads(path.read_text(encoding='utf8'))
build.update(straight_limb_revision=audit['revision'],straight_limbs=audit,joint_rows=json.loads(rig['joint_table_json']),collision_shapes=json.loads(rig['collision_shapes_json']))
path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
from export_description import export_robot
export_robot()
bpy.ops.object.mode_set(mode='OBJECT');scene.frame_set(1)
meshes=json.loads(rig['mesh_table_json']);vv=np.array([v.co for r in meshes for v in bpy.data.objects[r['object']].data.vertices])
center=Vector(((vv[:,0].min()+vv[:,0].max())/2,0,(vv[:,2].min()+vv[:,2].max())/2));camera=scene.camera
bpy.data.objects['BodyReference_visual'].hide_render=True
camera.data.ortho_scale=.29
for label,offset in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0)),('top',(0,0,.45))]:
    camera.hide_set(False);camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'));bpy.ops.render.render(write_still=True)
print('STRAIGHT_LIMBS_SAVED',json.dumps(audit['bone_translations_m']))
print('SHOULDER_CLEARANCE',json.dumps({s:r['penetrating'] for s,r in audit['shoulder_zero_clearance'].items()}))
