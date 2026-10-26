"""Apply independent mounting offsets to sable_rig; preserve new draft."""
import bpy,json,hashlib,sys
import numpy as np
from pathlib import Path
from mathutils import Vector
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'robot_authoring'))
from motor_offset_geometry import apply_motor_offsets,configure_chest_root
from mount_collision import components,interference
TARGET=ROOT/'blender/sable_rig.blend';DRAFT=ROOT/'blender/draft.blend'
SOURCE=ROOT/'robot_authoring/source/original_draft_2026-10-24.blend'
BACKUP=ROOT/'待删除/2026-10-25_before_independent_mount_offsets'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
before_hash,draft_hash,source_hash=digest(TARGET),digest(DRAFT),digest(SOURCE)
assert before_hash==digest(BACKUP/'blender/sable_rig.blend'),'Concurrent changes since backup'
assert draft_hash==digest(ROOT/'blender/sable_rig_final_2026-10-25.blend')
rig,scene=bpy.data.objects['SableRobot'],bpy.context.scene
assert np.max(abs(np.array(rig.matrix_world)-np.eye(4)))<1e-8
old_frame=scene.frame_current

def actions():
    return [{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
      'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points]}
      for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]} for a in bpy.data.actions]

def limits():
    return {p.name:[[c.name,c.type]+[getattr(c,k,None) for k in
      ['min_x','max_x','min_y','max_y','min_z','max_z']] for c in p.constraints] for p in rig.pose.bones}

actions_before,limits_before=actions(),limits()
scene.frame_set(1);audit=apply_motor_offsets(rig,scene)
for name,values in audit['bone_rest_matrices_before'].items():
    expected=np.array(values);expected[:3,3]+=np.array(audit['bone_translations_m'].get(name,[0,0,0]))
    error=abs(np.array(rig.data.bones[name].matrix_local)-expected)
    assert error[:3,3].max()<1e-8 and error[:3,:3].max()<5e-7,name
for name,values in audit['mesh_vertices_before'].items():
    expected=np.array(values)+np.array(audit['mesh_translations_m'].get(name,[0,0,0]))
    actual=np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float64)
    assert np.max(abs(actual-expected))<1e-8,name
assert actions()==actions_before and limits()==limits_before
audit['chest_root']=configure_chest_root(rig,scene,DRAFT)
assert actions()==actions_before and limits()==limits_before
audit['shoulder_zero_clearance']={s:interference(components(bpy.data.objects[s+'_shoulder_roll_visual']),
    components(bpy.data.objects[s+'_shoulder_pitch_visual'])) for s in ['L','R']}
scene['robot_geometry_source_file']='robot_authoring/source/original_draft_2026-10-24.blend'
scene['robot_pose_reference_file']='blender/draft.blend';scene['robot_pose_reference_sha256']=draft_hash
notes=bpy.data.texts['START_HERE_使用说明'];notes.clear()
notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
scene.frame_set(old_frame);bpy.context.view_layer.update()
for o in bpy.context.selected_objects:o.select_set(False)
rig.select_set(True);bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='POSE')
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash,'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert digest(DRAFT)==draft_hash and digest(SOURCE)==source_hash
audit.update(input_rig_sha256=before_hash,output_rig_sha256=digest(TARGET),source_sha256=source_hash,
             draft_reference_sha256=draft_hash,actions_before=actions_before,limits_before=limits_before)
(ROOT/'robot_authoring/inspection/motor_offsets.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'robot_authoring/inspection/build_report.json';build=json.loads(path.read_text(encoding='utf8'))
build.update(motor_offset_revision=audit['revision'],motor_offsets=audit,joint_rows=json.loads(rig['joint_table_json']),
             original_geometry_source='robot_authoring/source/original_draft_2026-10-24.blend',
             pose_reference='blender/draft.blend',pose_reference_sha256=draft_hash)
path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
from export_description import export_robot
export_robot()
bpy.ops.object.mode_set(mode='OBJECT');scene.frame_set(1)
meshes=json.loads(rig['mesh_table_json']);vv=np.array([v.co for r in meshes for v in bpy.data.objects[r['object']].data.vertices])
center=Vector(((vv[:,0].min()+vv[:,0].max())/2,0,(vv[:,2].min()+vv[:,2].max())/2));camera=scene.camera
for label,offset in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0)),('top',(0,0,.45))]:
    bpy.data.objects['BodyReference_visual'].hide_render=True
    camera.hide_set(False);camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'));bpy.ops.render.render(write_still=True)
bpy.data.objects['BodyReference_visual'].hide_render=False
scene.render.filepath=str(ROOT/'robot_authoring/preview_body_reference.png')
camera.location=center+Vector((0,.45,0));camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.ortho_scale=.42
bpy.ops.render.render(write_still=True)
print('INDEPENDENT_MOTOR_OFFSETS_SAVED',json.dumps(audit['bone_translations_m']))
print('SHOULDER_CLEARANCE',json.dumps({s:r['penetrating'] for s,r in audit['shoulder_zero_clearance'].items()}))
