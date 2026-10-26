"""Apply the user-approved inward pitch mounting to the current saved rig."""
import bpy,json,hashlib,sys
import numpy as np
from pathlib import Path
from mathutils import Vector
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'robot_authoring'))
from pitch_mount_geometry import apply_pitch_mount
from mount_collision import components,interference
TARGET=ROOT/'blender/sable_rig.blend';SOURCE=ROOT/'blender/draft.blend'
BACKUP=ROOT/'待删除/2026-10-25_before_shoulder_pitch_toein_5deg'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve()==TARGET.resolve()
before_hash,source_hash=digest(TARGET),digest(SOURCE)
assert before_hash==digest(BACKUP/'blender/sable_rig.blend'),'Concurrent saved changes since backup'
rig,scene=bpy.data.objects['SableRobot'],bpy.context.scene
old_frame,old_mode=scene.frame_current,rig.mode
assert np.max(abs(np.array(rig.matrix_world)-np.eye(4)))<1e-8

def actions():
    return [{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
      'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points]}
      for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]} for a in bpy.data.actions]

def limits():
    return {p.name:[[c.name,c.type]+[getattr(c,k,None) for k in
      ['min_x','max_x','min_y','max_y','min_z','max_z']] for c in p.constraints] for p in rig.pose.bones}

actions_before,limits_before=actions(),limits()
scene.frame_set(1);audit=apply_pitch_mount(rig,scene)
for n,values in audit['bone_rest_matrices_before'].items():
    expected=np.array(audit['bone_transforms_world'].get(n,np.eye(4)))@np.array(values)
    error=abs(np.array(rig.data.bones[n].matrix_local)-expected)
    assert error[:3,3].max()<1e-8 and error[:3,:3].max()<5e-7,(n,error.tolist(),np.array(rig.data.bones[n].matrix_local).tolist(),expected.tolist())
for n,values in audit['mesh_vertices_before'].items():
    m=np.array(audit['mesh_transforms_world'].get(n,np.eye(4)))
    expected=np.array(values)@m[:3,:3].T+m[:3,3]
    actual=np.array([v.co for v in bpy.data.objects[n].data.vertices],dtype=np.float64)
    assert np.max(abs(actual-expected))<1e-8,n
assert actions()==actions_before and limits()==limits_before
audit['final_clearance']={side:interference(components(bpy.data.objects[side+'_shoulder_roll_visual']),
    components(bpy.data.objects[side+'_shoulder_pitch_visual'])) for side in ['L','R']}
assert not any(v['penetrating'] for v in audit['final_clearance'].values())
notes=bpy.data.texts['START_HERE_使用说明'];notes.clear()
notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
scene.frame_set(old_frame);bpy.context.view_layer.update()
if old_mode!='OBJECT':bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version=0
assert digest(TARGET)==before_hash,'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET),check_existing=False)
assert digest(SOURCE)==source_hash
audit.update(input_rig_sha256=before_hash,output_rig_sha256=digest(TARGET),source_sha256=source_hash,
             actions_before=actions_before,limits_before=limits_before)
(ROOT/'robot_authoring/inspection/shoulder_pitch_mount.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'robot_authoring/inspection/build_report.json';build=json.loads(path.read_text(encoding='utf8'))
build.update(shoulder_pitch_mount_revision=audit['revision'],shoulder_pitch_mounting=audit,
             joint_rows=json.loads(rig['joint_table_json']),collision_shapes=audit['collision_shapes'])
for part in ['shoulder_pitch','elbow_pitch','wrist_fixed','hand']:
    build['parts'][part]['latest_installation_transform_authority']='shoulder_pitch_mounting; applied after the preceding source-to-zero transform'
path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
from export_description import export_robot
export_robot()
bpy.ops.object.mode_set(mode='OBJECT');scene.frame_set(1)
meshes=json.loads(rig['mesh_table_json'])
vv=np.array([v.co for r in meshes for v in bpy.data.objects[r['object']].data.vertices])
center=Vector(((vv[:,0].min()+vv[:,0].max())/2,0,(vv[:,2].min()+vv[:,2].max())/2))
camera=scene.camera
for label,offset in [('perspective',(.28,-.40,.23)),('side',(0,.45,0)),('front',(.45,0,0)),('top',(0,0,.45))]:
    camera.hide_set(False);camera.location=center+Vector(offset)
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('SHOULDER_PITCH_TOEIN5_OUTWARD2MM_SAVED')
