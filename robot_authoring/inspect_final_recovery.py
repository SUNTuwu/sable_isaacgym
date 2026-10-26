"""Read-only comparison with the last accepted authoring revision."""
import bpy,json,hashlib
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rig=bpy.data.objects.get('SableRobot');scene=bpy.context.scene
audit=json.loads((ROOT/'robot_authoring/inspection/shoulder_pitch_mount.json').read_text(encoding='utf8'))
report={'file':bpy.data.filepath,'sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        'frame':scene.frame_current,'rig_present':rig is not None,'pitch_revision':scene.get('shoulder_pitch_mount_revision')}
if rig:
    report['mode']=rig.mode
    report['changed_meshes']={};report['changed_bones']={}
    for name,vertices in audit['mesh_vertices_before'].items():
        obj=bpy.data.objects.get(name)
        if obj is None:report['changed_meshes'][name]='missing';continue
        m=np.array(audit['mesh_transforms_world'].get(name,np.eye(4)))
        expected=np.array(vertices)@m[:3,:3].T+m[:3,3]
        actual=np.array([v.co for v in obj.data.vertices],dtype=np.float64)
        error=float(np.max(abs(actual-expected))) if actual.shape==expected.shape else None
        if error is None or error>1e-8:report['changed_meshes'][name]=error
    for name,values in audit['bone_rest_matrices_before'].items():
        if name not in rig.data.bones:report['changed_bones'][name]='missing';continue
        m=np.array(audit['bone_transforms_world'].get(name,np.eye(4)))
        error=abs(np.array(rig.data.bones[name].matrix_local)-m@np.array(values))
        if error[:3,3].max()>1e-8 or error[:3,:3].max()>5e-7:report['changed_bones'][name]=error.max().item()
    actions=[{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
        'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points]}
        for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]} for a in bpy.data.actions]
    limits={p.name:[[c.name,c.type]+[getattr(c,k,None) for k in
        ['min_x','max_x','min_y','max_y','min_z','max_z']] for c in p.constraints] for p in rig.pose.bones}
    report['actions_equal_final']=actions==audit['actions_before']
    report['limits_equal_final']=limits==audit['limits_before']
    report['nonzero_pose']={p.name:list(p.rotation_euler) for p in rig.pose.bones if np.linalg.norm(p.rotation_euler)>1e-6}
print('RECOVERY_INSPECTION',json.dumps(report))
