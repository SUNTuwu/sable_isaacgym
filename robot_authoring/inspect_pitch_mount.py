import bpy,json,hashlib
import numpy as np
from pathlib import Path
r=bpy.data.objects['SableRobot'];s=bpy.context.scene
root=Path(__file__).resolve().parents[1]
data={'file':bpy.data.filepath,'frame':s.frame_current,'mode':r.mode,
      'bones':{b.name:{'rest':[list(x) for x in b.matrix_local],'pose_rotation':list(r.pose.bones[b.name].rotation_euler),
                     'pose_location':list(r.pose.bones[b.name].location)} for b in r.data.bones},
      'actions':[{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
        'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points]}
        for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]} for a in bpy.data.actions],
      'mesh_changes':{}}
audit=json.loads((root/'robot_authoring/inspection/layout_revision.json').read_text(encoding='utf8'))
for row in json.loads(r['mesh_table_json']):
    n=row['object'];vv=np.array([v.co for v in bpy.data.objects[n].data.vertices],dtype=np.float64)
    m=np.array(audit['mesh_transforms_world'].get(n,np.eye(4)))
    expected=np.array(audit['mesh_vertices_before'][n])@m[:3,:3].T+m[:3,3]
    data['mesh_changes'][n]=float(abs(vv-expected).max())
data['actions_equal_prior']=data['actions']==audit['actions_before']
(root/'robot_authoring/inspection/current_before_pitch_mount.json').write_text(json.dumps(data,indent=2),encoding='utf8')
print('CURRENT',json.dumps({'frame':data['frame'],'mode':data['mode'],'actions_equal_prior':data['actions_equal_prior'],
    'mesh_max_error':max(data['mesh_changes'].values()),
    'nonzero_pose':{n:v['pose_rotation'] for n,v in data['bones'].items() if np.linalg.norm(v['pose_rotation'])>1e-6}}))
for n in ['L_shoulder_roll','R_shoulder_roll','L_shoulder_pitch','R_shoulder_pitch']:
    print(n,json.dumps(data['bones'][n]))
