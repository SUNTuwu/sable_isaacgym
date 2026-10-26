import bpy,json
import numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent
audit=json.loads((p/'inspection/motor_offsets.json').read_text(encoding='utf8'))
rig=bpy.data.objects['SableRobot']
print('ROLL_CONTROL_LENGTHS',json.dumps({n:{'head':list(rig.data.bones[n].head_local),'tail':list(rig.data.bones[n].tail_local),'length':rig.data.bones[n].length} for n in ['L_shoulder_roll','R_shoulder_roll','spine_roll']}))
for name,values in audit['bone_rest_matrices_before'].items():
    expected=np.array(values);expected[:3,3]+=np.array(audit['bone_translations_m'].get(name,[0,0,0]))
    actual=np.array(rig.data.bones[name].matrix_local);error=abs(actual-expected)
    if error[:3,3].max()>1e-8 or error[:3,:3].max()>5e-7:
        print('BONE_DIFF',name,json.dumps({'before':expected.tolist(),'actual':actual.tolist(),'translation':(actual[:3,3]-expected[:3,3]).tolist(),'basis_error':float(error[:3,:3].max())}))
for name,values in audit['mesh_vertices_before'].items():
    expected=np.array(values)+np.array(audit['mesh_translations_m'].get(name,[0,0,0]))
    actual=np.array([v.co for v in bpy.data.objects[name].data.vertices]);error=abs(actual-expected)
    if error.max()>1e-8:print('MESH_DIFF',name,float(error.max()))
