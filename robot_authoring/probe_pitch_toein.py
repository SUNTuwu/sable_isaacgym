import bpy,json,math,sys
import numpy as np
from pathlib import Path
from mathutils import Matrix
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'robot_authoring'))
from mount_collision import components,interference
r=bpy.data.objects['SableRobot'];report={}
for side,sign in [('L',1),('R',-1)]:
    pivot=np.array(r.data.bones[side+'_shoulder_pitch'].head_local)
    rot=np.array(Matrix.Rotation(math.radians(-sign*5),3,'Z'),dtype=np.float64)
    roll=components(bpy.data.objects[side+'_shoulder_roll_visual'])
    report[side]={'before':interference(roll,components(bpy.data.objects[side+'_shoulder_pitch_visual'])),'probes':[]}
    for offset in [0.,.001,.002]:
        m=np.eye(4);m[:3,:3]=rot;m[:3,3]=pivot-rot@pivot+[0,sign*offset,0]
        test=interference(roll,components(bpy.data.objects[side+'_shoulder_pitch_visual'],m))
        report[side]['probes'].append({'outward_offset_m':offset,**test})
(ROOT/'robot_authoring/inspection/shoulder_toein_clearance_probe.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
