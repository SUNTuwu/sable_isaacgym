import bpy,json,hashlib,sys
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/model_sync/20261026_ankle_inline';r=bpy.data.objects['SableRobot'];s=bpy.context.scene
src=json.loads((ROOT/'robot_authoring/inspection/source_inventory.json').read_text());ids=next(c['indices'] for c in next(o for o in src['objects'] if o['name']=='膝关节pitch-dc1628')['components'] if c['count']==144)
checks={}
def check(n,c):checks[n]=bool(c);assert c,n
old=json.loads((OUT/'before/robot_authoring/exports/poses.json').read_text())['poses'];new=json.loads((OUT/'exports/poses.json').read_text())['poses']
for a,b in zip(old,new):
 check('unchanged_angles_frame'+str(a['frame']),all(abs(a['joint_angles_rad'][n]-q)<1e-6 for n,q in b['joint_angles_rad'].items()))
for frame in [1,5,6,50,100]:
 s.frame_set(frame);bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
 for side in ['L','R']:
  o=bpy.data.objects[side+'_ankle_pitch_visual'].evaluated_get(dg);mesh=o.to_mesh();center=np.mean([o.matrix_world@mesh.vertices[i].co for i in ids],axis=0);o.to_mesh_clear()
  ankle=np.array(r.matrix_world@r.pose.bones[side+'_ankle_pitch'].head);knee=np.array(r.matrix_world@r.pose.bones[side+'_knee_pitch'].head);axis=(knee-ankle)/np.linalg.norm(knee-ankle);distance=np.linalg.norm(np.cross(center-ankle,axis))
  check('housing_on_posed_shank_'+side+str(frame),distance<1e-7)
  check('shank55_'+side+str(frame),abs(np.linalg.norm(knee-ankle)-.055)<1e-7)
manifest=json.loads((OUT/'before_manifest.json').read_text());check('draft_preserved',hashlib.sha256((ROOT/'blender/draft.blend').read_bytes()).hexdigest()==manifest['blender/draft.blend'])
check('15_actuators',r['independent_motor_count']==15)
for side in ['L','R']:
 p=r.pose.bones[side+'_ankle_pitch'];c=next(c for c in p.constraints if c.type=='LIMIT_ROTATION');check('ankle_limits_'+side,abs(c.min_y)<1e-7 and abs(c.max_y-np.radians(160))<1e-6)
report={'passed':True,'checks':checks,'scope':'Reopened authored rig; ankle housing center on knee-ankle line in rest and sampled animated poses; all original joint animation preserved. No hardware assertion.'};(OUT/'validation.json').write_text(json.dumps(report,indent=2));print('INLINE_VALIDATION_OK',len(checks))
