"""Reopen and independently verify powered ankle geometry, saved poses and USD exports."""
import bpy,json,math,sys,hashlib
from pathlib import Path
import numpy as np
from mathutils import Vector
from pxr import Usd,UsdGeom,UsdSkel,UsdPhysics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/model_sync/20261026_powered_ankles'
sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import effective_angle
r=bpy.data.objects['SableRobot'];s=bpy.context.scene
j=json.loads(r['joint_table_json']); meshes=json.loads(r['mesh_table_json']);cs=json.loads(r['collision_shapes_json'])
before=json.loads((OUT/'scene_before.json').read_text());after=json.loads((OUT/'scene_after.json').read_text())
checks={};data={}
def check(n,c):
 checks[n]=bool(c)
 assert c,n
def verts(o):
 ev=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=ev.to_mesh()
 try:return np.asarray([ev.matrix_world@v.co for v in m.vertices])
 finally:ev.to_mesh_clear()
check('15_independent_actuators',sum(x['actuated'] for x in j)==15 and r['independent_motor_count']==15)
check('ankle_control_metadata',all(r.pose.bones[side+'_ankle_pitch']['actuated'] and r.pose.bones[side+'_ankle_pitch']['joint_type']=='revolute' for side in ['L','R']))
check('16_hinges_2_fixed_wrists',sum(x['kind']=='revolute' for x in j)==16 and sum(x['kind']=='fixed' for x in j)==2)
for n,v in after['bones'].items():check('saved_rest_'+n,np.max(abs(np.asarray(r.data.bones[n].matrix_local)-v['matrix']))<1e-8)
for n,v in after['objects'].items():
 o=bpy.data.objects[n]
 if o.type=='MESH':check('saved_mesh_'+n,hashlib.sha256(np.asarray([v.co for v in o.data.vertices],dtype=np.float32).tobytes()).hexdigest()==v['vertices'])
oldposes=json.loads((OUT/'before/robot_authoring/exports/poses.json').read_text())['poses']
newposes=json.loads((OUT/'exports/poses.json').read_text())['poses']
check('100_exported_pose_frames',len(oldposes)==len(newposes)==100)
for old,new in zip(oldposes,newposes):
 for name,q in old['joint_angles_rad'].items():check('preserved_pose_'+str(old['frame'])+'_'+name,abs(new['joint_angles_rad'][name]-q)<2e-6)
 for side in ['L','R']:check('saved_ankle70_frame_'+str(new['frame'])+side,abs(new['joint_angles_deg'][side+'_ankle_pitch']-70)<2e-4)
s.frame_set(5);bpy.context.view_layer.update()
check('frame5_standard_ankles70',all(abs(math.degrees(effective_angle(r,side+'_ankle_pitch'))-70)<2e-4 for side in ['L','R']))
# Detach animation only in this unsaved validation process to exercise user limits.
r.animation_data.action=None
source=json.loads((ROOT/'robot_authoring/inspection/source_inventory.json').read_text())
source_motor=next(x for x in source['objects'] if x['name']=='膝关节pitch-dc1628')
ids=next(x['indices'] for x in source_motor['components'] if x['count']==64)
for side in ['L','R']:
 name=side+'_ankle_pitch';pb=r.pose.bones[name]; knee=r.data.bones[side+'_knee_pitch']; ankle=r.data.bones[name]
 check('shank55_'+side,abs((ankle.head_local-knee.head_local).length-.055)<1e-7)
 check('ankle_axis_minus_y_'+side,(ankle.matrix_local.to_3x3()@Vector((0,1,0))-Vector((0,-1,0))).length<1e-6)
 motor=bpy.data.objects[name+'_visual']; template=bpy.data.objects[side+'_knee_pitch_visual']
 mv=np.asarray([v.co for v in motor.data.vertices]);tv=np.asarray([v.co for v in template.data.vertices]);shift=np.asarray(ankle.head_local-knee.head_local)
 check('rigid_knee_motor_copy_'+side,np.max(abs(mv-tv-shift))<1e-8)
 pts=mv[ids];_,_,vh=np.linalg.svd(pts-pts.mean(0));axis=vh[-1]
 # Cylinder shaft is shortest PCA axis; verify named geometry, not reference cylinders.
 check('named_motor_axis_parallel_'+side,abs(axis[1])>1-1e-5)
 check('named_motor_center_at_ankle_'+side,np.linalg.norm(pts.mean(0)-np.asarray(ankle.head_local))<1e-6)
 c=next(c for c in cs if c['bone']==name)
 check('capsule40_radius5_'+side,abs(c['total_length_m']-.040)<1e-8 and abs(c['radius_m']-.005)<1e-7)
 check('capsule_proximal_at_shaft_'+side,Vector(c['endpoints_bone_m'][0]).length<1e-7)
 foot=bpy.data.objects[side+'_foot_visual'];edges={}
 for p in foot.data.polygons:
  for e in p.edge_keys:edges[e]=edges.get(e,0)+1
 check('closed_capsule_'+side,all(v==2 for v in edges.values()))
 pts=np.asarray([v.co for v in foot.data.vertices])-np.asarray(c['center_world_zero_m']);ax=np.asarray(c['axis_world_zero']);pr=np.clip(pts@ax,-c['cylinder_length_m']/2,c['cylinder_length_m']/2)
 check('analytic_capsule_matches_mesh_'+side,np.max(abs(np.linalg.norm(pts-pr[:,None]*ax,axis=1)-c['radius_m']))<1e-7)
 sweep=[];housing=verts(motor)
 for requested,expected in [(-10,0),(0,0),(70,70),(160,160),(170,160)]:
  pb.rotation_euler.y=math.radians(requested);bpy.context.view_layer.update()
  q=math.degrees(effective_angle(r,name));check('clamp_'+side+str(requested),abs(q-expected)<2e-4)
  axisworld=(pb.matrix.to_3x3()@Vector(c['axis_bone'])).normalized()
  parentrelative=(pb.parent.matrix.to_3x3().inverted()@axisworld).normalized()
  want=Vector((math.sin(math.radians(expected)),0,-math.cos(math.radians(expected))))
  check('direction_'+side+str(requested),(parentrelative-want).length<1e-6)
  check('housing_stays_on_shank_'+side+str(requested),np.max(abs(verts(motor)-housing))<1e-7)
  evaluated=verts(foot);center=Vector(evaluated.mean(0));pred=pb.matrix@Vector(c['center_bone_m'])
  check('skin_follows_ankle_'+side+str(requested),(center-pred).length<1e-7)
  sweep.append({'requested_deg':requested,'actual_deg':q,'axis_relative_shank':list(parentrelative)})
 pb.rotation_euler.y=math.radians(70);data[side]=sweep
for angle in [-15,0,15]:
 r.pose.bones['hip_roll'].rotation_euler.y=math.radians(angle);bpy.context.view_layer.update()
 check('opposed_hip_roll_'+str(angle),abs(effective_angle(r,'hip_roll')+effective_angle(r,'R_hip_roll_mimic'))<1e-6)
for filename in ['robot_visual.usdc','foot_colliders.usda']:
 stage=Usd.Stage.Open(str(OUT/'exports'/filename));check('usd_reopens_'+filename,bool(stage))
 if filename=='robot_visual.usdc':
  check('usd_meshes22',sum(p.IsA(UsdGeom.Mesh) for p in stage.Traverse())==22)
  sk=next(UsdSkel.Skeleton(p) for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton));names=list(sk.GetJointsAttr().Get())
  check('usd_ankles_actuated_bones',all(any(n.endswith(side+'_ankle_pitch') for n in names) for side in ['L','R']))
 else:check('usd_four_capsules',sum(p.IsA(UsdGeom.Capsule) and p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())==4)
for f,h in json.loads((OUT/'before_manifest.json').read_text()).items():
 if f=='blender/draft.blend':check('original_preserved_'+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h)
 check('backup_preserved_'+f,hashlib.sha256((OUT/'before'/f).read_bytes()).hexdigest()==h)
(OUT/'validation.json').write_text(json.dumps({'all_passed':all(checks.values()),'check_count':len(checks),'checks':checks,'angle_sweep':data,'scope':'static authored geometry, pose, limits and USD; no dynamics/hardware validation'},indent=2))
print('VALIDATED',len(checks),'checks')
