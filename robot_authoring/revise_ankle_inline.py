"""Rigid ankle housing remount onto knee-ankle line; shaft lines/poses unchanged."""
import bpy,json,hashlib,math,sys
import numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/model_sync/20261026_ankle_inline';sys.path.insert(0,str(ROOT/'robot_authoring'))
r=bpy.data.objects['SableRobot'];s=bpy.context.scene
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((OUT/'before_manifest.json').read_text());assert sha(ROOT/'blender/sable_rig.blend')==manifest['blender/sable_rig.blend']
assert not s.get('ankle_inline_revision')
source=json.loads((ROOT/'robot_authoring/inspection/source_inventory.json').read_text());motor=next(x for x in source['objects'] if x['name']=='膝关节pitch-dc1628')
housing_ids=next(c['indices'] for c in motor['components'] if c['count']==144);shaft_ids=next(c['indices'] for c in motor['components'] if c['count']==64)
def curves():return {a.name:{f'{f.data_path}|{f.array_index}':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points] for l in a.layers for st in l.strips for bag in st.channelbags for f in bag.fcurves} for a in bpy.data.actions}
keys=curves();bones={b.name:[list(x) for x in b.matrix_local] for b in r.data.bones};constraints={p.name:[(c.name,c.type,[(k,getattr(c,k,None)) for k in ['min_x','max_x','min_y','max_y','min_z','max_z']]) for c in p.constraints] for p in r.pose.bones}
meshes={o.name:np.array([v.co for v in o.data.vertices]) for o in bpy.data.objects if o.type=='MESH'};audit={}
for side in ['L','R']:
 name=side+'_ankle_pitch_visual';o=bpy.data.objects[name];pivot=r.data.bones[side+'_ankle_pitch'].head_local.copy();knee=r.data.bones[side+'_knee_pitch'].head_local.copy();v=meshes[name];h=Vector(v[housing_ids].mean(0))-pivot
 angle=-math.atan2(h.x,h.z);rotation=Matrix.Rotation(angle,4,'Y')
 # Axial translation centers the housing on the shank in 3D. A translation
 # along the output shaft leaves the entire physical joint axis line unchanged.
 axial=Vector((0,-h.y,0));T=Matrix.Translation(pivot+axial)@rotation@Matrix.Translation(-pivot)
 for point in o.data.vertices:point.co=T@point.co
 o.data.update();new=np.array([p.co for p in o.data.vertices]);center=new[housing_ids].mean(0)
 direction=np.array((knee-pivot).normalized());offset=center-np.array(pivot);perp=offset-direction*np.dot(offset,direction)
 assert np.linalg.norm(perp)<1e-8 and 0<np.dot(offset,direction)<.055
 old_dist=np.linalg.norm(v[:,None]-v[None,:],axis=2);new_dist=np.linalg.norm(new[:,None]-new[None,:],axis=2);assert np.max(abs(new_dist-old_dist))<1e-7
 sc=new[shaft_ids].mean(0);_,_,vh=np.linalg.svd(new[shaft_ids]-sc);axis=vh[-1];check_axis=axis if axis[1]>0 else -axis
 assert abs(check_axis[1]-1)<1e-6
 axis_error=np.linalg.norm(np.cross(np.array(pivot)-sc,check_axis));assert axis_error<1e-7
 o['motor_mount_note']='Housing center on knee-to-ankle line, toward knee; rigid rotation and tiny output-axis translation; shaft line/pivot unchanged'
 o['mount_transform_world_json']=json.dumps([list(x) for x in T]);o['mount_revision']='ankle_housing_inline_20261026'
 audit[side]={'rotation_about_world_y_deg':math.degrees(angle),'translation_along_shaft_m':list(axial),'housing_center_world_zero_m':center.tolist(),'housing_center_distance_along_shank_m':float(np.dot(offset,direction)),'housing_center_line_error_m':float(np.linalg.norm(perp)),'shaft_axis_line_error_m':float(axis_error),'shaft_axis_world':check_axis.tolist(),'shaft_geometry_center_world_zero_m':sc.tolist(),'joint_pivot_world_m':list(pivot),'rigid_shape_max_pair_distance_error_m':float(np.max(abs(new_dist-old_dist))),'transform_world':[list(x) for x in T]}
for name,v in meshes.items():
 if name not in ['L_ankle_pitch_visual','R_ankle_pitch_visual']:assert np.array_equal(v,np.array([p.co for p in bpy.data.objects[name].data.vertices])),name
assert curves()==keys
assert bones=={b.name:[list(x) for x in b.matrix_local] for b in r.data.bones}
assert constraints=={p.name:[(c.name,c.type,[(k,getattr(c,k,None)) for k in ['min_x','max_x','min_y','max_y','min_z','max_z']]) for c in p.constraints] for p in r.pose.bones}
s['ankle_inline_revision']='ankle_housing_inline_20261026';info=json.loads(s['powered_ankle_json']);info['housing_mount_revision']=s['ankle_inline_revision'];info['housing_mount']=audit;s['powered_ankle_json']=json.dumps(info)
text=bpy.data.texts.get('POWERED_ANKLES_20261026');text.write('\n修正：踝电机壳中心沿膝—踝连线向膝方向，保持输出轴线和关节轴心；壳体作刚体旋转及沿输出轴约0.395mm平移，不改变外形、关节角、限位或55mm小腿。\n')
s.frame_set(5);bpy.context.view_layer.update();bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'sable_rig.blend'),check_existing=False)
(OUT/'edit_receipt.json').write_text(json.dumps({'passed':True,'input_sha256':manifest['blender/sable_rig.blend'],'candidate_sha256':sha(OUT/'sable_rig.blend'),'mounting':audit,'all_unrelated_meshes_bones_curves_constraints_unchanged':True},indent=2))
from export_description import export_robot
export_robot(OUT/'exports')
print('INLINE_MOTOR_CANDIDATE',json.dumps(audit))
