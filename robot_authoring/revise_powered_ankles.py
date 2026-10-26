"""Apply the user-authorized two powered ankles to a staged copy, never regenerate the rig."""
import bpy, json, math, hashlib, sys
from pathlib import Path
from mathutils import Matrix, Vector
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'robot_authoring')); sys.dont_write_bytecode=True
from capsule_geometry import capsule_vertices
OUT=ROOT/'results/model_sync/20261026_powered_ankles'
rig=bpy.data.objects['SableRobot']; scene=bpy.context.scene
assert not scene.get('powered_ankle_revision')
assert Path(bpy.data.filepath).resolve()==ROOT/'blender/sable_rig.blend'
manifest=json.loads((OUT/'before_manifest.json').read_text())
assert hashlib.sha256((ROOT/'blender/sable_rig.blend').read_bytes()).hexdigest()==manifest['blender/sable_rig.blend']

def curves():
 return {a.name:{f'{f.data_path}|{f.array_index}':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in f.keyframe_points] for l in a.layers for s in l.strips for b in s.channelbags for f in b.fcurves} for a in bpy.data.actions}
def snapshot():
 return {'bones':{b.name:{'matrix':[list(x) for x in b.matrix_local],'parent':b.parent.name if b.parent else None} for b in rig.data.bones},
 'objects':{o.name:{'matrix':[list(x) for x in o.matrix_world], 'vertices':hashlib.sha256(np.asarray([v.co for v in o.data.vertices],dtype=np.float32).tobytes()).hexdigest() if o.type=='MESH' else None,'groups':[g.name for g in o.vertex_groups] if o.type=='MESH' else []} for o in bpy.data.objects},
 'constraints':{p.name:[{'name':c.name,'type':c.type,**{k:getattr(c,k,None) for k in ['min_x','max_x','min_y','max_y','min_z','max_z','owner_space','use_limit_y']}} for c in p.constraints] for p in rig.pose.bones},'actions':curves()}
before=snapshot(); (OUT/'scene_before.json').write_text(json.dumps(before,indent=2))
rows=json.loads(rig['joint_table_json']); meshes=json.loads(rig['mesh_table_json'])
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active=rig
# World -Y makes a positive ankle angle rotate the downward foot toward +X.
bpy.ops.object.mode_set(mode='EDIT')
for side in ['L','R']:
 b=rig.data.edit_bones[side+'_ankle_fixed']; length=b.length
 pivot=rig.data.edit_bones[side+'_knee_pitch'].head+Vector((0,0,-.055))
 b.name=side+'_ankle_pitch'; b.matrix=Matrix.Translation(pivot)@Matrix.Rotation(math.pi,4,'Z'); b.length=length
bpy.ops.object.mode_set(mode='OBJECT')
for side in ['L','R']:
 name=side+'_ankle_pitch'; pb=rig.pose.bones[name]; pb.lock_rotation=(True,False,True)
 pb.rotation_mode='XYZ'; pb.rotation_euler=(0,math.radians(70),0)
 c=next(c for c in pb.constraints if c.type=='LIMIT_ROTATION'); c.min_y=0;c.max_y=math.radians(160)
 pb['joint_type']='revolute';pb['actuated']=True;pb['source_object']='膝关节pitch-dc1628'
 pb['angle_reference']='shank downward extension = 0; positive forward; standard stand 70 deg'
 pb['motor_template']='same named dc1628 geometry as knee/elbow; calibration pending'
 for bc in list(pb.bone.collections):bc.unassign(pb.bone)
 rig.data.collections['Hind legs'].assign(pb.bone)
 # Change only the former fixed ankle angle curves, preserving their key times/interpolation.
 for a in bpy.data.actions:
  for l in a.layers:
   for s in l.strips:
    for bag in s.channelbags:
     for f in bag.fcurves:
      if f.data_path==f'pose.bones["{name}"].rotation_euler' and f.array_index==1:
       for k in f.keyframe_points:
        for v in [k.co,k.handle_left,k.handle_right]:v.y+=math.radians(70)
       f.update()
 # Replace the provisional ankle block with a rigid copy of the same-side named knee motor.
 old=bpy.data.objects[side+'_ankle_fixed_visual']; src=bpy.data.objects[side+'_knee_pitch_visual']
 pivot=rig.data.bones[name].head_local.copy(); knee=rig.data.bones[side+'_knee_pitch'].head_local.copy()
 old.name=side+'_ankle_pitch_visual'; old.data=src.data.copy(); old.data.name=old.name+'_mesh'
 for v in old.data.vertices:v.co+=pivot-knee
 old.vertex_groups.clear(); vg=old.vertex_groups.new(name=side+'_knee_pitch');vg.add(list(range(len(old.data.vertices))),1,'REPLACE')
 old['rigid_link_bone']=side+'_knee_pitch';old['source_object']=src['source_object'];old['part_geometry_changed']=False
 old['motor_template_object']=src.name;old['motor_mount_note']='same knee housing orientation, translated to ankle; fixed to shank'
 for row in meshes:
  if row['object']==side+'_ankle_fixed_visual':
   row.update(object=old.name,source_object=src['source_object'],source_key='ankle_pitch',bone=side+'_knee_pitch',source_vertices=len(old.data.vertices),faces=len(old.data.polygons),motor_template=src.name)
 # Capsule is vertical at q=0; proximal sphere center remains exactly at the ankle shaft.
 foot=bpy.data.objects[side+'_foot_visual']; c=json.loads(foot['collision_description_json']);radius=c['radius_m'];segment=.040-2*radius
 axis=Vector((0,0,-1));center=pivot+axis*segment/2
 verts,faces=capsule_vertices(center,axis,radius,segment)
 oldmesh=foot.data; mesh=bpy.data.meshes.new(foot.name+'_40mm');mesh.from_pydata(verts,[],faces);mesh.update()
 for m in oldmesh.materials:mesh.materials.append(m)
 for p in mesh.polygons:p.use_smooth=True
 foot.data=mesh;foot.vertex_groups.clear();foot.vertex_groups.new(name=name).add(list(range(len(verts))),1,'REPLACE')
 inv=rig.data.bones[name].matrix_local.inverted();lc=inv@center;la=(inv.to_3x3()@axis).normalized()
 c.update(bone=name,total_length_m=.040,cylinder_length_m=segment,center_world_zero_m=list(center),axis_world_zero=list(axis),center_bone_m=list(lc),axis_bone=list(la),endpoints_bone_m=[list(lc-la*segment/2),list(lc+la*segment/2)],proximal_sphere_center_world_zero_m=list(pivot),mount_angle_deg=0,standard_stand_angle_deg=70,mount_mode='actuated ankle pitch; angle is joint coordinate',source='40 mm total length requested by user; existing radius preserved; proximal sphere at powered ankle shaft')
 foot['collision_description_json']=json.dumps(c);foot['rigid_link_bone']=name
 foot['collision_total_length_m']=.040;foot['collision_cylinder_length_m']=segment;foot['fixed_mount_angle_deg']=0
 foot['geometry_change_reason']='User requested powered ankle, 40 mm capsule with zero angle along downward shank extension'
 for row in meshes:
  if row['object']==foot.name:row.update(bone=name,geometry_revision='40 mm capsule; powered ankle; zero along downward shank',authored_vertex_count=len(verts),authored_face_count=len(faces))
 # Only the shank display rod changes length.
 guide=bpy.data.objects['GUIDE_'+side+'_knee_pitch'];q=(pivot-knee).to_track_quat('Z','Y')
 vs=[p+q@Vector((.0014*math.cos(2*math.pi*i/12),.0014*math.sin(2*math.pi*i/12),0)) for p in [knee,pivot] for i in range(12)]
 for v,p in zip(guide.data.vertices,vs):v.co=p
 guide.data.update()
 for row in rows:
  if row['name']==side+'_ankle_fixed':row.update(name=name,kind='revolute',actuated=True,source_key='ankle_pitch',position=list(pivot),axis=[0,-1,0],limits_deg=[0,160],motor_template_source=src['source_object'])
rig['joint_table_json']=json.dumps(rows,ensure_ascii=False);rig['mesh_table_json']=json.dumps(meshes,ensure_ascii=False)
rig['collision_shapes_json']=json.dumps([json.loads(bpy.data.objects[r['object']]['collision_description_json']) for r in meshes if bpy.data.objects[r['object']].get('collision_description_json')])
allow=json.loads(scene['source_allowlist_json']);allow['ankle_pitch']=allow['knee_pitch'];scene['source_allowlist_json']=json.dumps(allow,ensure_ascii=False)
scene['powered_ankle_revision']='powered_ankle_shank55_foot40_20261026'
scene['powered_ankle_json']=json.dumps({'revision':scene['powered_ankle_revision'],'joints':['L_ankle_pitch','R_ankle_pitch'],'angle_reference':'relative to downward shank extension, forward positive','axis_world_zero':[0,-1,0],'standard_stand_deg':70,'limits_deg':[0,160],'shank_length_m':.055,'foot_total_length_m':.040,'motor_source':'膝关节pitch-dc1628','housing_binding':'shank / knee_pitch','physical_calibration':'pending; geometry reuse only'})
scene['shank_target_length_m']=.055;scene['ankle_mount_deg']=0
scene['independent_actuator_count']=15;rig['independent_motor_count']=15
notes=bpy.data.texts.get('START_HERE_使用说明')
if notes:notes.clear();notes.write((ROOT/'robot_authoring/README.md').read_text())
# Portable embedded exporter locates the project from the opened blend path.
export_text=bpy.data.texts.get('EXPORT_ROBOT_DESCRIPTION.py')
if export_text:
 export_text.clear();export_text.write("import bpy, sys\nfrom pathlib import Path\nroot=next(p for p in Path(bpy.data.filepath).parents if (p/'robot_authoring/export_description.py').exists())\nsys.path.insert(0,str(root/'robot_authoring'))\nfrom export_description import export_robot\nexport_robot()\n")
# New note prevents old embedded fixed-ankle instructions from appearing current.
text=bpy.data.texts.get('POWERED_ANKLES_20261026') or bpy.data.texts.new('POWERED_ANKLES_20261026')
text.write('左右踝新增独立 pitch 电机，沿用膝电机外形；15 个独立执行器。小腿55mm，后脚胶囊总长40mm、原半径不变。踝角以小腿向下延长线为0，向前为正；站姿70度，限位0–160度。腕仍固定。旧训练资产/策略不兼容，未做动力学或硬件验证。\n')
scene.frame_set(5);bpy.context.view_layer.update()
after=snapshot()
for n,b in before['bones'].items():
 if 'ankle' not in n:assert after['bones'][n]==b,('unrelated bone changed',n)
for n,v in before['objects'].items():
 if any(n==s+x for s in ['L','R'] for x in ['_ankle_fixed_visual','_foot_visual']) or n in ['GUIDE_L_knee_pitch','GUIDE_R_knee_pitch']:continue
 assert after['objects'][n]==v,('unrelated object changed',n)
for n,c in before['constraints'].items():
 if 'ankle' not in n:assert after['constraints'][n]==c,n
for a,fs in before['actions'].items():
 for k,v in fs.items():
  if 'ankle' not in k:assert after['actions'][a][k]==v,k
assert sum(r['actuated'] for r in rows)==15
bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='POSE')
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'sable_rig.blend'),check_existing=False)
(OUT/'scene_after.json').write_text(json.dumps(after,indent=2))
(OUT/'edit_receipt.json').write_text(json.dumps({'revision':scene['powered_ankle_revision'],'blender_version':bpy.app.version_string,'unrelated_bones_meshes_constraints_keyframes_unchanged':True,'original_saved_with_newer_blender_warning':True,'candidate':str(OUT/'sable_rig.blend')},indent=2))
from export_description import export_robot
export_robot(OUT/'exports')
print('POWERED_ANKLES_STAGED')
