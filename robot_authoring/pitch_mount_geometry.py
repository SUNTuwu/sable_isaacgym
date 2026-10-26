"""Toe-in the pitch mounting (not shoulder roll) and carry its descendants."""
import bpy,json,math
import numpy as np
from mathutils import Matrix,Vector
from layout_geometry import rebuild_guides
from mount_collision import components,interference

REVISION='shoulder_pitch_toein5_outward2mm_v1'
PARTS={'shoulder_pitch','elbow_pitch','wrist_fixed','hand'}

def apply_pitch_mount(rig,scene):
    assert not scene.get('shoulder_pitch_mount_revision'),'Already applied'
    rows,meshes=json.loads(rig['joint_table_json']),json.loads(rig['mesh_table_json'])
    bones_before={b.name:[list(x) for x in b.matrix_local] for b in rig.data.bones}
    vertices_before={r['object']:[list(v.co) for v in bpy.data.objects[r['object']].data.vertices] for r in meshes}
    bone_transforms={};mesh_transforms={};sides={}
    for side,sign in [('L',1),('R',-1)]:
        pivot=np.array(rig.data.bones[side+'_shoulder_pitch'].head_local,dtype=np.float64)
        angle=math.radians(-sign*5);c,s=math.cos(angle),math.sin(angle)
        rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        roll=components(bpy.data.objects[side+'_shoulder_roll_visual']);probes=[]
        selected=None
        for offset in [0.,.001,.002]:
            m=np.eye(4);m[:3,:3]=rot;m[:3,3]=pivot-rot@pivot+[0,sign*offset,0]
            result=interference(roll,components(bpy.data.objects[side+'_shoulder_pitch_visual'],m))
            probes.append({'outward_offset_m':offset,**result})
            if not result['penetrating']:
                selected=(offset,m);break
        assert selected is not None,'Interference remains at the authorized 2 mm limit'
        offset,m=selected
        sides[side]={'yaw_world_z_deg':-sign*5,'outward_offset_m':offset,'pivot_before_m':pivot.tolist(),
                     'transform_world':m.tolist(),'probes':probes}
        for n in bones_before:
            if n.startswith(side+'_') and n[2:] in PARTS:bone_transforms[n]=m.tolist()
        for r in meshes:
            if r['object'].startswith(side+'_') and r['source_key'] in PARTS:mesh_transforms[r['object']]=m.tolist()
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
    for n,m in bone_transforms.items():
        b=rig.data.edit_bones[n];transform=Matrix(m)
        head,tail=b.head.copy(),b.tail.copy()
        expected=np.array(m)@np.array(bones_before[n])
        b.head=transform@head;b.tail=transform@tail
        # Explicitly retain the transformed local Z. EditBone.transform's
        # inferred roll can drift even for a pure world-Z installation yaw.
        b.align_roll(Vector(expected[:3,2]))
    bpy.ops.object.mode_set(mode='OBJECT')
    for n,m in mesh_transforms.items():
        obj=bpy.data.objects[n];m=np.array(m)
        vv=np.array(vertices_before[n])@m[:3,:3].T+m[:3,3]
        for v,p in zip(obj.data.vertices,vv):v.co=p
        obj.data.update()
        if obj.get('collision_description_json'):
            collider=json.loads(obj['collision_description_json'])
            for key in ['center_world_zero_m','proximal_sphere_center_world_zero_m']:
                collider[key]=(m[:3,:3]@np.array(collider[key])+m[:3,3]).tolist()
            collider['axis_world_zero']=(m[:3,:3]@np.array(collider['axis_world_zero'])).tolist()
            collider['mount_angle_reference']='Downward forearm extension = 0 deg; positive toward this limb forward direction, including its 5 deg inward mounting yaw'
            obj['collision_description_json']=json.dumps(collider)
    guides=rebuild_guides(rig)
    for row in rows:
        b=rig.data.bones[row['name']]
        row['position']=list(b.head_local)
        row['axis']=list((b.matrix_local.to_3x3()@Vector((0,1,0))).normalized())
    colliders=[json.loads(bpy.data.objects[r['object']]['collision_description_json']) for r in meshes
               if bpy.data.objects[r['object']].get('collision_description_json')]
    rig['joint_table_json']=json.dumps(rows,ensure_ascii=False)
    rig['collision_shapes_json']=json.dumps(colliders)
    scene['shoulder_pitch_mount_revision']=REVISION
    scene['shoulder_pitch_mount_json']=json.dumps(sides)
    scene['terminal_mount_angle_reference']='Downward limb extension = 0 deg; positive toward local limb forward; front limb includes its inward mounting yaw'
    return {'revision':REVISION,'sides':sides,'bone_transforms_world':bone_transforms,
            'mesh_transforms_world':mesh_transforms,'bone_rest_matrices_before':bones_before,
            'mesh_vertices_before':vertices_before,'guides':guides,'collision_shapes':colliders,
            'scope':'Rest installation geometry; unchanged numeric pose keys and limits; zero-pose mesh clearance only'}
