"""Translate selected joint centers and their motors without moving children."""
import bpy,json
import numpy as np
from mathutils import Vector
from layout_geometry import rebuild_guides

REVISION='independent_mount_offsets_torso100_60_chest_root_roll_v1'

def apply_motor_offsets(rig,scene):
    assert not scene.get('motor_offset_revision'),'Already applied'
    rows,meshes=json.loads(rig['joint_table_json']),json.loads(rig['mesh_table_json'])
    bones_before={b.name:[list(r) for r in b.matrix_local] for b in rig.data.bones}
    vertices_before={r['object']:[list(v.co) for v in bpy.data.objects[r['object']].data.vertices] for r in meshes}
    bone_deltas={'spine_roll':[.001,0,0]};mesh_deltas={'spine_roll_visual':[.001,0,0]}
    waist_x=rig.data.bones['spine_pitch'].head_local.x
    for side,sign in [('L',1),('R',-1)]:
        shoulder_dx=waist_x+.100-rig.data.bones[side+'_shoulder_pitch'].head_local.x
        hip_dx=waist_x-.060-rig.data.bones[side+'_hip_pitch'].head_local.x
        for name,delta in [('shoulder_roll',[-.004,0,0]),('shoulder_pitch',[shoulder_dx,-sign*.004,0]),('hip_pitch',[hip_dx,-sign*.005,0])]:
            bone_deltas[side+'_'+name]=delta;mesh_deltas[side+'_'+name+'_visual']=delta
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
    for name,delta in bone_deltas.items():
        bone=rig.data.edit_bones[name];bone.head+=Vector(delta);bone.tail+=Vector(delta)
    bpy.ops.object.mode_set(mode='OBJECT')
    for name,delta in mesh_deltas.items():
        obj=bpy.data.objects[name]
        assert np.max(abs(np.array(obj.matrix_world)-np.eye(4)))<1e-8,'Unexpected manual object transform: '+name
        for v in obj.data.vertices:v.co+=Vector(delta)
        obj.data.update()
    guides=rebuild_guides(rig)
    for r in rows:r['position']=list(rig.data.bones[r['name']].head_local)
    rig['joint_table_json']=json.dumps(rows,ensure_ascii=False)
    scene['motor_offset_revision']=REVISION
    scene['motor_offset_json']=json.dumps(bone_deltas)
    scene['waist_shoulder_horizontal_m']=.100;scene['waist_hip_horizontal_m']=.060
    return {'revision':REVISION,'bone_translations_m':bone_deltas,'mesh_translations_m':mesh_deltas,
            'bone_rest_matrices_before':bones_before,'mesh_vertices_before':vertices_before,'guides':guides,
            'policy':'Move only listed motor meshes and joint centers in rest geometry; all unlisted bone rest matrices and mesh vertices remain fixed.'}

def configure_chest_root(rig,scene,draft_path):
    """Use the existing topmost ROOT; no new motor/physical joint is added."""
    root=rig.pose.bones['ROOT'];assert root.parent is None
    assert (root.bone.head_local-rig.data.bones['spine_pitch'].head_local).length<1e-8
    root.lock_rotation=(False,True,True)
    root['control_label']='胸口 / 全身 Roll（最根部）'
    root['rotation_control']='Select ROOT, R X X: chest and entire robot roll about the waist pitch center'
    root.custom_shape_rotation_euler=(0,0,np.pi/2)
    root.color.palette='THEME01'
    with bpy.data.libraries.load(str(draft_path),link=False) as (src,dst):
        assert 'Cylinder.001' in src.objects,'Expected user-authored body reference in new draft'
        dst.objects=['Cylinder.001']
    body=dst.objects[0];body.name='BodyReference_visual'
    collection=bpy.data.collections.get('BodyReference_nonexport') or bpy.data.collections.new('BodyReference_nonexport')
    if collection.name not in scene.collection.children:scene.collection.children.link(collection)
    collection.objects.link(body)
    body['role']='User body-shape reference, rigidly follows ROOT; excluded from physical robot export'
    body['source_file']='blender/draft.blend';body['source_object_name']='Cylinder.001'
    group=body.vertex_groups.get('ROOT') or body.vertex_groups.new(name='ROOT')
    group.add(list(range(len(body.data.vertices))),1.,'REPLACE')
    modifier=body.modifiers.new('Chest root roll','ARMATURE');modifier.object=rig
    scene['chest_root_roll_revision']='existing_ROOT_global_roll_about_waist_pitch_v1'
    bpy.context.view_layer.update()
    return {'bone':'ROOT','parent':None,'rotation_axis_local':'X','rotation_axis_world_zero':[1,0,0],
            'pivot_world_zero_m':list(rig.matrix_world@root.bone.head_local),'body_reference_object':body.name,
            'physical_actuators_added':0,'export_reference_body':False,
            'body_vertices':[list(v.co) for v in body.data.vertices],
            'body_matrix_world':[list(row) for row in body.matrix_world]}
