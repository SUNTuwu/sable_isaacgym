"""Restore vertical pitch chains and synchronize the user's roll mounts."""
import bpy
import json
import numpy as np
from mathutils import Vector, Matrix
from layout_geometry import rebuild_guides

REVISION='vertical_44_40_50_manual_roll_sync_v1'


def apply_straight_limbs(rig,scene,shoulder_shift=(.008,0,0),waist_shift=(-.005,0,0),existing_roll_bone_deltas=None):
    assert not scene.get('straight_limb_revision'),'Revision already applied'
    rows=json.loads(rig['joint_table_json']);meshes=json.loads(rig['mesh_table_json'])
    bones={b.name:[list(r) for r in b.matrix_local] for b in rig.data.bones}
    vertices={r['object']:[list(v.co) for v in bpy.data.objects[r['object']].data.vertices] for r in meshes}
    matrices={r['object']:[list(v) for v in bpy.data.objects[r['object']].matrix_world] for r in meshes}
    existing_roll_bone_deltas=existing_roll_bone_deltas or {}
    bd={'spine_roll':(np.array(waist_shift)-np.array(existing_roll_bone_deltas.get('spine_roll',[0,0,0]))).tolist()}
    md={'spine_roll_visual':list(waist_shift)}
    for side in ['L','R']:
        shift=np.array(shoulder_shift)*([1,1,1] if side=='L' else [1,-1,1])
        bd[side+'_shoulder_roll']=(shift-np.array(existing_roll_bone_deltas.get(side+'_shoulder_roll',[0,0,0]))).tolist()
        md[side+'_shoulder_roll_visual']=shift.tolist()
        for proximal,middle,distal,terminal,lower in [('shoulder_pitch','elbow_pitch','wrist_fixed','hand',.040),
                                                    ('hip_pitch','knee_pitch','ankle_fixed','foot',.050)]:
            top=np.array(rig.data.bones[side+'_'+proximal].head_local)
            for name,length in [(middle,.044),(distal,.044+lower)]:
                target=top+np.array([0,0,-length]);joint=side+'_'+name
                delta=target-np.array(rig.data.bones[joint].head_local)
                bd[joint]=delta.tolist();md[joint+'_visual']=delta.tolist()
                if name==distal:md[side+'_'+terminal+'_visual']=delta.tolist()
    # Only the observed pure translations may be baked. Do not silently erase
    # a new manual rotation/scale or an unrecognized edit to another object.
    for name,matrix in matrices.items():
        matrix=np.array(matrix);expected=np.eye(4)
        if name in ['L_shoulder_roll_visual','spine_roll_visual']:
            expected[:3,3]=md[name]
        assert np.max(abs(matrix-np.eye(4)))<1e-8 or np.max(abs(matrix-expected))<1e-8,name
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
    for name,delta in bd.items():
        bone=rig.data.edit_bones[name]
        bone.head+=Vector(delta);bone.tail+=Vector(delta)
    bpy.ops.object.mode_set(mode='OBJECT')
    for name,delta in md.items():
        obj=bpy.data.objects[name]
        for v,p in zip(obj.data.vertices,np.array(vertices[name])+np.array(delta)):v.co=p
        obj.matrix_world=Matrix.Identity(4);obj.data.update()
        if obj.get('collision_description_json'):
            collider=json.loads(obj['collision_description_json'])
            for key in ['center_world_zero_m','proximal_sphere_center_world_zero_m']:
                collider[key]=(np.array(collider[key])+np.array(delta)).tolist()
            obj['collision_description_json']=json.dumps(collider)
    bpy.context.view_layer.update()
    guides=rebuild_guides(rig)
    for r in rows:
        b=rig.data.bones[r['name']]
        r['position']=list(b.head_local)
        r['axis']=list((b.matrix_local.to_3x3()@Vector((0,1,0))).normalized())
    colliders=[json.loads(bpy.data.objects[r['object']]['collision_description_json']) for r in meshes
               if bpy.data.objects[r['object']].get('collision_description_json')]
    rig['joint_table_json']=json.dumps(rows,ensure_ascii=False)
    rig['collision_shapes_json']=json.dumps(colliders)
    scene['straight_limb_revision']=REVISION
    cumulative={n:(np.array(d)+np.array(existing_roll_bone_deltas.get(n,[0,0,0]))).tolist() for n,d in bd.items()}
    scene['straight_limb_deltas_json']=json.dumps(cumulative)
    scene['manual_roll_sync_json']=json.dumps({'shoulder_roll_both_world_m':list(shoulder_shift),
        'spine_roll_world_m':list(waist_shift),'source':'User-saved left shoulder / waist motor object translations, synchronized to pivots; right shoulder mirrored'})
    return {'revision':REVISION,'bone_translations_m':bd,'mesh_translations_m':md,
        'bone_rest_matrices_before':bones,'mesh_vertices_before':vertices,'mesh_matrices_before':matrices,
        'manual_bone_translations_m':existing_roll_bone_deltas,
        'guides':guides,'collision_shapes':colliders,'length_targets_m':{'upper_arm':.044,'thigh':.044,'forearm':.040,'shank':.050},
        'policy':'Vertical child pitch chains from unchanged shoulder/hip pitch centers. Bake approved manual roll translations into mesh and joint centers. Preserve numeric keyframes, limits, axes and mounts.'}
