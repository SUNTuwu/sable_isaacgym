"""Authored 100/50 mm torso layout, 40 mm forearms and fixed paw mounts."""
import json
import math
import bpy
import numpy as np
from mathutils import Vector

REVISION = 'waist_100_50_forearm40_wrist20_ankle70_v1'
ANGLE_REFERENCE = 'downward extension of forearm/shank = 0 deg; forward (+X) positive'
FRONT_PARTS = {'shoulder_roll', 'shoulder_pitch', 'elbow_pitch', 'wrist_fixed', 'hand'}
REAR_PARTS = {'hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_fixed', 'foot'}


def rebuild_guides(rig):
    edges = [('GUIDE_waist', 'spine_pitch', 'spine_roll', .002)]
    for side in ['L', 'R']:
        hr = 'hip_roll' if side == 'L' else 'R_hip_roll_mimic'
        edges += [('GUIDE_'+side+'_chest', 'spine_pitch', side+'_shoulder_roll', .002),
                  ('GUIDE_'+side+'_pelvis', 'spine_roll', hr, .002)]
        for label, a, b in [('shoulder_roll', side+'_shoulder_roll', side+'_shoulder_pitch'),
                            ('shoulder_pitch', side+'_shoulder_pitch', side+'_elbow_pitch'),
                            ('elbow_pitch', side+'_elbow_pitch', side+'_wrist_fixed'),
                            ('hip_roll', hr, side+'_hip_pitch'),
                            ('hip_pitch', side+'_hip_pitch', side+'_knee_pitch'),
                            ('knee_pitch', side+'_knee_pitch', side+'_ankle_fixed')]:
            edges.append(('GUIDE_'+side+'_'+label, a, b, .0014))
    for name, a, b, radius in edges:
        obj = bpy.data.objects[name]
        start, end = rig.data.bones[a].head_local, rig.data.bones[b].head_local
        rotation = (end-start).to_track_quat('Z', 'Y')
        vertices = [p+rotation@Vector((radius*math.cos(2*math.pi*i/12),
                                      radius*math.sin(2*math.pi*i/12), 0))
                    for p in [start, end] for i in range(12)]
        faces = [tuple(reversed(range(12))), tuple(range(12, 24))]
        faces += [(i, (i+1)%12, (i+1)%12+12, i+12) for i in range(12)]
        group_name = obj.vertex_groups[0].name
        old = obj.data
        mesh = bpy.data.meshes.new(name+'_layout'); mesh.from_pydata(vertices, [], faces); mesh.update()
        for mat in old.materials: mesh.materials.append(mat)
        obj.data = mesh
        group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
        group.add(list(range(24)), 1., 'REPLACE')
        if old.users == 0: bpy.data.meshes.remove(old)
    return [{'object': n, 'from': a, 'to': b} for n, a, b, _ in edges]


def apply_layout(rig, scene, shoulder_distance=.100):
    assert not scene.get('layout_revision'), 'Layout already applied'
    rows = json.loads(rig['joint_table_json']); meshes = json.loads(rig['mesh_table_json'])
    before_bones = {b.name: [list(r) for r in b.matrix_local] for b in rig.data.bones}
    before_vertices = {r['object']: [list(v.co) for v in bpy.data.objects[r['object']].data.vertices] for r in meshes}
    waist_x = rig.data.bones['spine_pitch'].head_local.x
    front_dx = waist_x+shoulder_distance-rig.data.bones['L_shoulder_pitch'].head_local.x
    rear_dx = waist_x-.050-rig.data.bones['L_hip_pitch'].head_local.x
    forearm_dz = {s: rig.data.bones[s+'_elbow_pitch'].head_local.z-.040-
                    rig.data.bones[s+'_wrist_fixed'].head_local.z for s in ['L', 'R']}
    bone_deltas = {}
    for b in rig.data.bones:
        key = b.name[2:] if b.name[:2] in ['L_', 'R_'] else b.name
        if b.name in ['hip_roll', 'R_hip_roll_mimic']: key = 'hip_roll'
        if key in FRONT_PARTS: bone_deltas[b.name] = [front_dx, 0., forearm_dz[b.name[0]] if key == 'wrist_fixed' else 0.]
        elif key in REAR_PARTS: bone_deltas[b.name] = [rear_dx, 0., 0.]
    if bpy.context.object and bpy.context.object.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name, delta in bone_deltas.items():
        b = rig.data.edit_bones[name]; b.head += Vector(delta); b.tail += Vector(delta)
    bpy.ops.object.mode_set(mode='OBJECT')

    transforms = {}; colliders = []
    for row in meshes:
        name, key = row['object'], row['source_key']; obj = bpy.data.objects[name]
        matrix = np.eye(4)
        if key in ['hand', 'foot']:
            c = json.loads(obj['collision_description_json'])
            old_axis = np.array(c['axis_world_zero']); old_center = np.array(c['center_world_zero_m'])
            old_pivot = np.array(before_bones[c['bone']])[:3, 3]
            segment = c['cylinder_length_m']
            ends = [old_center-old_axis*segment/2, old_center+old_axis*segment/2]
            index = int(np.argmin([np.linalg.norm(p-old_pivot) for p in ends]))
            old_proximal = ends[index]; old_distal_axis = old_axis if index == 0 else -old_axis
            degrees = 20. if key == 'hand' else 70.  # User corrected 110 internal angle to 180-110.
            angle = math.radians(degrees); axis = np.array([math.sin(angle), 0., -math.cos(angle)])
            pivot = np.array(rig.data.bones[c['bone']].head_local)
            rotation = np.array(Vector(old_distal_axis).rotation_difference(Vector(axis)).to_matrix())
            matrix[:3, :3] = rotation; matrix[:3, 3] = pivot-rotation@old_proximal
            center = pivot+axis*segment/2
            frame = rig.data.bones[c['bone']].matrix_local.inverted()
            local_center = frame@Vector(center); local_axis = (frame.to_3x3()@Vector(axis)).normalized()
            c.update(center_world_zero_m=center.tolist(), axis_world_zero=axis.tolist(),
                     center_bone_m=list(local_center), axis_bone=list(local_axis),
                     endpoints_bone_m=[list(local_center-local_axis*segment/2), list(local_center+local_axis*segment/2)],
                     proximal_sphere_center_world_zero_m=pivot.tolist(), mount_angle_deg=degrees,
                     mount_angle_reference=ANGLE_REFERENCE, mount_mode='fixed geometry; not a new actuated coordinate',
                     source='Original named terminal dimensions retained; proximal hemisphere center at wrist/ankle pivot; user-specified sagittal mounting angle')
            obj['collision_description_json'] = json.dumps(c)
            obj['fixed_mount_angle_deg'] = degrees
            obj['geometry_change_reason'] = 'User: capsule proximal sphere center at terminal joint, wrist 20 and ankle 70 degrees'
            row['geometry_revision'] = '30 mm capsule, proximal sphere at joint; fixed mounting angle in collision metadata'
            colliders.append(c)
        elif key in FRONT_PARTS:
            matrix[:3, 3] = [front_dx, 0., forearm_dz[name[0]] if key == 'wrist_fixed' else 0.]
        elif key in REAR_PARTS:
            matrix[:3, 3] = [rear_dx, 0., 0.]
        if not np.array_equal(matrix, np.eye(4)):
            vv = np.array(before_vertices[name])@matrix[:3, :3].T+matrix[:3, 3]
            for v, co in zip(obj.data.vertices, vv): v.co = co
            obj.data.update(); transforms[name] = matrix.tolist()
    guides = rebuild_guides(rig)
    for row in rows: row['position'] = list(rig.data.bones[row['name']].head_local)
    rig['joint_table_json'] = json.dumps(rows, ensure_ascii=False)
    rig['mesh_table_json'] = json.dumps(meshes, ensure_ascii=False)
    rig['collision_shapes_json'] = json.dumps(colliders)
    revision = REVISION if shoulder_distance == .100 else 'waist_90_50_forearm40_wrist20_ankle70_v2'
    scene['layout_revision'] = revision
    if shoulder_distance == .090: scene['shoulder_distance_revision'] = 'waist_to_shoulder_90mm_v1'
    scene['forearm_target_length_m'] = .040
    scene['waist_shoulder_horizontal_m'] = shoulder_distance; scene['waist_hip_horizontal_m'] = .050
    scene['wrist_mount_deg'] = 20.; scene['ankle_mount_deg'] = 70.
    scene['terminal_mount_angle_reference'] = ANGLE_REFERENCE
    return {'revision': revision, 'bone_rest_matrices_before': before_bones,
            'mesh_vertices_before': before_vertices, 'bone_translations_m': bone_deltas,
            'mesh_transforms_world': transforms, 'front_translation_x_m': front_dx,
            'rear_translation_x_m': rear_dx, 'forearm_translation_z_m': forearm_dz,
            'guides': guides, 'wrist_mount_deg': 20., 'ankle_mount_deg': 70.,
            'angle_reference': ANGLE_REFERENCE, 'collision_shapes': colliders}


def update_build_report(report, rig, audit):
    report.update(layout_revision=audit['revision'], layout_adjustment=audit,
                  joint_rows=json.loads(rig['joint_table_json']), mesh_rows=json.loads(rig['mesh_table_json']),
                  collision_shapes=json.loads(rig['collision_shapes_json']))
    # Parts describe the left source assembly before the common global shift.
    for key, part in report['parts'].items():
        if key in FRONT_PARTS:
            part['zero_pivot_before_global_shift_m'][0] += audit['front_translation_x_m']
            if key in ['wrist_fixed', 'hand']:
                part['zero_pivot_before_global_shift_m'][2] += audit['forearm_translation_z_m']['L']
        elif key in REAR_PARTS:
            part['zero_pivot_before_global_shift_m'][0] += audit['rear_translation_x_m']
        if key in ['hand', 'foot']:
            part['terminal_geometry_authority'] = 'collision_shapes; original box transform is historical provenance only'
