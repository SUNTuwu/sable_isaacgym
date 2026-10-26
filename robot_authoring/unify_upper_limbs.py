"""Set shoulder-elbow and hip-knee center distances to 44 mm, once.

Moves distal assemblies rigidly, preserving the lower limb lengths and all
authored action channels, constraints, transmissions and motor geometry.
"""
import bpy
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from mathutils import Vector

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'blender/sable_rig.blend'
SOURCE = ROOT / 'blender/draft.blend'
BACKUP = ROOT / '待删除/2026-10-25_before_upper_limbs_44mm/blender/sable_rig.blend'
LENGTH = .044
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve() == TARGET.resolve()
before_hash, source_hash = digest(TARGET), digest(SOURCE)
assert before_hash == digest(BACKUP), 'Rig changed after backup'
rig, scene = bpy.data.objects['SableRobot'], bpy.context.scene
assert not scene.get('upper_limb_length_revision'), 'Length revision already applied'
assert np.max(abs(np.array(rig.matrix_world) - np.eye(4))) < 1e-8
rows = json.loads(rig['joint_table_json'])
meshes = json.loads(rig['mesh_table_json'])
old_frame, old_mode = scene.frame_current, rig.mode


def actions_snapshot():
    return [{'name': a.name, 'curves': [
        {'path': fc.data_path, 'index': fc.array_index,
         'keys': [[list(k.co), list(k.handle_left), list(k.handle_right), k.interpolation]
                  for k in fc.keyframe_points]}
        for layer in a.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]}
        for a in bpy.data.actions]


def limits_snapshot():
    return {p.name: [[c.name, c.type] + [getattr(c, k, None) for k in
            ['min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']] for c in p.constraints]
            for p in rig.pose.bones}


def measurements():
    out = {}
    for side in ['L', 'R']:
        for kind, a, b, c in [('upper_arm', 'shoulder_pitch', 'elbow_pitch', 'wrist_fixed'),
                               ('thigh', 'hip_pitch', 'knee_pitch', 'ankle_fixed')]:
            p = [rig.data.bones[side + '_' + n].head_local for n in [a, b, c]]
            out[side + '_' + kind] = {'length_m': (p[0] - p[1]).length,
                'lower_segment_length_m': (p[1] - p[2]).length,
                'proximal_height_m': p[0].z, 'distal_height_m': p[1].z}
    return out


before_measure = measurements()
heights = [v['proximal_height_m'] for v in before_measure.values()]
assert max(heights) - min(heights) < 1e-7, 'Shoulder and hip centers must already be level'
bone_before = {b.name: [list(row) for row in b.matrix_local] for b in rig.data.bones}
mesh_before = {r['object']: [list(v.co) for v in bpy.data.objects[r['object']].data.vertices] for r in meshes}
actions_before, limits_before = actions_snapshot(), limits_snapshot()
bone_deltas, mesh_deltas = {}, {}
for side in ['L', 'R']:
    for proximal, distal, terminal, paw in [('shoulder_pitch', 'elbow_pitch', 'wrist_fixed', 'hand'),
                                           ('hip_pitch', 'knee_pitch', 'ankle_fixed', 'foot')]:
        z = rig.data.bones[side + '_' + proximal].head_local.z - LENGTH
        delta = Vector((0, 0, z - rig.data.bones[side + '_' + distal].head_local.z))
        assert abs(delta.z) < .002, 'Unexpected assembly change'
        for name in [distal, terminal]:
            bone_deltas[side + '_' + name] = delta.copy()
        for name in [distal, terminal, paw]:
            mesh_deltas[side + '_' + name + '_visual'] = delta.copy()

scene.frame_set(1)
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='EDIT')
for name, delta in bone_deltas.items():
    b = rig.data.edit_bones[name]
    b.head += delta
    b.tail += delta
bpy.ops.object.mode_set(mode='OBJECT')
for name, delta in mesh_deltas.items():
    obj = bpy.data.objects[name]
    for v in obj.data.vertices:
        v.co += delta
    obj.data.update()

# Rebuild only display rods whose endpoints changed. The motor meshes stay rigid.
for side in ['L', 'R']:
    for a, b in [('shoulder_pitch', 'elbow_pitch'), ('elbow_pitch', 'wrist_fixed'),
                 ('hip_pitch', 'knee_pitch'), ('knee_pitch', 'ankle_fixed')]:
        obj = bpy.data.objects['GUIDE_' + side + '_' + a]
        start = rig.data.bones[side + '_' + a].head_local.copy()
        end = rig.data.bones[side + '_' + b].head_local.copy()
        rotation = (end - start).to_track_quat('Z', 'Y')
        points = [p + rotation @ Vector((.0014 * math.cos(2 * math.pi * i / 12),
                                        .0014 * math.sin(2 * math.pi * i / 12), 0))
                  for p in [start, end] for i in range(12)]
        # Blender primitive cylinder vertex order can interleave endpoint rings.
        # Replace topology as well as coordinates, retaining material and binding.
        faces = [tuple(reversed(range(12))), tuple(range(12, 24))]
        faces += [(i, (i+1) % 12, (i+1) % 12 + 12, i+12) for i in range(12)]
        group_name = obj.vertex_groups[0].name
        old = obj.data
        mesh = bpy.data.meshes.new(obj.name + '_44mm')
        mesh.from_pydata(points, [], faces)
        for material in old.materials:
            mesh.materials.append(material)
        obj.data = mesh
        group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
        group.add(list(range(24)), 1., 'REPLACE')
        if old.users == 0:
            bpy.data.meshes.remove(old)

colliders = []
for row in meshes:
    obj = bpy.data.objects[row['object']]
    if obj.get('collision_description_json'):
        c = json.loads(obj['collision_description_json'])
        # Capsule and its bone received the same translation: local shape unchanged.
        c['center_world_zero_m'] = list(Vector(c['center_world_zero_m']) + mesh_deltas[obj.name])
        obj['collision_description_json'] = json.dumps(c)
        colliders.append(c)
rig['collision_shapes_json'] = json.dumps(colliders)
for row in rows:
    row['position'] = list(rig.data.bones[row['name']].head_local)
rig['joint_table_json'] = json.dumps(rows, ensure_ascii=False)
scene['upper_limb_length_revision'] = 'upper_arm_and_thigh_44mm_v1'
scene['upper_limb_target_length_m'] = LENGTH
notes = bpy.data.texts['START_HERE_使用说明']
notes.clear()
notes.write((ROOT / 'robot_authoring/README.md').read_text(encoding='utf8'))
after_measure = measurements()
for key, value in after_measure.items():
    assert abs(value['length_m'] - LENGTH) < 1e-8
    assert abs(value['lower_segment_length_m'] - before_measure[key]['lower_segment_length_m']) < 1e-8
assert max(v['distal_height_m'] for v in after_measure.values()) - min(v['distal_height_m'] for v in after_measure.values()) < 1e-8
for name, values in bone_before.items():
    expected = np.array(values)
    expected[:3, 3] += np.array(bone_deltas.get(name, (0, 0, 0)))
    error = abs(np.array(rig.data.bones[name].matrix_local) - expected)
    assert error[:3, 3].max() < 1e-8 and error[:3, :3].max() < 5e-7, name
for name, values in mesh_before.items():
    actual = np.array([v.co for v in bpy.data.objects[name].data.vertices])
    expected = np.array(values) + np.array(mesh_deltas.get(name, (0, 0, 0)))
    assert np.max(abs(actual - expected)) < 1e-8, name
assert actions_before == actions_snapshot() and limits_before == limits_snapshot()
scene.frame_set(old_frame)
bpy.context.view_layer.update()
if old_mode != 'OBJECT':
    bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version = 0
assert digest(TARGET) == before_hash, 'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET), check_existing=False)
assert digest(SOURCE) == source_hash
audit = {'revision': scene['upper_limb_length_revision'], 'target_length_m': LENGTH,
         'input_rig_sha256': before_hash, 'output_rig_sha256': digest(TARGET),
         'source_sha256': source_hash, 'source_unchanged': True,
         'measurements_before': before_measure, 'measurements_after': after_measure,
         'bone_translations_m': {k: list(v) for k, v in bone_deltas.items()},
         'mesh_translations_m': {k: list(v) for k, v in mesh_deltas.items()},
         'bone_rest_matrices_before': bone_before, 'mesh_vertices_before': mesh_before,
         'actions_before': actions_before, 'limits_before': limits_before,
         'keyframes_preserved': True, 'limits_preserved': True}
(ROOT / 'robot_authoring/inspection/upper_limb_lengths.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
path = ROOT / 'robot_authoring/inspection/build_report.json'
build = json.loads(path.read_text(encoding='utf8'))
build.update(joint_rows=rows, collision_shapes=colliders,
             upper_limb_length_revision=scene['upper_limb_length_revision'], upper_limb_target_length_m=LENGTH)
for key in ['elbow_pitch', 'wrist_fixed', 'hand', 'knee_pitch', 'ankle_fixed', 'foot']:
    build['parts'][key]['zero_pivot_before_global_shift_m'][2] += mesh_deltas['L_' + key + '_visual'].z
path.write_text(json.dumps(build, ensure_ascii=False, indent=2), encoding='utf8')
sys.path.insert(0, str(ROOT / 'robot_authoring'))
from export_description import export_robot
export_robot()
print('UPPER_LIMBS_UNIFIED', json.dumps(after_measure))
bpy.ops.object.mode_set(mode='OBJECT')
scene.frame_set(1)
camera = scene.camera
allverts = np.array([v.co for row in meshes for v in bpy.data.objects[row['object']].data.vertices])
center = Vector(((allverts[:,0].min()+allverts[:,0].max())/2, 0, (allverts[:,2].min()+allverts[:,2].max())/2))
for label, offset in [('perspective', (.28, -.40, .23)), ('side', (0, .45, 0)), ('front', (.45, 0, 0))]:
    camera.hide_set(False)
    camera.location = center + Vector(offset)
    camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(ROOT / 'robot_authoring' / ('preview_' + label + '.png'))
    bpy.ops.render.render(write_still=True)
print('UPPER_LIMBS_EXPORT_COMPLETE')
