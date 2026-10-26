"""Apply the 2026-10-25 requested hip-pitch upper limit without changing pose keys."""
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'robot_authoring'))
from export_description import effective_angle, export_robot

OUT = ROOT / 'results/model_sync/20261025_hip_pitch24'
TARGET = ROOT / 'blender/sable_rig.blend'
NAMES = ['L_hip_pitch', 'R_hip_pitch']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rig = bpy.data.objects['SableRobot']
scene = bpy.context.scene


def snapshot():
    actions = []
    for action in bpy.data.actions:
        curves = []
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        curves.append([curve.data_path, curve.array_index,
                            [[list(k.co), list(k.handle_left), list(k.handle_right),
                              k.interpolation, k.handle_left_type, k.handle_right_type]
                             for k in curve.keyframe_points]])
        actions.append([action.name, curves])
    geometry = {obj.name: {
        'matrix_basis': [list(r) for r in obj.matrix_basis],
        'vertices': [list(v.co) for v in obj.data.vertices],
        'faces': [list(p.vertices) for p in obj.data.polygons],
        'vertex_groups': [g.name for g in obj.vertex_groups],
    } for obj in bpy.data.objects if obj.type == 'MESH'}
    rest = {b.name: [b.parent.name if b.parent else None,
            [list(r) for r in b.matrix_local], list(b.head_local), list(b.tail_local)]
            for b in rig.data.bones}
    limits = {p.name: [[c.name, c.owner_space, c.influence,
               [getattr(c, 'use_limit_' + a) for a in 'xyz'],
               [getattr(c, 'min_' + a) for a in 'xyz'],
               [getattr(c, 'max_' + a) for a in 'xyz']]
              for c in p.constraints if c.type == 'LIMIT_ROTATION']
              for p in rig.pose.bones}
    return {'actions': actions, 'geometry': geometry, 'rest': rest, 'limits': limits}


if '--verify-only' in sys.argv:
    audit = json.loads((OUT / 'authoring_audit.json').read_text(encoding='utf8'))
    current = snapshot()
    for key in ['actions', 'geometry', 'rest', 'limits']:
        value = hashlib.sha256(json.dumps(current[key], sort_keys=True).encode()).hexdigest()
        assert value == audit['after_snapshot_sha256'][key], key
    assert sha(TARGET) == audit['rig_sha256']
    print('SAVED_RIG_REOPEN_VERIFIED', json.dumps(audit['limits_after_deg']))
else:
    before_files = json.loads((OUT / 'local_before_manifest.json').read_text(encoding='utf8'))
    assert sha(TARGET) == before_files['files']['blender/sable_rig.blend']
    protected = {str(p): sha(ROOT / p) for p in [Path('blender/draft.blend'),
        Path('robot_authoring/source/original_draft_2026-10-24.blend')]}
    previous_model = json.loads((ROOT / 'deployment/legged_gym/assets/authored_model.json').read_text(encoding='utf8'))
    before = snapshot()
    for name in NAMES:
        constraints = [c for c in rig.pose.bones[name].constraints if c.type == 'LIMIT_ROTATION']
        assert len(constraints) == 1
        c = constraints[0]
        assert c.use_limit_y and c.owner_space == 'LOCAL'
        c.max_y = math.radians(24)
    bpy.context.view_layer.update()
    after = snapshot()
    assert all(before[k] == after[k] for k in ['actions', 'geometry', 'rest'])
    for name in before['limits']:
        expected = json.loads(json.dumps(before['limits'][name]))
        if name in NAMES:
            expected[0][-1][1] = after['limits'][name][0][-1][1]
            assert abs(expected[0][-1][1] - math.radians(24)) < 1e-7
        assert expected == after['limits'][name], name
    assert sha(TARGET) == before_files['files']['blender/sable_rig.blend']
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(TARGET), check_existing=False)
    manifest = export_robot()
    saved_frame = scene.frame_current
    scene.frame_set(5)
    bpy.context.view_layer.update()
    q = {r['name']: effective_angle(rig, r['name']) for r in manifest['joints'] if r['kind'] == 'revolute'}
    assert abs(q['spine_pitch'] - math.radians(-40)) < 1e-6
    assert all(abs(q[n] - previous_model['angles_rad'][n]) < 1e-6 for n in q)
    geometries = []
    for row in json.loads(rig['mesh_table_json']):
        obj = bpy.data.objects[row['object']]
        assert max(abs(obj.matrix_world[i][j] - float(i == j)) for i in range(4) for j in range(4)) < 1e-8
        geometries.append(dict(row, vertices=[list(v.co) for v in obj.data.vertices],
            faces=[list(f.vertices) for f in obj.data.polygons], bone=obj.vertex_groups[0].name))
    payload = dict(previous_model, rig_sha256=sha(TARGET),
        source_rig_sha256=before_files['files']['blender/sable_rig.blend'],
        angles_rad=q, angles_deg={k: math.degrees(v) for k, v in q.items()},
        root_matrix_world=[list(r) for r in rig.matrix_world @ rig.pose.bones['ROOT'].matrix],
        joints=manifest['joints'], meshes=geometries, capsules=manifest['collision_shapes'],
        bone_world_at_stand={p.name: [list(r) for r in rig.matrix_world @ p.matrix] for p in rig.pose.bones},
        limit_revision='2026-10-25: bilateral hip pitch upper limit +24 deg; all other authoring state preserved',
        scope='Static authoring and pose preservation checked; new-limit dynamics/training not performed')
    (ROOT / 'deployment/legged_gym/assets/authored_model.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')
    scene.frame_set(saved_frame)
    assert all(sha(ROOT / p) == value for p, value in protected.items())
    audit = {'revision': payload['limit_revision'], 'rig_sha256': sha(TARGET),
        'previous_rig_sha256': before_files['files']['blender/sable_rig.blend'],
        'protected_sha256': protected, 'preserved': ['pose keys and handles', 'mesh geometry',
        'rest bones and joint axes', 'all other limits', 'frame-5 standing pose', 'single opposed hip roll'],
        'limits_before_deg': {n: [math.degrees(before['limits'][n][0][-2][1]), math.degrees(before['limits'][n][0][-1][1])] for n in NAMES},
        'limits_after_deg': {n: [math.degrees(after['limits'][n][0][-2][1]), math.degrees(after['limits'][n][0][-1][1])] for n in NAMES},
        'after_snapshot_sha256': {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k, v in after.items()},
        'training_started': False}
    (OUT / 'authoring_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    print('HIP_PITCH_LIMIT_UPDATED', json.dumps(audit['limits_after_deg']))
