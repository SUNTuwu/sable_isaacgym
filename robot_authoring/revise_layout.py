"""Revise the saved rig once; protect source, animation and joint constraints."""
import bpy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from mathutils import Vector

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'robot_authoring'))
from layout_geometry import apply_layout, update_build_report

TARGET = ROOT/'blender/sable_rig.blend'
SOURCE = ROOT/'blender/draft.blend'
BACKUP = ROOT/'待删除/2026-10-25_before_layout_100_50_forearm40_capsule_mount'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(bpy.data.filepath).resolve() == TARGET.resolve()
before_hash, source_hash = digest(TARGET), digest(SOURCE)
assert before_hash == digest(BACKUP/'blender/sable_rig.blend'), 'Concurrent edit since backup'
baseline = json.loads((BACKUP/'robot_authoring/validation.json').read_text(encoding='utf8'))
assert baseline['passed'] and baseline['rig_sha256'] == before_hash
rig, scene = bpy.data.objects['SableRobot'], bpy.context.scene
assert np.max(abs(np.array(rig.matrix_world)-np.eye(4))) < 1e-8
old_frame, old_mode = scene.frame_current, rig.mode

def actions_snapshot():
    return [{'name': a.name, 'curves': [{'path': f.data_path, 'index': f.array_index,
            'keys': [[list(k.co), list(k.handle_left), list(k.handle_right), k.interpolation] for k in f.keyframe_points]}
            for layer in a.layers for strip in layer.strips for bag in strip.channelbags for f in bag.fcurves]}
            for a in bpy.data.actions]

def limits_snapshot():
    return {p.name: [[c.name, c.type]+[getattr(c, k, None) for k in
            ['min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']] for c in p.constraints] for p in rig.pose.bones}

actions_before, limits_before = actions_snapshot(), limits_snapshot()
scene.frame_set(1)
audit = apply_layout(rig, scene)
assert actions_snapshot() == actions_before and limits_snapshot() == limits_before
for name, values in audit['bone_rest_matrices_before'].items():
    expected = np.array(values); expected[:3, 3] += np.array(audit['bone_translations_m'].get(name, (0, 0, 0)))
    error = abs(np.array(rig.data.bones[name].matrix_local)-expected)
    assert error[:3, 3].max() < 1e-8 and error[:3, :3].max() < 5e-7, name
for name, values in audit['mesh_vertices_before'].items():
    m = np.array(audit['mesh_transforms_world'].get(name, np.eye(4)))
    actual = np.array([v.co for v in bpy.data.objects[name].data.vertices])
    assert np.max(abs(actual-(np.array(values)@m[:3, :3].T+m[:3, 3]))) < 1e-8, name
notes = bpy.data.texts['START_HERE_使用说明']; notes.clear()
notes.write((ROOT/'robot_authoring/README.md').read_text(encoding='utf8'))
scene.frame_set(old_frame); bpy.context.view_layer.update()
if old_mode != 'OBJECT': bpy.ops.object.mode_set(mode=old_mode)
bpy.context.preferences.filepaths.save_version = 0
assert digest(TARGET) == before_hash, 'Concurrent save detected'
bpy.ops.wm.save_as_mainfile(filepath=str(TARGET), check_existing=False)
assert digest(SOURCE) == source_hash
audit.update(input_rig_sha256=before_hash, output_rig_sha256=digest(TARGET), source_sha256=source_hash,
             baseline_validation_passed=True, baseline_check_count=len(baseline['checks']),
             actions_before=actions_before, limits_before=limits_before, keyframes_preserved=True, limits_preserved=True)
(ROOT/'robot_authoring/inspection/layout_revision.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
path = ROOT/'robot_authoring/inspection/build_report.json'
build = json.loads(path.read_text(encoding='utf8')); update_build_report(build, rig, audit)
path.write_text(json.dumps(build, ensure_ascii=False, indent=2), encoding='utf8')
from export_description import export_robot
export_robot()
print('LAYOUT_APPLIED', json.dumps({k: audit[k] for k in ['revision', 'front_translation_x_m', 'rear_translation_x_m', 'forearm_translation_z_m']}))
bpy.ops.object.mode_set(mode='OBJECT'); scene.frame_set(1)
camera = scene.camera
meshes = json.loads(rig['mesh_table_json'])
vv = np.array([v.co for r in meshes for v in bpy.data.objects[r['object']].data.vertices])
center = Vector(((vv[:, 0].min()+vv[:, 0].max())/2, 0, (vv[:, 2].min()+vv[:, 2].max())/2))
for label, offset in [('perspective', (.28, -.40, .23)), ('side', (0, .45, 0)), ('front', (.45, 0, 0))]:
    camera.hide_set(False); camera.location = center+Vector(offset)
    camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(ROOT/'robot_authoring'/('preview_'+label+'.png'))
    bpy.ops.render.render(write_still=True)
print('LAYOUT_EXPORT_AND_PREVIEWS_COMPLETE')
