"""File/kinematic checks only. Does not claim dynamic or training validation."""
import ast
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from run_io import atomic_json, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT / 'runs/stand5_20261025_v11_symmetric/runtime'
if not OLD.exists():
    OLD = ROOT / '待删除/2026-10-25_cleanup_latest/files/runs/stand5_20261025_v11_symmetric/runtime'


def main():
    model = json.loads((HERE / 'assets/physical_manifest.json').read_text())
    old_model = json.loads((OLD / 'assets/physical_manifest.json').read_text())
    differences = [k for k in model if model[k] != old_model.get(k)]
    assert differences == ['authored_sha256'], differences
    new = ET.parse(HERE / 'assets/sable.xml')
    old = ET.parse(OLD / 'assets/sable.xml')
    bounds = {}
    for name in ['L_hip_pitch', 'R_hip_pitch']:
        joint = new.find(".//joint[@name='%s']" % name)
        old_joint = old.find(".//joint[@name='%s']" % name)
        values = np.fromstring(joint.get('range'), sep=' ')
        assert np.allclose(np.degrees(values), [-85, 24], atol=1e-6)
        bounds[name] = np.degrees(values).tolist()
        joint.set('range', old_joint.get('range'))
    assert ET.tostring(new.getroot()) == ET.tostring(old.getroot())
    meshes = list((HERE / 'assets/meshes').glob('*.obj'))
    assert len(meshes) == 22
    assert all(sha256(p) == sha256(OLD / 'assets/meshes' / p.name) for p in meshes)
    table = json.loads((HERE / 'assets/gait_reference.json').read_text())
    assert table['source_asset_sha256'] == sha256(HERE / 'assets/sable.xml')
    assert table['source_manifest_sha256'] == sha256(HERE / 'assets/physical_manifest.json')
    assert table['builder_sha256'] == sha256(HERE / 'guided_reference.py')
    limits = {j.get('name'): np.fromstring(j.get('range'), sep=' ') for j in ET.parse(HERE / 'assets/sable.xml').iter('joint') if j.get('range')}
    deltas = np.asarray(table['deltas_rad'])
    for foot, pair in enumerate(table['joint_pairs']):
        for axis, name in enumerate(pair):
            angles = model['default_angles'][name] + deltas[:, :, foot, axis]
            assert angles.min() >= limits[name][0] and angles.max() <= limits[name][1]
    old_table = json.loads((HERE / 'assets/gait_reference_previous.json').read_text())
    common_speeds=[v for v in old_table['speeds_m_s'] if v in table['speeds_m_s']]
    old_deltas=np.asarray(old_table['deltas_rad'])
    table_difference = max(float(np.max(np.abs(deltas[table['speeds_m_s'].index(v)]-old_deltas[old_table['speeds_m_s'].index(v)]))) for v in common_speeds)
    for source in HERE.glob('*.py'):
        ast.parse(source.read_text(), filename=str(source))
    immutable = json.loads((OLD.parent / 'runtime_manifest.json').read_text())
    assert all(sha256(OLD / name) == digest for name, digest in immutable['files'].items())
    files = sorted(HERE.glob('*.py')) + sorted((HERE / 'assets').rglob('*')) + [HERE / 'dashboard.html']
    hashes = {str(f.relative_to(HERE)): sha256(f) for f in files if f.is_file()}
    report = {'passed': True, 'scope': 'Static only; GPU standing, PPO, migration replay, HTTP service and runtime limits still require live checks.',
              'hip_limits_deg': bounds, 'xml_only_hip_upper_changed': True,
              'manifest_changed_keys': differences, 'unchanged_visual_meshes': len(meshes),
              'reference_table_max_difference_from_previous_rad': table_difference,'compared_speed_grid':common_speeds,'new_reference_speed_grid':table['speeds_m_s'],
              'reference_max_ik_error_m': table['max_ik_error_m'],
              'reference_min_limit_margin_deg': table['minimum_limit_margin_deg'],
              'old_v11_runtime_files_verified': len(immutable['files']),
              'old_v11_evidence_location':str(OLD),
              'source_sha256': hashes}
    atomic_json(HERE / 'verification/static_model.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
