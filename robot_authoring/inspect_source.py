"""Read-only Blender source inventory; run in a separate background process."""
import bpy
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'robot_authoring' / 'inspection'
OUT.mkdir(parents=True, exist_ok=True)
source = Path(bpy.data.filepath)
report = {'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'blender_version': bpy.app.version_string, 'scale_length': bpy.context.scene.unit_settings.scale_length,
          'objects': []}
for obj in bpy.data.objects:
    row = {'name': obj.name, 'type': obj.type, 'parent': obj.parent.name if obj.parent else None,
           'matrix': [list(x) for x in obj.matrix_world],
           'modifiers': [{'name': m.name, 'type': m.type,
                          'mirror_axis': list(m.use_axis) if m.type == 'MIRROR' else None,
                          'mirror_object': m.mirror_object.name if m.type == 'MIRROR' and m.mirror_object else None}
                         for m in obj.modifiers], 'custom': {k: str(v) for k,v in obj.items()},
           'hidden': obj.hide_viewport, 'collections': [c.name for c in obj.users_collection]}
    if obj.type == 'MESH':
        vs = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
        row['vertices'] = vs.tolist()
        row['faces'] = [list(p.vertices) for p in obj.data.polygons]
        adjacency = [set() for v in vs]
        for edge in obj.data.edges:
            a,b = edge.vertices
            adjacency[a].add(b); adjacency[b].add(a)
        remaining = set(range(len(vs))); components = []
        while remaining:
            seed = min(remaining); todo = [seed]; ids = set()
            while todo:
                i = todo.pop()
                if i in ids: continue
                ids.add(i); todo.extend(adjacency[i] - ids)
            remaining -= ids
            vertices = vs[sorted(ids)]
            eig, axes = np.linalg.eigh(np.cov(vertices.T)) if len(ids)>2 else (np.zeros(3),np.eye(3))
            components.append({'indices': sorted(ids), 'count': len(ids), 'center': vertices.mean(0).tolist(),
                               'eigenvalues': eig.tolist(), 'axes': axes.T.tolist()})
        row['components'] = sorted(components, key=lambda x:x['count'])
        row['bounds'] = [vs.min(0).tolist(), vs.max(0).tolist()] if len(vs) else []
    report['objects'].append(row)
(OUT/'source_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('SOURCE_SHA256',report['sha256'],'BLENDER',report['blender_version'],'SCALE',report['scale_length'])
for o in report['objects']:
    print(json.dumps({k:o[k] for k in ['name','type','parent','modifiers','custom']},ensure_ascii=True))
    for c in o.get('components',[]):
        print('COMP',c['count'],np.round(c['center'],6).tolist(),'EIG',np.round(c['eigenvalues'],9).tolist(), 'AXES',np.round(c['axes'],4).tolist())
