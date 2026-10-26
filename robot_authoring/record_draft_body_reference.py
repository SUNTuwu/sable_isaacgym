"""Record the source body's evaluated matrix, without saving the source."""
import bpy,json,hashlib
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert Path(bpy.data.filepath).resolve()==(ROOT/'blender/draft.blend').resolve()
bpy.context.view_layer.update()
path=ROOT/'robot_authoring/inspection/motor_offsets.json'
audit=json.loads(path.read_text(encoding='utf8'))
assert hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest()==audit['draft_reference_sha256']
body=bpy.data.objects['Cylinder.001']
assert np.max(abs(np.array([v.co for v in body.data.vertices])-np.array(audit['chest_root']['body_vertices'])))<1e-8
audit['chest_root']['body_matrix_world']=[list(r) for r in body.matrix_world]
audit['chest_root']['body_transform_provenance']='Read directly from user draft after dependency-graph update; initial pre-evaluation import snapshot was identity'
path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'robot_authoring/inspection/build_report.json';build=json.loads(path.read_text(encoding='utf8'))
build['motor_offsets']=audit;path.write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
print('DRAFT_BODY_REFERENCE_MATRIX_RECORDED')
