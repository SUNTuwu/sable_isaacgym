"""Read current saved authored model; never alter/save Blender state."""
import bpy,json,math,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import effective_angle
rig=bpy.data.objects['SableRobot'];bpy.context.scene.frame_set(5);bpy.context.view_layer.update()
m=json.loads((ROOT/'robot_authoring/exports/robot_description.json').read_text());geometries=[]
for row in json.loads(rig['mesh_table_json']):
 o=bpy.data.objects[row['object']];geometries.append(dict(row,vertices=[list(v.co) for v in o.data.vertices],faces=[list(p.vertices) for p in o.data.polygons],bone=o.vertex_groups[0].name))
q={r['name']:effective_angle(rig,r['name']) for r in m['joints'] if r['kind']=='revolute'}
payload={'rig_sha256':hashlib.sha256((ROOT/'blender/sable_rig.blend').read_bytes()).hexdigest(),'frame':5,'angles_rad':q,'angles_deg':{k:math.degrees(v) for k,v in q.items()},'root_matrix_world':[list(x) for x in rig.matrix_world@rig.pose.bones['ROOT'].matrix],'joints':m['joints'],'meshes':geometries,'capsules':m['collision_shapes'],'bone_world_at_stand':{p.name:[list(x) for x in rig.matrix_world@p.matrix] for p in rig.pose.bones},'scope':'Read-only powered ankle export; physical validation pending'}
out=ROOT/'deployment/legged_gym_ankle_inline_20261026/assets/authored_model.json';assert not out.exists();out.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
print('EXPORTED',out)
