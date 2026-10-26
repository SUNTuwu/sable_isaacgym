"""CPU-only review renders; no training, physics, or saved pose changes."""
import bpy, json
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/model_sync/20261026_powered_ankles'
s=bpy.context.scene;r=bpy.data.objects['SableRobot'];s.frame_set(5)
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=32;s.cycles.use_denoising=False
s.render.resolution_x=1000;s.render.resolution_y=650;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG';s.render.film_transparent=False
if bpy.data.objects.get('BodyReference_visual'):bpy.data.objects['BodyReference_visual'].hide_render=True
rows=json.loads(r['mesh_table_json']);dg=bpy.context.evaluated_depsgraph_get();vs=[]
for row in rows:
 o=bpy.data.objects[row['object']].evaluated_get(dg);m=o.to_mesh();vs.extend(o.matrix_world@v.co for v in m.vertices);o.to_mesh_clear()
center=Vector(tuple((min(v[i] for v in vs)+max(v[i] for v in vs))/2 for i in range(3)))
# Explicit review lighting/materials affect only this unsaved preview scene.
for mat in bpy.data.materials:
 color=tuple(mat.diffuse_color);mat.use_nodes=True
 bsdf=mat.node_tree.nodes.get('Principled BSDF')
 if bsdf:bsdf.inputs['Base Color'].default_value=color;bsdf.inputs['Roughness'].default_value=.6
s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.65,.65,.65,1);s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.3
for label,offset,power in [('key',(.2,-.35,.4),2),('fill',(-.25,.2,.2),1)]:
 light=bpy.data.lights.new('Review_'+label,'AREA');light.energy=power;light.shape='DISK';light.size=.4
 obj=bpy.data.objects.new('Review_'+label,light);s.collection.objects.link(obj);obj.location=center+Vector(offset);obj.rotation_euler=(center-obj.location).to_track_quat('-Z','Y').to_euler()
cam=s.camera;cam.data.type='ORTHO';cam.data.ortho_scale=.34
for name,offset in [('side',(0,-.5,.025)),('perspective',(.3,-.5,.27))]:
 cam.location=center+Vector(offset);cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();s.render.filepath=str(out/('stand_'+name+'.png'));bpy.ops.render.render(write_still=True)
print('PREVIEWS_RENDERED')
