"""Frame the complete user body reference without saving camera changes."""
import bpy
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
s=bpy.context.scene
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
s.frame_set(1);bpy.context.view_layer.update()
body=bpy.data.objects['BodyReference_visual'];body.hide_render=False
dep=bpy.context.evaluated_depsgraph_get();points=[]
for obj in bpy.data.objects:
    if obj.type!='MESH' or obj.hide_render or obj.hide_get():continue
    evaluated=obj.evaluated_get(dep)
    points.extend([evaluated.matrix_world@Vector(corner) for corner in evaluated.bound_box])
v=np.array(points,dtype=float);lo=v.min(0);hi=v.max(0)
center=Vector((lo+hi)/2);camera=s.camera
camera.hide_set(False);camera.location=center+Vector((0,.8,0))
camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.ortho_scale=max(hi[0]-lo[0],(hi[2]-lo[2])*s.render.resolution_x/s.render.resolution_y)*1.12
s.render.filepath=str(ROOT/'robot_authoring/preview_body_reference.png')
bpy.ops.render.render(write_still=True)
print('FULL_BODY_PREVIEW_RENDERED')
