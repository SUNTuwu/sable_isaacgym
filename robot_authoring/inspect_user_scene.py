import bpy,json
import numpy as np
print('SCENE',json.dumps({'active':bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else None,
    'objects':[{'name':o.name,'type':o.type,'parent':o.parent.name if o.parent else None,
    'hide':o.hide_get(),'render_hide':o.hide_render,'matrix':[list(r) for r in o.matrix_world],
    'modifiers':[(m.type,getattr(getattr(m,'object',None),'name',None)) for m in o.modifiers]}
    for o in bpy.data.objects if not o.name.startswith(('GUIDE','CTRL','Authoring'))],
    'armatures':{o.name:{'action':o.animation_data.action.name if o.animation_data and o.animation_data.action else None,
    'pose':{p.name:{'rotation':list(p.rotation_euler),'location':list(p.location),'scale':list(p.scale)} for p in o.pose.bones
    if np.linalg.norm(p.rotation_euler)>1e-6 or np.linalg.norm(p.location)>1e-6}} for o in bpy.data.objects if o.type=='ARMATURE'}},ensure_ascii=False))
