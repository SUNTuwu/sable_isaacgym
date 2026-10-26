"""Analytic capsule metadata plus a matching rigidly skinned visual mesh."""
import bpy
import json
import math
import numpy as np
from mathutils import Matrix, Vector

TERMINALS={'L_hand_visual','R_hand_visual','L_foot_visual','R_foot_visual'}

def capsule_vertices(center,axis,radius,segment,radial=32,hemisphere=8):
    center=Vector(center); rotation=Vector((0,0,1)).rotation_difference(Vector(axis).normalized())
    local=[(0,0,-segment/2-radius)]
    for sign in [-1,1]:
        angles=[-math.pi/2+j*math.pi/(2*hemisphere) for j in range(1,hemisphere+1)] if sign<0 else [j*math.pi/(2*hemisphere) for j in range(hemisphere)]
        for a in angles:
            for k in range(radial):
                t=2*math.pi*k/radial
                local.append((radius*math.cos(a)*math.cos(t),radius*math.cos(a)*math.sin(t),sign*segment/2+radius*math.sin(a)))
    local.append((0,0,segment/2+radius)); top=len(local)-1
    faces=[(0,1+(i+1)%radial,1+i) for i in range(radial)]
    rings=2*hemisphere
    for j in range(rings-1):
        a=1+j*radial; b=a+radial
        faces += [(a+i,a+(i+1)%radial,b+(i+1)%radial,b+i) for i in range(radial)]
    a=1+(rings-1)*radial
    faces += [(a+i,a+(i+1)%radial,top) for i in range(radial)]
    return [center+rotation@Vector(v) for v in local],faces

def replace_terminal(obj,rig):
    assert obj.name in TERMINALS and not obj.get('collision_shape')
    assert len(obj.vertex_groups)==1
    bone=obj.vertex_groups[0].name
    world=np.array([obj.matrix_world@v.co for v in obj.data.vertices])
    assert len(world)==8, 'Expected the original eight-corner terminal box'
    center=world.mean(0); centered=world-center
    eigen,basis=np.linalg.eigh(centered.T@centered/len(world))
    axis=basis[:,-1]
    if axis[np.argmax(abs(axis))]<0: axis=-axis
    radius=float(np.mean(np.sqrt(eigen[:2])))
    total=float(np.ptp(centered@axis)); segment=total-2*radius
    assert segment>0 and abs(eigen[0]-eigen[1])<.02*eigen[1]
    vertices,faces=capsule_vertices(center,axis,radius,segment)
    inv=obj.matrix_world.inverted()
    mesh=bpy.data.meshes.new(obj.name+'_capsule_mesh')
    mesh.from_pydata([inv@v for v in vertices],[],faces); mesh.update()
    old=obj.data
    for mat in old.materials: mesh.materials.append(mat)
    for p in mesh.polygons: p.use_smooth=True
    obj.data=mesh
    vg=obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
    vg.add(list(range(len(vertices))),1.,'REPLACE')
    if old.users==0: bpy.data.meshes.remove(old)
    frame=(rig.matrix_world@rig.data.bones[bone].matrix_local).inverted()
    local_center=frame@Vector(center); local_axis=(frame.to_3x3()@Vector(axis)).normalized()
    collider={'name':obj.name.replace('_visual','_contact'),'visual_object':obj.name,'bone':bone,'type':'capsule',
        'radius_m':radius,'cylinder_length_m':segment,'total_length_m':total,
        'center_bone_m':list(local_center),'axis_bone':list(local_axis),
        'center_world_zero_m':center.tolist(),'axis_world_zero':axis.tolist(),
        'endpoints_bone_m':[list(local_center-local_axis*segment/2),list(local_center+local_axis*segment/2)],
        'source':'fit to current named terminal box; same center, long axis and overall length; rounded corners',
        'blender_simulation_active':False,'native_usd_collision_enabled':True}
    obj['collision_shape']='CAPSULE'; obj['collision_radius_m']=radius
    obj['collision_cylinder_length_m']=segment; obj['collision_total_length_m']=total
    obj['collision_description_json']=json.dumps(collider)
    obj['part_geometry_changed']=True; obj['geometry_change_reason']='User requested terminal capsule instead of box'
    return collider

def export_collision_stage(path,rig,colliders,frames,fps,start,end):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1.); UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    stage.SetFramesPerSecond(fps); stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(start); stage.SetEndTimeCode(end)
    root=UsdGeom.Xform.Define(stage,'/FootColliders'); stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().customLayerData={'scope':'Four terminal colliders only; link animation follows the Blender rig. Merge into physical links when constructing the complete robot articulation.'}
    def usd_matrix(matrix): return Gf.Matrix4d(*[float(matrix[j][i]) for i in range(4) for j in range(4)])
    for c in colliders:
        link=UsdGeom.Xform.Define(stage,'/FootColliders/'+c['bone'])
        op=link.AddTransformOp()
        op.Set(usd_matrix(rig.matrix_world@rig.data.bones[c['bone']].matrix_local))
        for sample in frames: op.Set(usd_matrix(sample['matrices'][c['bone']]),sample['frame'])
        link.GetPrim().SetCustomDataByKey('robot:bone',c['bone'])
        capsule=UsdGeom.Capsule.Define(stage,str(link.GetPath())+'/'+c['name'])
        capsule.CreateAxisAttr('Z'); capsule.CreateRadiusAttr(c['radius_m'])
        capsule.CreateHeightAttr(c['cylinder_length_m'])
        capsule.CreateExtentAttr([(-c['radius_m'],-c['radius_m'],-c['total_length_m']/2),(c['radius_m'],c['radius_m'],c['total_length_m']/2)])
        capsule.CreateDisplayColorAttr([(.08,.53,.49)])
        q=Vector((0,0,1)).rotation_difference(Vector(c['axis_bone']))
        transform=Matrix.Translation(Vector(c['center_bone_m']))@q.to_matrix().to_4x4()
        capsule.AddTransformOp().Set(usd_matrix(transform))
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim()).CreateCollisionEnabledAttr(True)
    stage.GetRootLayer().Export(str(path))
    return {'path':path.name,'type':'native UsdGeomCapsule + UsdPhysicsCollisionAPI','count':len(colliders),
            'scope':'terminal collider definitions; no complete robot mass/inertia/articulation yet'}
