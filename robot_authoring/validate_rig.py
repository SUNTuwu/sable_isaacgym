"""Reopen the saved rig and exercise poses/constraints without saving mutations."""
import bpy
import hashlib
import json
import math
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from mathutils import Vector, Matrix
from pxr import Usd, UsdGeom, UsdSkel, UsdPhysics, Gf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'robot_authoring'))
from export_description import effective_angle
from capsule_geometry import TERMINALS

rig=bpy.data.objects['SableRobot']; scene=bpy.context.scene
build=json.loads((ROOT/'robot_authoring/inspection/build_report.json').read_text(encoding='utf8'))
source=json.loads((ROOT/'robot_authoring/inspection/source_inventory.json').read_text(encoding='utf8'))
src={x['name']:x for x in source['objects']}
rows=json.loads(rig['joint_table_json']); meshes=json.loads(rig['mesh_table_json'])
colliders=json.loads(rig.get('collision_shapes_json','[]'))
waist_offset=scene.get('waist_rearward_offset_m',0.)
limb_audit=None
if scene.get('upper_limb_length_revision'):
    limb_audit=json.loads((ROOT/'robot_authoring/inspection/upper_limb_lengths.json').read_text(encoding='utf8'))
limb_mesh_deltas=limb_audit['mesh_translations_m'] if limb_audit else {}
limb_bone_deltas=dict(limb_audit['bone_translations_m']) if limb_audit else {}
hind_audit=None
if scene.get('hind_assembly_revision'):
    hind_audit=json.loads((ROOT/'robot_authoring/inspection/hind_assembly.json').read_text(encoding='utf8'))
    for name,delta in hind_audit['bone_translations_m'].items():
        limb_bone_deltas[name]=(np.array(limb_bone_deltas.get(name,(0,0,0)))+np.array(delta)).tolist()
elbow_audit=None
if scene.get('elbow_motor_mount_revision'):
    elbow_audit=json.loads((ROOT/'robot_authoring/inspection/elbow_motor_rear.json').read_text(encoding='utf8'))
hip_audit=None
roll_moved_meshes=set();roll_bone_deltas={}
if scene.get('hip_roll_lowering_revision'):
    hip_audit=json.loads((ROOT/'robot_authoring/inspection/hip_roll_lowering.json').read_text(encoding='utf8'))
    roll_moved_meshes=set(hip_audit['moved_meshes'])
    roll_bone_deltas={n:hip_audit['translation_world_m'] for n in hip_audit['moved_bones']}
    for n,delta in roll_bone_deltas.items():
        limb_bone_deltas[n]=(np.array(limb_bone_deltas.get(n,(0,0,0)))+np.array(delta)).tolist()
shoulder_audit=None
if scene.get('shoulder_roll_lowering_revision'):
    shoulder_audit=json.loads((ROOT/'robot_authoring/inspection/shoulder_roll_lowering.json').read_text(encoding='utf8'))
    roll_moved_meshes.update(shoulder_audit['moved_meshes'])
    for n in shoulder_audit['moved_bones']:
        roll_bone_deltas[n]=shoulder_audit['translation_world_m']
        limb_bone_deltas[n]=(np.array(limb_bone_deltas.get(n,(0,0,0)))+np.array(roll_bone_deltas[n])).tolist()
roll_mesh_deltas={n:audit['translation_world_m'] for audit in [hip_audit,shoulder_audit] if audit for n in audit['moved_meshes']}
layout_audit=None
layout_meshes=set();layout_bone_deltas={}
if scene.get('layout_revision'):
    layout_audit=json.loads((ROOT/'robot_authoring/inspection/layout_revision.json').read_text(encoding='utf8'))
    layout_meshes=set(layout_audit['mesh_transforms_world'])
    layout_bone_deltas=layout_audit['bone_translations_m']
    for n,delta in layout_bone_deltas.items():
        limb_bone_deltas[n]=(np.array(limb_bone_deltas.get(n,(0,0,0)))+np.array(delta)).tolist()
        roll_bone_deltas[n]=(np.array(roll_bone_deltas.get(n,(0,0,0)))+np.array(delta)).tolist()
repositioned_meshes=set(limb_mesh_deltas)|roll_moved_meshes|layout_meshes
checks={}; measurements={}
pitch_audit=None
if scene.get('shoulder_pitch_mount_revision'):
    pitch_audit=json.loads((ROOT/'robot_authoring/inspection/shoulder_pitch_mount.json').read_text(encoding='utf8'))
offset_audit=None
offset_meshes=set()
if scene.get('motor_offset_revision'):
    offset_audit=json.loads((ROOT/'robot_authoring/inspection/motor_offsets.json').read_text(encoding='utf8'))
    offset_meshes=set(offset_audit['mesh_translations_m'])
    repositioned_meshes.update(offset_meshes)

straight_audit=None
if scene.get('straight_limb_revision'):
    straight_audit=json.loads((ROOT/'robot_authoring/inspection/straight_limbs.json').read_text(encoding='utf8'))
    offset_meshes.update(straight_audit['mesh_translations_m'])
    repositioned_meshes.update(offset_meshes)

def offset_delta(name,kind='bone'):
    table='bone_translations_m' if kind=='bone' else 'mesh_translations_m'
    delta=np.array(offset_audit[table].get(name,[0,0,0])) if offset_audit else np.zeros(3)
    if straight_audit:delta=delta+np.array(straight_audit[table].get(name,[0,0,0]))
    if straight_audit and kind=='bone':delta=delta+np.array(straight_audit['manual_bone_translations_m'].get(name,[0,0,0]))
    return delta

def pitch_matrix(name,kind='bone'):
    table='bone_transforms_world' if kind=='bone' else 'mesh_transforms_world'
    m=np.array(pitch_audit[table].get(name,np.eye(4))) if pitch_audit else np.eye(4)
    m[:3,3]+=offset_delta(name,kind)
    return m

def with_layout_mesh_revision(name,vertices):
    vv=np.array(vertices)
    if layout_audit and name in layout_audit['mesh_transforms_world']:
        m=np.array(layout_audit['mesh_transforms_world'][name])
        vv=vv@m[:3,:3].T+m[:3,3]
    m=pitch_matrix(name,'mesh')
    return vv@m[:3,:3].T+m[:3,3]

def with_elbow_mesh_revision(name,vertices):
    vv=np.array(vertices)
    if elbow_audit and name in elbow_audit['mesh_transforms_world']:
        m=np.array(elbow_audit['mesh_transforms_world'][name])
        vv=vv@m[:3,:3].T+m[:3,3]
    if name in roll_mesh_deltas:vv+=np.array(roll_mesh_deltas[name])
    return with_layout_mesh_revision(name,vv)

def with_hind_mesh_revision(name,vertices):
    vv=np.array(vertices)
    if hind_audit and name in hind_audit['mesh_transforms_world']:
        m=np.array(hind_audit['mesh_transforms_world'][name])
        vv=vv@m[:3,:3].T+m[:3,3]
    return with_elbow_mesh_revision(name,vv)
def check(name,condition,detail=None):
    checks[name]=bool(condition)
    if detail is not None: measurements[name]=detail
    if not condition: raise AssertionError(name+': '+str(detail))
def update(): bpy.context.view_layer.update()
def setq(name,degrees): rig.pose.bones[name].rotation_euler.y=math.radians(degrees); update()
def worldverts(name):
    obj=bpy.data.objects[name]; ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh=ev.to_mesh()
    try: return np.array([ev.matrix_world@v.co for v in mesh.vertices])
    finally: ev.to_mesh_clear()
def center(name): return worldverts(name).mean(0)
def rest_matrix_matches(name,expected,include_pitch_mount=True):
    if include_pitch_mount:expected=pitch_matrix(name)@expected
    error=abs(np.array(rig.data.bones[name].matrix_local)-expected)
    return np.max(error[:3,3])<1e-8 and np.max(error[:3,:3])<5e-7

geometry_source=ROOT/scene.get('robot_geometry_source_file','blender/draft.blend')
check('source_sha256_unchanged',hashlib.sha256(geometry_source.read_bytes()).hexdigest()==source['sha256'])
if scene.get('robot_pose_reference_sha256'):
    check('user_draft_reference_preserved',hashlib.sha256((ROOT/'blender/draft.blend').read_bytes()).hexdigest()==scene['robot_pose_reference_sha256'])
check('saved_in_pose_mode',rig.mode=='POSE',rig.mode)
check('13_independent_actuators',sum(r['actuated'] for r in rows)==13)
check('14_hinges_and_4_fixed_terminals',sum(r['kind']=='revolute' for r in rows)==14 and sum(r['kind']=='fixed' for r in rows)==4)
check('named_sources_only',all(x['source_object'] in build['allowlist'].values() for x in meshes))
check('keyframe_action_present',rig.animation_data and rig.animation_data.action.name=='StandPose_EDIT_ME')
check('front_world_z_correction_minus_5_degrees',abs(build['front_shoulder_world_z_correction_deg']+5)<.02,build['front_shoulder_world_z_correction_deg'])
scene.frame_set(1); update()
check('zero_pose_angles',max(abs(effective_angle(rig,r['name'])) for r in rows if r['kind']=='revolute')<1e-6)
if scene.get('waist_alignment_revision'):
    z=rig.data.bones['spine_roll'].head_local.z
    check('straight_waist_with_later_shoulder_offset',max(abs(rig.data.bones[n].head_local.z-z+
        (scene.get('shoulder_roll_lowering_m',0.) if 'shoulder' in n else 0.))
        for n in ['ROOT','spine_pitch','L_shoulder_roll','R_shoulder_roll'])<1e-6)
    correction_path=ROOT/'robot_authoring/inspection/waist_correction.json'
    if correction_path.exists():
        correction=json.loads(correction_path.read_text(encoding='utf8'))
        for name,expected in correction['protected_mesh_vertex_sha256'].items():
            if scene.get('terminal_capsule_revision') and name in TERMINALS: continue
            if name in repositioned_meshes: continue  # Exact revision translation is checked below.
            vertices=np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32)
            check('unchanged_mesh_'+name,hashlib.sha256(vertices.tobytes()).hexdigest()==expected)
        for name,p in correction['protected_bones'].items():
            b=rig.data.bones[name]
            delta=Vector(limb_bone_deltas.get(name,(0,0,0)))
            m=pitch_matrix(name)
            expected_head=m[:3,:3]@np.array(Vector(p['head'])+delta)+m[:3,3]
            expected_tail=m[:3,:3]@np.array(Vector(p['tail'])+delta)+m[:3,3]
            # The user's manual roll edit moved the bone head, leaving its
            # display tail fixed. Preserve that handle length; the head and
            # named-cylinder axis define the physical joint.
            if straight_audit:expected_tail-=np.array(straight_audit['manual_bone_translations_m'].get(name,[0,0,0]))
            check('joint_with_requested_length_revision_'+name,np.linalg.norm(np.array(b.head_local)-expected_head)<1e-8 and np.linalg.norm(np.array(b.tail_local)-expected_tail)<1e-8,
                  {'head_error_m':float(np.linalg.norm(np.array(b.head_local)-expected_head)),
                   'tail_error_m':float(np.linalg.norm(np.array(b.tail_local)-expected_tail)),'control_length_m':b.length})
for pre in ['L','R']:
    for chain in [('shoulder_pitch','elbow_pitch','wrist_fixed'),('hip_pitch','knee_pitch','ankle_fixed')]:
        vv=np.array([rig.data.bones[pre+'_'+n].head_local for n in chain])
        err=float(np.max(np.ptp(vv[:,:2],axis=0)))
        if offset_audit:
            before=np.array([np.array(offset_audit['bone_rest_matrices_before'][pre+'_'+n])[:3,3] for n in chain])
            expected=before+np.array([offset_delta(pre+'_'+n) for n in chain])
            check(pre+'_'+chain[0]+'_independent_center_offsets',np.max(abs(vv-expected))<1e-8,{'max_horizontal_offset_m':err})
        else:check(pre+'_'+chain[0]+'_vertical',err<1e-6,{'max_horizontal_offset_m':err})
        if straight_audit:check(pre+'_'+chain[0]+'_restored_vertical',err<1e-8,err)

max_shape_error=0.; max_bind_error=0.
for r in meshes:
    original=np.array(src[r['source_object']]['vertices'])
    obj=bpy.data.objects[r['object']]
    authored=np.array([v.co for v in obj.data.vertices]); evaluated=worldverts(obj.name)
    if scene.get('terminal_capsule_revision') and obj.name in TERMINALS:
        c=next(c for c in colliders if c['visual_object']==obj.name)
        points=authored-np.array(c['center_world_zero_m']); axis=np.array(c['axis_world_zero'])
        projection=np.clip(points@axis,-c['cylinder_length_m']/2,c['cylinder_length_m']/2)
        surface_error=float(np.max(abs(np.linalg.norm(points-projection[:,None]*axis,axis=1)-c['radius_m'])))
        check('capsule_surface_'+obj.name,surface_error<1e-7,surface_error)
        check('capsule_dimensions_'+obj.name,abs(c['total_length_m']-.030)<1e-6 and abs(c['radius_m']-.005)<1e-6)
        check('capsule_center_'+obj.name,np.linalg.norm(authored.mean(0)-np.array(c['center_world_zero_m']))<1e-7)
        edges={}
        for polygon in obj.data.polygons:
            for edge in polygon.edge_keys: edges[edge]=edges.get(edge,0)+1
        check('capsule_closed_manifold_'+obj.name,all(count==2 for count in edges.values()))
    else:
        check('vertex_count_'+obj.name,len(original)==len(authored))
        # Pairwise distances are invariant under rigid assembly transformations.
        d0=np.linalg.norm(original[:,None,:]-original[None,:,:],axis=-1)
        d1=np.linalg.norm(authored[:,None,:]-authored[None,:,:],axis=-1)
        max_shape_error=max(max_shape_error,float(np.max(abs(d0-d1))))
    max_bind_error=max(max_bind_error,float(np.max(np.linalg.norm(authored-evaluated,axis=1))))
    for v in obj.data.vertices:
        check('rigid_weights_'+obj.name, len(v.groups)==1 and abs(v.groups[0].weight-1)<1e-7)
check('nonterminal_mesh_shapes_rigidly_preserved',max_shape_error<1e-6,max_shape_error)
check('rest_skinning_matches_authored_geometry',max_bind_error<1e-6,max_bind_error)

if colliders:
    audit=json.loads((ROOT/'robot_authoring/inspection/terminal_capsules.json').read_text(encoding='utf8'))
    for name,expected in audit['protected_mesh_vertex_sha256'].items():
        if waist_offset and name=='spine_pitch_visual':continue
        if name in repositioned_meshes:continue
        actual=hashlib.sha256(np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32).tobytes()).hexdigest()
        check('capsule_revision_preserved_'+name,actual==expected)
    expected_bones={n:np.array(m) for n,m in audit['protected_bone_rest_matrices'].items()}
    if waist_offset:
        for n in ['ROOT','spine_pitch']:expected_bones[n][0,3]-=waist_offset
    for n,delta in limb_bone_deltas.items():expected_bones[n][:3,3]+=np.array(delta)
    check('capsule_revision_bones_with_requested_waist_offset',all(rest_matrix_matches(n,m) for n,m in expected_bones.items()))
    collider_stage=Usd.Stage.Open(str(ROOT/'robot_authoring/exports/foot_colliders.usda'))
    capsules=[p for p in collider_stage.Traverse() if p.IsA(UsdGeom.Capsule)]
    check('native_usd_four_capsules',len(capsules)==4)
    cache=UsdGeom.XformCache(Usd.TimeCode(1))
    for c in colliders:
        prim=collider_stage.GetPrimAtPath('/FootColliders/'+c['bone']+'/'+c['name'])
        shape=UsdGeom.Capsule(prim)
        check('usd_collision_enabled_'+c['name'],prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        check('usd_capsule_dimensions_'+c['name'],abs(shape.GetRadiusAttr().Get()-c['radius_m'])<1e-9 and abs(shape.GetHeightAttr().Get()-c['cylinder_length_m'])<1e-9)
        transform=cache.GetLocalToWorldTransform(prim)
        world_center=np.array(transform.Transform(Gf.Vec3d(0,0,0)))
        world_axis=np.array(transform.TransformDir(Gf.Vec3d(0,0,1)))
        check('usd_capsule_transform_'+c['name'],np.linalg.norm(world_center-np.array(c['center_world_zero_m']))<1e-7 and np.linalg.norm(world_axis-np.array(c['axis_world_zero']))<1e-6)

if waist_offset:
    audit=json.loads((ROOT/'robot_authoring/inspection/waist_backshift.json').read_text(encoding='utf8'))
    delta=np.array(audit['translation_world_m'])
    check('waist_shift_exactly_5mm_rearward',np.linalg.norm(delta-[-.005,0,0])<1e-8)
    actual=np.array([v.co for v in bpy.data.objects['spine_pitch_visual'].data.vertices])
    check('waist_mesh_translation',np.max(abs(actual-np.array(audit['waist_vertices_before'])-delta))<1e-8)
    for name,expected in audit['protected_mesh_vertex_sha256'].items():
        if name in repositioned_meshes:continue
        actual=hashlib.sha256(np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32).tobytes()).hexdigest()
        check('waist_shift_preserved_'+name,actual==expected)
    for name,values in audit['bone_rest_matrices_before'].items():
        expected=np.array(values)
        if name in audit['moved_bones']:expected[:3,3]+=delta
        expected[:3,3]+=np.array(limb_bone_deltas.get(name,(0,0,0)))
        check('waist_shift_bone_'+name,rest_matrix_matches(name,expected))

if limb_audit:
    heights=[]
    for side in ['L','R']:
        for kind,a,b,c in [('upper_arm','shoulder_pitch','elbow_pitch','wrist_fixed'),
                            ('thigh','hip_pitch','knee_pitch','ankle_fixed')]:
            pp=[rig.data.bones[side+'_'+n].head_local for n in [a,b,c]]
            length=(pp[0]-pp[1]).length; lower=(pp[1]-pp[2]).length
            heights.append(pp[1].z)
            if offset_audit:
                pa=np.array(offset_audit['bone_rest_matrices_before'][side+'_'+a])[:3,3]+offset_delta(side+'_'+a)
                pb=np.array(offset_audit['bone_rest_matrices_before'][side+'_'+b])[:3,3]+offset_delta(side+'_'+b)
                check('current_center_distance_'+side+'_'+kind,abs(length-np.linalg.norm(pa-pb))<1e-8,length)
            else:check('44mm_'+side+'_'+kind,abs(length-.044)<1e-8,length)
            if straight_audit:check('restored_44mm_'+side+'_'+kind,abs(length-.044)<1e-8,length)
            target=.050 if hind_audit and kind=='thigh' else limb_audit['measurements_before'][side+'_'+kind]['lower_segment_length_m']
            if layout_audit and kind=='upper_arm':target=.040
            check('lower_segment_current_'+side+'_'+kind,abs(lower-target)<1e-8,lower)
    check('elbows_and_knees_same_zero_height',max(heights)-min(heights)<1e-8,heights)
    for name,values in limb_audit['mesh_vertices_before'].items():
        actual=np.array([v.co for v in bpy.data.objects[name].data.vertices])
        expected=np.array(values)+np.array(limb_mesh_deltas.get(name,(0,0,0)))
        expected=with_hind_mesh_revision(name,expected)
        check('upper_limb_revision_mesh_'+name,np.max(abs(actual-expected))<5e-8)
    for name,values in limb_audit['bone_rest_matrices_before'].items():
        expected=np.array(values); expected[:3,3]+=np.array(limb_bone_deltas.get(name,(0,0,0)))
        check('upper_limb_revision_bone_'+name,rest_matrix_matches(name,expected))
    actions=[{'name':a.name,'curves':[{'path':fc.data_path,'index':fc.array_index,
        'keys':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in fc.keyframe_points]}
        for layer in a.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]}
        for a in bpy.data.actions]
    limits={p.name:[[c.name,c.type]+[getattr(c,k,None) for k in ['min_x','max_x','min_y','max_y','min_z','max_z']]
            for c in p.constraints] for p in rig.pose.bones}
    check('upper_limb_revision_keyframes_preserved',actions==limb_audit['actions_before'])
    check('upper_limb_revision_limits_preserved',limits==limb_audit['limits_before'])

if hind_audit:
    for name,values in hind_audit['mesh_vertices_before'].items():
        actual=np.array([v.co for v in bpy.data.objects[name].data.vertices])
        check('hind_revision_mesh_'+name,np.max(abs(actual-with_hind_mesh_revision(name,values)))<5e-8)
    for name,values in hind_audit['bone_rest_matrices_before'].items():
        expected=np.array(values)
        expected[:3,3]+=np.array(hind_audit['bone_translations_m'].get(name,(0,0,0)))
        expected[:3,3]+=np.array(roll_bone_deltas.get(name,(0,0,0)))
        check('hind_revision_joint_'+name,rest_matrix_matches(name,expected))
    check('hind_revision_keyframes_preserved',actions==hind_audit['actions_before'])
    check('hind_revision_limits_preserved',limits==hind_audit['limits_before'])
    for side in ['L','R']:
        vectors=[]
        for key in ['elbow_pitch','knee_pitch']:
            obj=bpy.data.objects[side+'_'+key+'_visual']
            components=src[obj['source_object']]['components']
            ids=next(c['indices'] for c in components if c['count']==144)
            vertices=np.array([v.co for v in obj.data.vertices])
            pivot=np.array(rig.data.bones[side+'_'+key].head_local)
            vector=vertices[ids].mean(0)-pivot
            vectors.append(vector[[0,2]]/np.linalg.norm(vector[[0,2]]))
            if key=='knee_pitch':
                shaft_ids=next(c['indices'] for c in components if c['count']==64)
                shaft=vertices[shaft_ids]
                _,basis=np.linalg.eigh(np.cov(shaft.T))
                check('knee_output_pivot_preserved_'+side,np.linalg.norm(shaft.mean(0)-pivot)<1e-7)
                check('knee_output_axis_preserved_'+side,abs(basis[1,0])>1-1e-6)
                check('knee_housing_forward_of_joint_'+side,vector[0]>0)
                check('knee_stator_bound_to_thigh_'+side,len(obj.vertex_groups)==1 and obj.vertex_groups[0].name==side+'_hip_pitch')
        reference=np.array(hind_audit['mounting'][side]['reference_elbow_housing_offset_m'])[[0,2]]
        reference/=np.linalg.norm(reference)
        check('knee_housing_retains_forward_mount_'+side,np.linalg.norm(reference-vectors[1])<1e-6)
        shank=(rig.data.bones[side+'_knee_pitch'].head_local-rig.data.bones[side+'_ankle_fixed'].head_local).length
        check('shank_50mm_'+side,abs(shank-.050)<1e-8,shank)

if elbow_audit:
    for name,expected in elbow_audit['protected_mesh_vertex_sha256'].items():
        if name in roll_moved_meshes|layout_meshes|offset_meshes:continue
        vv=np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32)
        check('elbow_move_preserved_'+name,hashlib.sha256(vv.tobytes()).hexdigest()==expected)
    for name,expected in elbow_audit['bone_rest_matrices_before'].items():
        matrix=np.array(expected);matrix[:3,3]+=np.array(roll_bone_deltas.get(name,(0,0,0)))
        check('elbow_move_joint_with_later_mount_revision_'+name,rest_matrix_matches(name,matrix))
    check('elbow_move_keyframes_preserved',actions==elbow_audit['actions_before'])
    check('elbow_move_limits_preserved',limits==elbow_audit['limits_before'])
    for side in ['L','R']:
        name=side+'_elbow_pitch_visual';obj=bpy.data.objects[name]
        vv=np.array([v.co for v in obj.data.vertices])
        expected=with_elbow_mesh_revision(name,elbow_audit['mesh_vertices_before'][name])
        check('elbow_move_rigid_transform_'+side,np.max(abs(vv-expected))<5e-8)
        components=src[obj['source_object']]['components']
        housing_ids=next(c['indices'] for c in components if c['count']==144)
        shaft_ids=next(c['indices'] for c in components if c['count']==64)
        pivot=np.array(rig.data.bones[side+'_elbow_pitch'].head_local)
        offset=vv[housing_ids].mean(0)-pivot
        before=np.array(elbow_audit['mounting'][side]['housing_offset_before_m'])
        expected_offset=pitch_matrix(side+'_elbow_pitch')[:3,:3]@(before*[-1,1,1])
        check('elbow_center_side_view_mirrored_'+side,np.linalg.norm(offset-expected_offset)<1e-7,offset.tolist())
        check('elbow_motor_rear_of_arm_'+side,offset[0]<0 and offset[2]>0)
        check('elbow_output_pivot_preserved_'+side,np.linalg.norm(vv[shaft_ids].mean(0)-pivot)<1e-7)
        _,basis=np.linalg.eigh(np.cov(vv[shaft_ids].T))
        expected_axis=pitch_matrix(side+'_elbow_pitch')[:3,:3]@np.array([0,1,0])
        check('elbow_output_axis_preserved_'+side,abs(np.dot(basis[:,0],expected_axis))>1-1e-6)
        check('elbow_stator_bound_to_upper_arm_'+side,len(obj.vertex_groups)==1 and obj.vertex_groups[0].name==side+'_shoulder_pitch')

if hip_audit:
    delta=np.array(hip_audit['translation_world_m'])
    check('hip_roll_down_exactly_10mm',np.linalg.norm(delta-[0,0,-.010])<1e-8)
    for name,expected in hip_audit['protected_mesh_vertex_sha256'].items():
        if name in layout_meshes|offset_meshes:continue
        if shoulder_audit and name in shoulder_audit['moved_meshes']:continue
        vv=np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32)
        check('hip_lowering_preserved_mesh_'+name,hashlib.sha256(vv.tobytes()).hexdigest()==expected)
    for name,values in hip_audit['mesh_vertices_before'].items():
        vv=np.array([v.co for v in bpy.data.objects[name].data.vertices])
        check('hip_lowering_mesh_translation_'+name,np.max(abs(vv-with_layout_mesh_revision(name,np.array(values)+delta)))<1e-8)
    for name,values in hip_audit['bone_rest_matrices_before'].items():
        expected=np.array(values);expected[:3,3]+=np.array(roll_bone_deltas.get(name,(0,0,0)))
        check('hip_lowering_joint_'+name,rest_matrix_matches(name,expected))
    check('hip_lowering_keyframes_preserved',actions==hip_audit['actions_before'])
    check('hip_lowering_limits_preserved',limits==hip_audit['limits_before'])
    for side,joint in [('L','hip_roll'),('R','R_hip_roll_mimic')]:
        obj=bpy.data.objects[side+'_hip_roll_visual']
        vv=np.array([v.co for v in obj.data.vertices],dtype=np.float64)
        pivot=np.array(rig.data.bones[joint].head_local)
        check('lowered_hip_roll_mesh_center_'+side,np.linalg.norm(vv.mean(0)-pivot)<1e-7)

if shoulder_audit:
    delta=np.array(shoulder_audit['translation_world_m'])
    check('shoulder_roll_down_exactly_10mm',np.linalg.norm(delta-[0,0,-.010])<1e-8)
    for name,expected in shoulder_audit['protected_mesh_vertex_sha256'].items():
        if name in layout_meshes|offset_meshes:continue
        vv=np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float32)
        check('shoulder_lowering_preserved_mesh_'+name,hashlib.sha256(vv.tobytes()).hexdigest()==expected)
    for name,values in shoulder_audit['mesh_vertices_before'].items():
        vv=np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float64)
        check('shoulder_lowering_mesh_translation_'+name,np.max(abs(vv-with_layout_mesh_revision(name,np.array(values)+delta)))<1e-8)
    for name,values in shoulder_audit['bone_rest_matrices_before'].items():
        expected=np.array(values)
        if name in shoulder_audit['moved_bones']:expected[:3,3]+=delta
        expected[:3,3]+=np.array(layout_bone_deltas.get(name,(0,0,0)))
        check('shoulder_lowering_joint_'+name,rest_matrix_matches(name,expected))
    check('shoulder_lowering_keyframes_preserved',actions==shoulder_audit['actions_before'])
    check('shoulder_lowering_limits_preserved',limits==shoulder_audit['limits_before'])

if layout_audit:
    check('layout_keyframes_preserved',actions==layout_audit['actions_before'])
    check('layout_limits_preserved',limits==layout_audit['limits_before'])
    for name,values in layout_audit['mesh_vertices_before'].items():
        actual=np.array([v.co for v in bpy.data.objects[name].data.vertices])
        check('layout_exact_mesh_transform_'+name,np.max(abs(actual-with_layout_mesh_revision(name,values)))<1e-8)
    for name,values in layout_audit['bone_rest_matrices_before'].items():
        expected=np.array(values);expected[:3,3]+=np.array(layout_bone_deltas.get(name,(0,0,0)))
        check('layout_exact_bone_transform_'+name,rest_matrix_matches(name,expected))
    waist=rig.data.bones['spine_pitch'].head_local
    for side in ['L','R']:
        front=rig.data.bones[side+'_shoulder_pitch'].head_local
        rear=rig.data.bones[side+'_hip_pitch'].head_local
        target_front=.100 if offset_audit else (.090 if scene.get('shoulder_distance_revision') else .100)
        check('waist_to_shoulder_horizontal_'+str(round(target_front*1000))+'mm_'+side,
              abs(front.x-waist.x-target_front)<1e-8,(front.x-waist.x)*1000)
        target_rear=.060 if offset_audit else .050
        check('waist_to_hip_horizontal_'+str(round(target_rear*1000))+'mm_'+side,abs(waist.x-rear.x-target_rear)<1e-8,(waist.x-rear.x)*1000)
    for c in colliders:
        pivot=np.array(rig.data.bones[c['bone']].head_local)
        axis=np.array(c['axis_world_zero']);capsule_center=np.array(c['center_world_zero_m'])
        proximal=capsule_center-axis*c['cylinder_length_m']/2
        check('capsule_proximal_sphere_at_joint_'+c['name'],np.linalg.norm(proximal-pivot)<1e-8)
        local_mount_axis=pitch_matrix(c['bone'])[:3,:3].T@axis
        degrees=math.degrees(math.atan2(local_mount_axis[0],-local_mount_axis[2]));target=20. if 'hand' in c['name'] else 70.
        check('capsule_mount_angle_'+c['name'],abs(degrees-target)<1e-5 and abs(local_mount_axis[1])<1e-8,degrees)
        bone_frame=np.array(rig.data.bones[c['bone']].matrix_local)
        endpoints=np.array(c['endpoints_bone_m'])@bone_frame[:3,:3].T+bone_frame[:3,3]
        check('capsule_bone_endpoint_at_joint_'+c['name'],np.linalg.norm(endpoints[0]-pivot)<1e-8)
    for key in ['hand','foot']:
        a=next(c for c in colliders if c['visual_object']=='L_'+key+'_visual')
        b=next(c for c in colliders if c['visual_object']=='R_'+key+'_visual')
        check('capsule_left_right_mount_symmetry_'+key,
              np.linalg.norm(np.array(a['center_world_zero_m'])*[1,-1,1]-b['center_world_zero_m'])<1e-8 and
              np.linalg.norm(np.array(a['axis_world_zero'])*[1,-1,1]-b['axis_world_zero'])<1e-8)
    for g in layout_audit['guides']:
        vv=np.array([v.co for v in bpy.data.objects[g['object']].data.vertices],dtype=np.float64)
        expected=np.array([rig.data.bones[g[k]].head_local for k in ['from','to']],dtype=np.float64)
        err=float(np.max(abs(np.array([vv[:12].mean(0),vv[12:].mean(0)])-expected)))
        check('guide_endpoints_'+g['object'],err<1e-8,err)

if pitch_audit:
    from mount_collision import components,interference
    check('pitch_mount_keyframes_preserved',actions==pitch_audit['actions_before'])
    check('pitch_mount_limits_preserved',limits==pitch_audit['limits_before'])
    for name,values in pitch_audit['bone_rest_matrices_before'].items():
        expected=pitch_matrix(name)@np.array(values)
        check('pitch_mount_bone_'+name,rest_matrix_matches(name,expected,include_pitch_mount=False))
    for name,values in pitch_audit['mesh_vertices_before'].items():
        actual=np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float64)
        m=pitch_matrix(name,'mesh');expected=np.array(values)@m[:3,:3].T+m[:3,3]
        check('pitch_mount_mesh_'+name,np.max(abs(actual-expected))<1e-8)
    for side,sign in [('L',1),('R',-1)]:
        details=pitch_audit['sides'][side]
        matrix=np.array(rig.data.bones[side+'_shoulder_pitch'].matrix_local)
        yaw=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
        # Edit-bone endpoints use float32; translating a 14 mm axis can change
        # its reconstructed yaw by about 0.00001 degrees without a real rotation.
        check('pitch_mount_inward_5deg_'+side,abs(yaw+sign*5)<2e-5,yaw)
        before=np.array(pitch_audit['bone_rest_matrices_before'][side+'_shoulder_pitch'])
        outward=sign*(matrix[1,3]-before[1,3])
        expected_outward=.002+sign*offset_delta(side+'_shoulder_pitch')[1]
        check('pitch_mount_current_lateral_offset_'+side,abs(outward-expected_outward)<1e-8,outward)
        result=interference(components(bpy.data.objects[side+'_shoulder_roll_visual']),components(bpy.data.objects[side+'_shoulder_pitch_visual']))
        if offset_audit:
            latest=straight_audit or offset_audit
            check('pitch_roll_zero_collision_diagnostic_'+side,result['penetrating']==latest['shoulder_zero_clearance'][side]['penetrating'],result)
        else:check('pitch_roll_zero_mesh_interference_free_'+side,not result['penetrating'],result)
        vertices=np.array([v.co for v in bpy.data.objects[side+'_shoulder_pitch_visual'].data.vertices],dtype=np.float64)
        _,axis_basis=np.linalg.eigh(np.cov(vertices.T))
        check('pitch_axis_matches_named_cylinder_'+side,abs(np.dot(axis_basis[:,0],matrix[:3,1]))>1-1e-6)

if offset_audit:
    check('offset_revision_keyframes_preserved',actions==offset_audit['actions_before'])
    check('offset_revision_limits_preserved',limits==offset_audit['limits_before'])
    for name,values in offset_audit['bone_rest_matrices_before'].items():
        expected=np.array(values);expected[:3,3]+=offset_delta(name)
        check('offset_target_or_fixed_child_'+name,rest_matrix_matches(name,expected,include_pitch_mount=False))
    for name,values in offset_audit['mesh_vertices_before'].items():
        actual=np.array([v.co for v in bpy.data.objects[name].data.vertices],dtype=np.float64)
        check('offset_target_or_fixed_mesh_'+name,np.max(abs(actual-np.array(values)-offset_delta(name,'mesh')))<1e-8)
    check('hip_roll_not_shifted','hip_roll' not in offset_audit['bone_translations_m'] and 'R_hip_roll_mimic' not in offset_audit['bone_translations_m'])
    check('historical_waist_roll_forward_1mm',np.linalg.norm(np.array(offset_audit['bone_translations_m']['spine_roll'])-[.001,0,0])<1e-9)
    root=rig.pose.bones['ROOT']
    check('chest_roll_is_topmost_root',root.parent is None)
    check('chest_root_pivot_is_waist_pitch',(root.bone.head_local-rig.data.bones['spine_pitch'].head_local).length<1e-8)
    check('chest_root_roll_control_x',tuple(root.lock_rotation)==(False,True,True))
    body=bpy.data.objects[offset_audit['chest_root']['body_reference_object']]
    check('body_reference_geometry_preserved',np.max(abs(np.array([v.co for v in body.data.vertices])-np.array(offset_audit['chest_root']['body_vertices'])))<1e-8)
    body_transform_error=float(np.max(abs(np.array(body.matrix_world)-np.array(offset_audit['chest_root']['body_matrix_world']))))
    check('body_reference_transform_preserved',body_transform_error<1e-7,{'error':body_transform_error,'actual':[list(r) for r in body.matrix_world]})
    check('body_reference_uses_root_weights',all(len(v.groups)==1 and body.vertex_groups[v.groups[0].group].name=='ROOT' and abs(v.groups[0].weight-1)<1e-7 for v in body.data.vertices))

if straight_audit:
    check('straight_revision_keyframes_preserved',actions==straight_audit['actions_before'])
    check('straight_revision_limits_preserved',limits==straight_audit['limits_before'])
    for name,values in straight_audit['bone_rest_matrices_before'].items():
        expected=np.array(values);expected[:3,3]+=np.array(straight_audit['bone_translations_m'].get(name,[0,0,0]))
        check('straight_revision_bone_'+name,rest_matrix_matches(name,expected,include_pitch_mount=False))
    for name,values in straight_audit['mesh_vertices_before'].items():
        obj=bpy.data.objects[name]
        expected=np.array(values)+np.array(straight_audit['mesh_translations_m'].get(name,[0,0,0]))
        check('straight_revision_mesh_'+name,np.max(abs(np.array([v.co for v in obj.data.vertices])-expected))<1e-8)
        check('manual_object_transform_baked_'+name,np.max(abs(np.array(obj.matrix_world)-np.eye(4)))<1e-8)
    for name in ['L_shoulder_roll_visual','spine_roll_visual']:
        before=np.array(straight_audit['mesh_vertices_before'][name]);m=np.array(straight_audit['mesh_matrices_before'][name])
        expected=before@m[:3,:3].T+m[:3,3]
        check('manual_world_position_preserved_'+name,np.max(abs(worldverts(name)-expected))<1e-7)
    left=np.array(rig.data.bones['L_shoulder_roll'].head_local)
    right=np.array(rig.data.bones['R_shoulder_roll'].head_local)
    check('shoulder_roll_pivots_mirrored',np.linalg.norm(left*[1,-1,1]-right)<1e-8)
    for name in ['L_shoulder_roll','R_shoulder_roll']:
        total=np.array(straight_audit['bone_translations_m'][name])+np.array(straight_audit['manual_bone_translations_m'].get(name,[0,0,0]))
        check('manual_shoulder_roll_sync_8mm_'+name,np.linalg.norm(total-[.008,0,0])<1e-9)
    total=np.array(straight_audit['bone_translations_m']['spine_roll'])+np.array(straight_audit['manual_bone_translations_m'].get('spine_roll',[0,0,0]))
    check('manual_waist_roll_sync_minus5mm',np.linalg.norm(total-[-.005,0,0])<1e-9)

# Disable the animation only in memory so test values are not overwritten by it.
action=rig.animation_data.action; rig.animation_data.action=None
for pb in rig.pose.bones: pb.rotation_euler=(0,0,0)
update()
if offset_audit:
    names=[r['object'] for r in meshes]+[offset_audit['chest_root']['body_reference_object']]
    initial={n:center(n) for n in names};pivot=np.array(rig.data.bones['ROOT'].head_local)
    rig.pose.bones['ROOT'].rotation_euler.x=math.radians(10);update()
    rotation=np.array(Matrix.Rotation(math.radians(10),3,'X'))
    for n,c0 in initial.items():
        expected=rotation@(c0-pivot)+pivot
        check('chest_root_roll_moves_entire_body_'+n,np.linalg.norm(center(n)-expected)<2e-6)
    rig.pose.bones['ROOT'].rotation_euler.x=0;update()
before_hand=center('L_hand_visual'); before_motor=center('L_shoulder_pitch_visual')
setq('L_shoulder_pitch',20)
check('proximal_rotation_moves_distal_foot',np.linalg.norm(center('L_hand_visual')-before_hand)>.01)
check('shoulder_motor_stator_stays_on_parent_link',np.linalg.norm(center('L_shoulder_pitch_visual')-before_motor)<1e-6)
check('requested_rotation_evaluates_correctly',abs(effective_angle(rig,'L_shoulder_pitch')-math.radians(20))<1e-5)
setq('L_shoulder_pitch',170)
check('upper_limit_clamps_pose',abs(effective_angle(rig,'L_shoulder_pitch')-math.radians(120))<1e-5)
setq('L_shoulder_pitch',-170)
check('lower_limit_clamps_pose',abs(effective_angle(rig,'L_shoulder_pitch')-math.radians(-120))<1e-5)
setq('L_shoulder_pitch',0)
setq('hip_roll',12)
check('opposed_rear_roll_evaluates',abs(effective_angle(rig,'R_hip_roll_mimic')+math.radians(12))<1e-5)
c=rig.pose.bones['hip_roll'].constraints['Joint limits (temporary)']
c.min_y=math.radians(-18); c.max_y=math.radians(22); update()
d=rig.pose.bones['R_hip_roll_mimic'].constraints['Joint limits (temporary)']
check('dependent_limits_follow_edits',abs(d.min_y+math.radians(22))<1e-6 and abs(d.max_y-math.radians(18))<1e-6)
setq('hip_roll',0)
for row in rows:
    if not row['actuated']: continue
    name=row['name']; setq(name,10)
    check('actuator_'+name,abs(effective_angle(rig,name)-math.radians(10))<1e-5)
    setq(name,0)

pb=rig.pose.bones['L_knee_pitch']
pb.rotation_euler.y=math.radians(-20); pb.keyframe_insert(data_path='rotation_euler',index=1,frame=110)
pb.rotation_euler.y=math.radians(30); pb.keyframe_insert(data_path='rotation_euler',index=1,frame=120)
scene.frame_set(110); update(); q1=effective_angle(rig,pb.name)
scene.frame_set(120); update(); q2=effective_angle(rig,pb.name)
check('keyframes_replay',abs(q1-math.radians(-20))<1e-5 and abs(q2-math.radians(30))<1e-5)

usd=ROOT/'robot_authoring/exports/robot_visual.usdc'; stage=Usd.Stage.Open(str(usd))
check('usd_reopens',stage is not None)
check('usd_units',UsdGeom.GetStageMetersPerUnit(stage)==1. and UsdGeom.GetStageUpAxis(stage)=='Z')
skels=[UsdSkel.Skeleton(p) for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
check('usd_skeleton_has_19_bones',len(skels)==1 and len(skels[0].GetJointsAttr().Get())==19)
usdmeshes=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
check('usd_has_only_22_robot_meshes',len(usdmeshes)==22 and all('GUIDE' not in str(p.GetPath()) for p in usdmeshes))
for p in usdmeshes:
    b=UsdSkel.BindingAPI(p)
    check('usd_skin_weights_'+p.GetName(),bool(b.GetJointWeightsPrimvar().Get()) and abs(min(b.GetJointWeightsPrimvar().Get())-1)<1e-6)
check('usd_animation_present',any(p.IsA(UsdSkel.Animation) for p in stage.Traverse()))
result={'passed':all(checks.values()),'checks':checks,'measurements':measurements,
        'source_sha256':source['sha256'],'rig_sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        'usd_sha256':hashlib.sha256(usd.read_bytes()).hexdigest(),
        'scope':'Static geometry + pose evaluation + constraints/keyframes + USD skeleton; no dynamics/training',
        'saved_test_mutations':False,
        'collision_diagnostics':(straight_audit or offset_audit)['shoulder_zero_clearance'] if offset_audit else None}
(ROOT/'robot_authoring/validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('VALIDATION_PASSED',len(checks),'checks','shape_error_m',max_shape_error,'bind_error_m',max_bind_error)
