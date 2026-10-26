"""Export current Blender rig constraints, sampled poses and visual USD."""
import bpy
import hashlib
import json
import math
import sys
from pathlib import Path
from mathutils import Matrix, Vector

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'robot_authoring'))
from capsule_geometry import export_collision_stage

def effective_angle(rig, name):
    pb=rig.pose.bones[name]; bone=pb.bone
    if pb.parent:
        rest=bone.parent.matrix_local.inverted()@bone.matrix_local
        posed=pb.parent.matrix.inverted()@pb.matrix
    else:
        rest=bone.matrix_local; posed=pb.matrix
    return (rest.inverted()@posed).to_euler('XYZ').y

def export_robot(output_dir=None):
    scene=bpy.context.scene; rig=bpy.data.objects.get('SableRobot')
    if not rig or scene.get('robot_authoring_schema')!='sable-rig/1':
        raise RuntimeError('Open the authored sable_rig.blend first')
    bpy.context.view_layer.update()
    out=Path(output_dir) if output_dir else ROOT/'robot_authoring'/'exports'; out.mkdir(parents=True,exist_ok=True)
    rows=json.loads(rig['joint_table_json']); meshes=json.loads(rig['mesh_table_json'])
    was_mode=rig.mode; was_frame=scene.frame_current
    was_selected=list(bpy.context.selected_objects); was_active=bpy.context.view_layer.objects.active
    if bpy.context.object and bpy.context.object.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    table=[]
    for r in rows:
        pb=rig.pose.bones[r['name']]; b=pb.bone
        c=next((c for c in pb.constraints if c.type=='LIMIT_ROTATION'),None)
        rr=dict(r)
        rr['limits_rad']=None if c is None else [float(c.min_y),float(c.max_y)]
        rr['limits_deg']=None if c is None else [math.degrees(c.min_y),math.degrees(c.max_y)]
        rr['zero_position_world_m']=list(rig.matrix_world@b.head_local)
        rr['zero_axis_world']=list((rig.matrix_world.to_3x3()@b.matrix_local.to_3x3()@Vector((0,1,0))).normalized())
        rr['rest_matrix_armature']=[list(x) for x in b.matrix_local]
        rr['rest_matrix_parent']=[list(x) for x in (b.parent.matrix_local.inverted()@b.matrix_local if b.parent else b.matrix_local)]
        rr['joint_local_axis']=[0,1,0]
        rr['mass_kg']=None; rr['inertia_kg_m2']=None
        rr['encoder_zero_offset_rad']=None
        rr['limit_status']='temporary, awaiting user hardware range confirmation'
        if r['name']=='R_hip_roll_mimic': rr['mimic']={'joint':'hip_roll','multiplier':-1.,'offset_rad':0.}
        table.append(rr)
    colliders=[json.loads(bpy.data.objects[r['object']]['collision_description_json'])
               for r in meshes if bpy.data.objects[r['object']].get('collision_description_json')]
    poses=[]; collision_frames=[]
    for frame in range(scene.frame_start,scene.frame_end+1):
        scene.frame_set(frame); bpy.context.view_layer.update()
        q={r['name']:effective_angle(rig,r['name']) for r in rows if r['kind']=='revolute'}
        if abs(q['R_hip_roll_mimic']+q['hip_roll'])>1e-5:
            raise RuntimeError('Rear-roll coupling was edited/broken')
        for r in table:
            if r['kind']=='revolute' and not r['limits_rad'][0]-1e-5<=q[r['name']]<=r['limits_rad'][1]+1e-5:
                raise RuntimeError('Evaluated pose violates limits: '+r['name'])
        poses.append({'frame':frame,'time_s':(frame-scene.frame_start)/(scene.render.fps/scene.render.fps_base),
                      'joint_angles_rad':q,'joint_angles_deg':{k:math.degrees(v) for k,v in q.items()},
                      'root_matrix_world':[list(x) for x in rig.matrix_world@rig.pose.bones['ROOT'].matrix]})
        collision_frames.append({'frame':frame,'matrices':{c['bone']:[list(x) for x in
            rig.matrix_world@rig.pose.bones[c['bone']].matrix] for c in colliders}})
    collision_usd=export_collision_stage(out/'foot_colliders.usda',rig,colliders,collision_frames,
        scene.render.fps/scene.render.fps_base,scene.frame_start,scene.frame_end) if colliders else None
    scene.frame_set(was_frame)
    for obj in bpy.context.selected_objects: obj.select_set(False)
    rig.select_set(True)
    for r in meshes: bpy.data.objects[r['object']].select_set(True)
    bpy.context.view_layer.objects.active=rig
    usd=out/'robot_visual.usdc'
    bpy.ops.wm.usd_export(filepath=str(usd),check_existing=False,selected_objects_only=True,
        export_animation=True,export_armatures=True,only_deform_bones=True,
        export_materials=True,export_custom_properties=True,relative_paths=True,
        root_prim_path='/Sable',export_lights=False,export_cameras=False,
        export_curves=False,export_hair=False,export_volumes=False,export_points=False,
        evaluation_mode='RENDER',convert_orientation=False)
    from pxr import Usd, UsdGeom, UsdSkel
    stage=Usd.Stage.Open(str(usd))
    if not stage: raise RuntimeError('Exported USD cannot be reopened')
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.GetRootLayer().Save()
    skeletons=[p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    usd_meshes=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(skeletons)!=1 or len(usd_meshes)!=len(meshes):
        raise RuntimeError('Unexpected USD skeleton/mesh count')
    manifest={'schema':'sable-robot-authoring/1','source_blend':scene.get('robot_geometry_source_file','blender/draft.blend'),
        'source_sha256':scene['robot_source_sha256'],'authored_blend':'blender/sable_rig.blend',
        'pose_reference_blend':scene.get('robot_pose_reference_file'),
        'pose_reference_sha256':scene.get('robot_pose_reference_sha256'),
        'units':{'length':'meter','angle':'radian','world_up':'Z','forward':'+X','left':'+Y'},
        'export_scope':'visual USD skeleton/animation, terminal capsule collision USD and joint metadata; not a complete physics articulation',
        'ready_for_training':False,'pending':['confirmed limits','mass and inertia','remaining body/link collisions and collision filtering','actuator torque-speed envelope','simulator importer and validation'],
        'independent_actuator_count':sum(r['actuated'] for r in rows),'joints':table,'meshes':meshes,
        'source_allowlist':json.loads(scene['source_allowlist_json']),
        'assembly_revision':scene.get('waist_alignment_revision','initial straight-limb rig'),
        'waist_rearward_offset_m':scene.get('waist_rearward_offset_m',0.),
        'hip_roll_lowering_revision':scene.get('hip_roll_lowering_revision','original hip-roll mount height'),
        'hip_roll_lowering_m':scene.get('hip_roll_lowering_m',0.),
        'shoulder_roll_lowering_revision':scene.get('shoulder_roll_lowering_revision','waist-plane shoulder-roll mounts'),
        'shoulder_roll_lowering_m':scene.get('shoulder_roll_lowering_m',0.),
        'layout_revision':scene.get('layout_revision','prior source torso layout'),
        'motor_offset_revision':scene.get('motor_offset_revision'),
        'straight_limb_revision':scene.get('straight_limb_revision'),
        'straight_limb_deltas_world_m':json.loads(scene.get('straight_limb_deltas_json','{}')),
        'manual_roll_sync':json.loads(scene.get('manual_roll_sync_json','{}')),
        'motor_offsets_world_m':json.loads(scene.get('motor_offset_json','{}')),
        'chest_root_control':{'revision':scene.get('chest_root_roll_revision'),'bone':'ROOT',
            'parent':None,'roll_axis':'local X / world +X at zero','physical_actuators_added':0},
        'shoulder_pitch_mount_revision':scene.get('shoulder_pitch_mount_revision','parallel front pitch installation'),
        'shoulder_pitch_mounting':json.loads(scene.get('shoulder_pitch_mount_json','{}')),
        'fixed_terminal_mounting':{'wrist_deg':scene.get('wrist_mount_deg'),
            'ankle_deg':None if scene.get('powered_ankle_revision') else scene.get('ankle_mount_deg'),'reference':scene.get('terminal_mount_angle_reference'),
            'applied_to':'wrist fixed capsule geometry; ankle is actuated when powered_ankle is present'},
        'powered_ankle':json.loads(scene.get('powered_ankle_json','null')),
        'torso_horizontal_measurements_m':{side:{
            'waist_to_shoulder':rig.data.bones[side+'_shoulder_pitch'].head_local.x-rig.data.bones['spine_pitch'].head_local.x,
            'waist_to_hip':rig.data.bones['spine_pitch'].head_local.x-rig.data.bones[side+'_hip_pitch'].head_local.x}
            for side in ['L','R']},
        'upper_limb_length_revision':scene.get('upper_limb_length_revision','original source center distances'),
        'upper_limb_measurements':{
            side+'_'+kind:{'center_distance_m':(rig.matrix_world.to_3x3()@
                (rig.data.bones[side+'_'+a].head_local-rig.data.bones[side+'_'+b].head_local)).length,
                'distal_height_world_m':(rig.matrix_world@rig.data.bones[side+'_'+b].head_local).z}
            for side in ['L','R'] for kind,a,b in [('upper_arm','shoulder_pitch','elbow_pitch'),('thigh','hip_pitch','knee_pitch')]},
        'terminal_geometry_revision':scene.get('terminal_capsule_revision','original terminal boxes'),
        'hind_assembly_revision':scene.get('hind_assembly_revision','original knee motor layout and shank length'),
        'knee_motor_mounting':json.loads(scene.get('knee_motor_mounting_json','{}')),
        'elbow_motor_mount_revision':scene.get('elbow_motor_mount_revision','original front-side elbow layout'),
        'elbow_motor_mounting':json.loads(scene.get('elbow_motor_mounting_json','{}')),
        'lower_limb_measurements':{
            side+'_'+kind:{'center_distance_m':(rig.matrix_world.to_3x3()@
                (rig.data.bones[side+'_'+a].head_local-rig.data.bones[side+'_'+('ankle_pitch' if b=='ankle_fixed' and scene.get('powered_ankle_revision') else b)].head_local)).length}
            for side in ['L','R'] for kind,a,b in [('forearm','elbow_pitch','wrist_fixed'),('shank','knee_pitch','ankle_fixed')]},
        'collision_shapes':colliders,'collision_usd':collision_usd,
        'usd':{'path':'robot_visual.usdc','sha256':hashlib.sha256(usd.read_bytes()).hexdigest(),
               'skeletons':len(skeletons),'mesh_count':len(usd_meshes),'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),
               'up_axis':str(UsdGeom.GetStageUpAxis(stage))},
        'notes':['Do not treat the mirrored rear-roll output as an independent actuator.',
                 'Guide rods are display-only and excluded from robot export.',
                 'The source was reassembled rigidly into a new modeling zero; encoder offset is unknown.']}
    (out/'robot_description.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'poses.json').write_text(json.dumps({'schema':'sable-rig-poses/1','fps':scene.render.fps/scene.render.fps_base,
                                            'action':rig.animation_data.action.name if rig.animation_data else None,
                                            'poses':poses},ensure_ascii=False,indent=2),encoding='utf8')
    for obj in bpy.context.selected_objects: obj.select_set(False)
    for obj in was_selected: obj.select_set(True)
    bpy.context.view_layer.objects.active=was_active
    if was_mode!='OBJECT':
        bpy.context.view_layer.objects.active=rig; bpy.ops.object.mode_set(mode=was_mode)
    print('EXPORTED',len(table),'bones',len(meshes),'meshes',len(poses),'pose frames',usd)
    return manifest

if __name__=='__main__': export_robot()
