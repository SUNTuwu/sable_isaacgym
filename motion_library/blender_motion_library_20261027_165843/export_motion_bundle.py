"""Run in background Blender on a delivery copy; never change the live editor.

Exports the complete current evaluated animation, then bakes the delivery copy.
The bundle is a kinematic reference library, not a bank of teacher actions.
"""
import bpy, json, sys, runpy, time
from pathlib import Path
import numpy as np
from mathutils import Matrix

BUNDLE=Path(sys.argv[sys.argv.index('--')+1]).resolve()
EDITOR=BUNDLE/'editor'
NS=runpy.run_path(str(EDITOR/'scripts/sable_motion_editor.py'),run_name='sable_motion_editor')
NS['register']();bpy.app.driver_namespace['sable_editor']=NS
MODEL=NS['MODEL'];scene=bpy.context.scene;rig=bpy.data.objects['SableRobot'];body=NS['control']('BODY','BODY_EDIT')
seed=bpy.data.objects['SABLE_SEED'];manifest=NS['MANIFEST'];fps=scene.render.fps/scene.render.fps_base
assert fps==50 and len(MODEL.names)==15
MOTIONS=BUNDLE/'motions';MOTIONS.mkdir(exist_ok=True)
POLICY_ORDER=['spine_roll','hip_roll','L_shoulder_roll','L_shoulder_pitch','L_elbow_pitch','L_hip_pitch','L_knee_pitch',
              'R_shoulder_roll','R_shoulder_pitch','R_elbow_pitch','R_hip_pitch','R_knee_pitch','L_ankle_pitch','R_ankle_pitch','spine_pitch']
assert set(POLICY_ORDER)==set(MODEL.names)
mapping=[MODEL.names.index(n) for n in POLICY_ORDER]
catalog=[];prepared=[];original_frame=scene.frame_current

def angular_velocity(rot,dt):
    result=[]
    for i in range(len(rot)):
        a=max(0,i-1);b=min(len(rot)-1,i+1)
        q=Matrix((rot[b]@rot[a].T).tolist()).to_quaternion()
        if q.w<0:q.negate()
        axis,angle=q.to_axis_angle();result.append(np.array(axis)*angle/((b-a)*dt))
    return np.array(result)

for clip in manifest['clips']:
    rows=[]
    for frame in range(clip['start'],clip['end']+1):
        scene.frame_set(frame);bpy.context.view_layer.update();NS['solve_now'](True);bpy.context.view_layer.update()
        q=np.array([rig.pose.bones[n].rotation_euler.y for n in MODEL.names])
        root=np.array(body.matrix_world);target=np.array([o.matrix_world.translation for o in NS['controls']()])
        actual=MODEL.fk(q,root)[0];surface=MODEL.surface_z(q,root)
        rows.append((q,root,target,actual,surface))
    q,root,target,actual,surface=[np.array([r[i] for r in rows]) for i in range(5)]
    assert all(np.isfinite(x).all() for x in (q,root,target,actual,surface))
    quat=np.array([list(Matrix(r[:3,:3].tolist()).to_quaternion()) for r in root])
    for i in range(1,len(quat)):
        if np.dot(quat[i-1],quat[i])<0:quat[i]*=-1
    dt=1/fps;qvel=np.gradient(q,dt,axis=0,edge_order=2)
    velocity=np.gradient(root[:,:3,3],dt,axis=0,edge_order=2)
    omega=angular_velocity(root[:,:3,:3],dt)
    t=np.arange(len(q))*dt
    payload=dict(time_s=t,joint_position=q,joint_velocity=qvel,
                 root_transform_world=root,root_position_world=root[:,:3,3],root_quaternion_wxyz=quat,
                 root_linear_velocity_world=velocity,root_angular_velocity_world=omega,
                 toe_target_world=target,toe_actual_world=actual,foot_capsule_min_z=surface,
                 joint_position_policy_order=q[:,mapping],joint_velocity_policy_order=qvel[:,mapping])
    np.savez_compressed(MOTIONS/(clip['id']+'.npz'),**payload)
    error=np.linalg.norm(target-actual,axis=2)*1000
    limits=np.maximum(MODEL.limits[:,0]-q,0)+np.maximum(q-MODEL.limits[:,1],0)
    yaw=np.unwrap(np.arctan2(root[:,1,0],root[:,0,0]));delta=yaw[-1]-yaw[0]
    cs,sn=np.cos(delta),np.sin(delta);cycle=np.eye(4);cycle[:3,:3]=[[cs,-sn,0],[sn,cs,0],[0,0,1]]
    cycle[:2,3]=root[-1,:2,3]-(cycle[:3,:3]@root[0,:3,3])[:2]
    seam=np.linalg.norm(actual[-1]-(actual[0]@cycle[:3,:3].T+cycle[:3,3]),axis=1)*1000
    quality=dict(max_ik_error_mm=float(error.max()),minimum_capsule_z_mm=float(surface.min()*1000),
                 ground_penetration_sample_fraction=float(np.mean(surface<-.0001)),
                 max_joint_limit_violation_deg=float(np.rad2deg(limits.max())),
                 max_joint_pose_seam_deg=float(np.rad2deg(np.max(abs(q[-1]-q[0])))),
                 toe_pose_seam_mm=seam.tolist(),
                 max_joint_velocity_seam_rad_s=float(np.max(abs(qvel[-1]-qvel[0]))),
                 max_authored_joint_velocity_rad_s=float(np.max(abs(qvel))),
                 root_displacement_world_m=(root[-1,:3,3]-root[0,:3,3]).tolist(),
                 root_yaw_change_deg=float(np.rad2deg(delta)))
    warnings=[]
    if quality['minimum_capsule_z_mm']<-.1:warnings.append('foot_capsule_ground_penetration')
    if quality['max_joint_pose_seam_deg']>.1:warnings.append('joint_pose_seam_not_closed')
    if quality['max_joint_velocity_seam_rad_s']>.2:warnings.append('joint_velocity_seam_not_closed')
    if quality['max_ik_error_mm']>.5:warnings.append('toe_target_projection_error')
    if quality['max_joint_limit_violation_deg']>.01:warnings.append('authoring_joint_limit_violation')
    quality['warnings']=warnings
    meta=dict(schema='sable-authored-motion/1',id=clip['id'],title=clip['title'],fps=fps,samples=len(q),
              duration_s=float(t[-1]),joint_names=MODEL.names,policy_joint_names=POLICY_ORDER,
              policy_from_motion_indices=mapping,foot_order=['FL','FR','HL','HR'],
              units={'length':'m','angle':'rad','time':'s'},quaternion_order='wxyz',
              axes='+X forward, +Y left, +Z up',source_clip=clip,
              toe_definition='Physical distal capsule tip, not capsule center or ground contact point',
              frame_interval=[clip['start'],clip['end']],cycle_transform_world=cycle.tolist(),quality=quality,
              training_status='reference_only_requires_preflight',
              contact_labels_available=False,teacher_actions_available=False,torques_available=False,
              use='Fit physically executable specialist teacher from scratch; do not use these q values as verified PD actions.',
              endpoint_note='Includes both time endpoints. Do not concatenate until pose/velocity seams and cycle transform are handled.')
    (MOTIONS/(clip['id']+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    catalog.append(meta);prepared.append((clip,q,root,target,quality))
    print('EXPORTED',clip['id'],json.dumps(quality),flush=True)

# Bake only the isolated delivery file, retaining its sparse visible controls.
scene['sable_live_ik']=False
bias_props={'spine_pitch':'waist_pitch_offset_deg','spine_roll':'waist_roll_preference_deg',
            'L_ankle_pitch':'L_ankle_offset_deg','R_ankle_pitch':'R_ankle_offset_deg'}
for clip,qs,roots,targets,quality in prepared:
    for i,frame in enumerate(list(range(clip['start'],clip['end']+1))+[clip['hold_end']]):
        k=min(i,len(qs)-1);q=qs[k];root=roots[k]
        scene.frame_set(frame);bpy.context.view_layer.update();NS['apply_pose'](q,root)
        for n in MODEL.names:rig.pose.bones[n].keyframe_insert('rotation_euler',frame=frame,group=n)
        for path in ('location','rotation_quaternion'):rig.pose.bones['ROOT'].keyframe_insert(path,frame=frame,group='ROOT')
        for n,v in zip(MODEL.names,q):
            offset=np.deg2rad(body.get(bias_props[n],0)) if n in bias_props else 0.
            seed[n]=float(v-offset);seed.keyframe_insert('["'+n+'"]',frame=frame)
    clip['delivery_quality']=quality
for o in (rig,seed):
    for f in NS['action_curves'](o):
        for k in f.keyframe_points:
            k.interpolation='CONSTANT' if any(abs(k.co.x-c['end'])<1e-4 or abs(k.co.x-c['hold_end'])<1e-4 for c in manifest['clips']) else 'LINEAR'
checks=[]
for clip,qs,roots,targets,quality in prepared:
    mq=mr=0.
    for i,f in enumerate(range(clip['start'],clip['end']+1)):
        scene.frame_set(f);bpy.context.view_layer.update()
        actual=np.array([rig.pose.bones[n].rotation_euler.y for n in MODEL.names])
        mq=max(mq,float(np.max(abs(actual-qs[i]))));mr=max(mr,float(np.max(abs(np.array(rig.pose.bones['ROOT'].matrix)-roots[i]))))
    assert mq<1e-5 and mr<2e-6
    checks.append({'clip':clip['id'],'max_q_error_rad':mq,'max_root_matrix_error':mr})
scene.frame_set(original_frame);scene['sable_live_ik']=True;NS['solve_now'](True);bpy.context.view_layer.update()
(EDITOR/'motion_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
scene['sable_motion_manifest']=json.dumps(manifest,ensure_ascii=False)
bpy.ops.wm.save_as_mainfile(filepath=str(EDITOR/'sable_motion_edit.blend'))
(BUNDLE/'motion_catalog.json').write_text(json.dumps({'schema':'sable-motion-library/1','created_local':time.strftime('%Y-%m-%dT%H:%M:%S'),
    'clip_count':len(catalog),'clips':catalog,'initialization_policy':'New specialist teachers and distilled student from scratch. Do not initialize from latest S3 checkpoints.'},ensure_ascii=False,indent=2),encoding='utf8')
(BUNDLE/'quality_report.json').write_text(json.dumps({'source':'Live editor snapshot, evaluated in isolated Blender',
    'training_ready':False,'kinematic_export_verified':True,'baked_playback_checks':checks,
    'clips':[{'id':c['id'],**c['quality']} for c in catalog]},ensure_ascii=False,indent=2),encoding='utf8')
print('BUNDLE_COMPLETE',str(BUNDLE),flush=True)
