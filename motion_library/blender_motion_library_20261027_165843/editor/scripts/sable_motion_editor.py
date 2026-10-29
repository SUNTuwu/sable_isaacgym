"""Project-local live constrained IK editor. Run via launch_editor.ps1 or Text Editor.
Keeps original rigid mesh bindings, hinge limits and opposed rear-hip constraint.
"""
import bpy, json, sys, time
import numpy as np
from pathlib import Path
from mathutils import Matrix, Vector
from bpy.app.handlers import persistent

BASE=Path(bpy.data.filepath).parent
if not (BASE/'source/rig_snapshot.json').exists():
    BASE=Path(__file__).resolve().parents[1]
if str(BASE/'scripts') not in sys.path:sys.path.insert(0,str(BASE/'scripts'))
from sable_kinematics import load_model, LEGS, ANCHORS, ENDS
MODEL=load_model(BASE/'source')
MANIFEST=json.loads((BASE/'motion_manifest.json').read_text(encoding='utf8'))
_busy=False;_signature=None;_last_frame=None;_last_solution=None;_last_root=None

def control(role, fallback):
    scene=bpy.context.scene
    o=scene.get('sable_control_'+role)
    if isinstance(o,bpy.types.Object):return o
    o=next((obj for obj in scene.objects if obj.get('sable_control_role')==role),None)
    return o if o else bpy.data.objects.get(fallback)

def controls():return [control(leg,'IK_'+leg) for leg in LEGS]

def is_handle(o):return o in controls() or o==control('BODY','BODY_EDIT')

def apply_pose(q,root):
    rig=bpy.data.objects['SableRobot'];pb=rig.pose.bones['ROOT']
    pb.matrix_basis=pb.bone.matrix_local.inverted()@Matrix(root.tolist())
    for n,a in zip(MODEL.names,q):rig.pose.bones[n].rotation_euler=(0,float(a),0)
    # The original COPY_ROTATION constraint imposes the passive -hip_roll coordinate.

def solve_now(force=False):
    global _busy,_signature,_last_frame,_last_solution,_last_root
    scene=bpy.context.scene
    if _busy or not scene.get('sable_live_ik',False) or 'SABLE_SEED' not in bpy.data.objects:return
    handles=controls();body=control('BODY','BODY_EDIT');seedobj=bpy.data.objects['SABLE_SEED']
    if body is None or any(o is None for o in handles):return
    frame=scene.frame_current
    target=np.array([list(o.matrix_world.translation) for o in handles]);root=np.array(body.matrix_world)
    q=np.array([seedobj[n] for n in MODEL.names],float)
    q[MODEL.index['spine_pitch']]+=np.deg2rad(body.get('waist_pitch_offset_deg',0.))
    q[MODEL.index['spine_roll']]+=np.deg2rad(body.get('waist_roll_preference_deg',0.))
    for side in ('L','R'):q[MODEL.index[side+'_ankle_pitch']]+=np.deg2rad(body.get(side+'_ankle_offset_deg',0.))
    signature=(frame,tuple(np.round(target.ravel(),8)),tuple(np.round(root.ravel(),8)),tuple(np.round(q,8)))
    if not force and signature==_signature:
        # Inserting a key or editing a curve can re-evaluate the baked armature
        # without changing the targets. Reapply IK if that evaluation overwrote it.
        rig=bpy.data.objects['SableRobot']
        actual_q=np.array([rig.pose.bones[n].rotation_euler.y for n in MODEL.names])
        actual_root=np.array(rig.pose.bones['ROOT'].matrix)
        if _last_solution is not None and np.max(abs(actual_q-_last_solution))<1e-6 and np.max(abs(actual_root-_last_root))<1e-6:return
    _busy=True
    try:
        # Unedited policy motion stays byte-for-byte at its recorded joint coordinates.
        err=np.linalg.norm(MODEL.fk(q,root)[0]-target,axis=1)
        if np.max(err)>.00002:q,err=MODEL.solve(target,root,q,q,iterations=40)
        apply_pose(q,root)
        scene['sable_max_error_mm']=float(max(err)*1000)
        for i,o in enumerate(handles):o['target_error_mm']=float(err[i]*1000)
        _,mats,_=MODEL.fk(q,root)
        for i,leg in enumerate(LEGS):
            anchor=bpy.data.objects.get('ANCHOR_'+leg)
            if anchor:anchor.location=mats[ANCHORS[i]][:3,3]
            actual=bpy.data.objects.get('ACTUAL_'+leg)
            if actual:actual.location=MODEL.fk(q,root)[0][i]
        _signature=signature;_last_frame=frame;_last_solution=q.copy();_last_root=root.copy()
    finally:_busy=False

@persistent
def sable_frame_changed(scene,*args):
    # Animation is already evaluated here; restore edited IK before drawing the frame.
    solve_now(True)

def tick():
    try:solve_now()
    except (ReferenceError,KeyError):pass
    except Exception as e:print('Sable IK:',e)
    return .06

def insert_handle_keys():
    for o in bpy.context.selected_objects:
        if is_handle(o):
            o.keyframe_insert(data_path='location',frame=bpy.context.scene.frame_current)
            if o==control('BODY','BODY_EDIT'):o.keyframe_insert(data_path='rotation_euler',frame=bpy.context.scene.frame_current)

class SABLE_OT_key(bpy.types.Operator):
    bl_idname='sable.key_handles';bl_label='记录选中控制器 / Key selected'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        context.view_layer.update();insert_handle_keys();context.view_layer.update();solve_now(True);context.view_layer.update();return {'FINISHED'}

class SABLE_OT_reset(bpy.types.Operator):
    bl_idname='sable.reset_handles';bl_label='恢复选中脚尖的原轨迹 / Reset selected'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        for o in context.selected_objects:
            if o in controls():
                leg=LEGS[controls().index(o)];path=bpy.data.objects.get('PATH_'+leg)
                if path:o.location=path.matrix_world.translation if o.parent is None else (0,0,0)
            elif o==control('BODY','BODY_EDIT'):o.location=(0,0,0);o.rotation_euler=(0,0,0)
        context.view_layer.update();solve_now(True);insert_handle_keys();return {'FINISHED'}

class SABLE_OT_jump(bpy.types.Operator):
    bl_idname='sable.jump_clip';bl_label='选择片段'
    clip: bpy.props.IntProperty(default=0)
    def execute(self,context):
        c=MANIFEST['clips'][self.clip];s=context.scene
        s.frame_preview_start=c['start'];s.frame_preview_end=c['end']-1;s.use_preview_range=True;s.frame_set(c['start'])
        return {'FINISHED'}

def action_curves(obj):
    ad=obj.animation_data
    if not ad or not ad.action:return []
    action=ad.action;slot=ad.action_slot
    result=[]
    if hasattr(action,'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                bag=strip.channelbag(slot) if hasattr(strip,'channelbag') else None
                if bag:result.extend(list(bag.fcurves))
    elif hasattr(action,'fcurves'):result.extend(list(action.fcurves))
    return result

class SABLE_OT_bake(bpy.types.Operator):
    bl_idname='sable.bake_edit';bl_label='烘焙当前片段并导出 / Bake + export clip'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        s=context.scene;frame=s.frame_current
        c=next((c for c in MANIFEST['clips'] if c['start']<=frame<=c['end']),None)
        if not c:self.report({'WARNING'},'请先选择动作片段');return {'CANCELLED'}
        rig=bpy.data.objects['SableRobot'];rows=[]
        for f in range(c['start'],c['end']+1):
            s.frame_set(f);context.view_layer.update();solve_now(True);context.view_layer.update()
            q=np.array([rig.pose.bones[n].rotation_euler.y for n in MODEL.names])
            root=np.array(rig.pose.bones['ROOT'].matrix);targets=np.array([o.matrix_world.translation for o in controls()])
            feet,_,_=MODEL.fk(q,root)
            for n in MODEL.names:rig.pose.bones[n].keyframe_insert('rotation_euler',frame=f,group=n)
            for path in ['location','rotation_quaternion']:rig.pose.bones['ROOT'].keyframe_insert(path,frame=f,group='ROOT')
            rows.append({'frame':f,'t':(f-c['start'])/s.render.fps,'q':q.tolist(),'root_matrix':root.tolist(),'target':targets.tolist(),'actual_foot':feet.tolist(),'error_mm':(np.linalg.norm(targets-feet,axis=1)*1000).tolist()})
        for fc in action_curves(rig):
            for k in fc.keyframe_points:k.interpolation='LINEAR'
        out=BASE/'exports';out.mkdir(exist_ok=True)
        path=out/(c['id']+'_edited.json')
        path.write_text(json.dumps({'joint_names':MODEL.names,'foot_order':list(LEGS),'source':c,'frames':rows,'scope':'Kinematic authoring export, not a validated hardware trajectory'},ensure_ascii=False,indent=2),encoding='utf8')
        s.frame_set(frame);self.report({'INFO'},str(path));return {'FINISHED'}

class SABLE_PT_editor(bpy.types.Panel):
    bl_label='Sable · 足点 IK 编辑';bl_idname='SABLE_PT_editor';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Sable IK'
    def draw(self,context):
        l=self.layout;s=context.scene
        l.label(text='50 fps · 每段一个周期 · 间隔为停顿/切换')
        for i,c in enumerate(MANIFEST['clips']):
            op=l.operator('sable.jump_clip',text=f"{c['title']}  {c['start']}–{c['end']}");op.clip=i
        l.prop(s,'["sable_live_ik"]',text='实时约束 IK')
        l.label(text=f"最大足点偏差: {s.get('sable_max_error_mm',0):.2f} mm")
        l.label(text='直接在 IK_FL/FR/HL/HR 上打位置关键帧')
        l.operator('sable.key_handles');l.operator('sable.reset_handles')
        l.separator();body=control('BODY','BODY_EDIT')
        if body:
            for k,title in [('waist_pitch_offset_deg','腰 pitch 偏移'),('waist_roll_preference_deg','腰 roll 偏好'),('L_ankle_offset_deg','左踝偏好'),('R_ankle_offset_deg','右踝偏好')]:l.prop(body,'["'+k+'"]',text=title)
        l.operator('sable.bake_edit')
        l.label(text='控制点为胶囊最远端脚尖，随踝关节旋转')
        l.label(text='保留原电机关节轴、限位及后髋反向联动')

CLASSES=(SABLE_OT_key,SABLE_OT_reset,SABLE_OT_jump,SABLE_OT_bake,SABLE_PT_editor)
def register():
    for leg,o in zip(LEGS,controls()):
        if o:
            o['sable_control_role']=leg;bpy.context.scene['sable_control_'+leg]=o
    body=control('BODY','BODY_EDIT')
    if body:body['sable_control_role']='BODY';bpy.context.scene['sable_control_BODY']=body
    for cls in CLASSES:
        old=getattr(bpy.types,cls.__name__,None)
        if old:
            try:bpy.utils.unregister_class(old)
            except Exception:pass
        bpy.utils.register_class(cls)
    for handler in list(bpy.app.handlers.frame_change_post):
        if getattr(handler,'__name__','')=='sable_frame_changed' or (getattr(handler,'__name__','')=='frame_changed' and getattr(handler,'__module__','') in ('__main__','sable_motion_editor')):bpy.app.handlers.frame_change_post.remove(handler)
    bpy.app.handlers.frame_change_post.append(sable_frame_changed)
    bpy.context.scene['sable_live_ik']=True
    old=bpy.app.driver_namespace.get('sable_ik_tick')
    if old and bpy.app.timers.is_registered(old):bpy.app.timers.unregister(old)
    bpy.app.driver_namespace['sable_ik_tick']=tick
    if not bpy.app.timers.is_registered(tick):bpy.app.timers.register(tick,first_interval=.2,persistent=True)
    solve_now(True)

if __name__=='__main__':register()
