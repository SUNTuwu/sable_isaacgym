import bpy,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from export_description import effective_angle
rig=bpy.data.objects['SableRobot'];s=bpy.context.scene
frames={}
for frame in [1,5]:
    s.frame_set(frame);bpy.context.view_layer.update()
    frames[str(frame)]={p.name:{'euler_deg':[math.degrees(x) for x in p.rotation_euler],
        'effective_y_deg':math.degrees(effective_angle(rig,p.name)),
        'matrix_world':[list(r) for r in rig.matrix_world@p.matrix],
        'limits':[{'name':c.name,'type':c.type,'space':c.owner_space,
            'min_deg':[math.degrees(getattr(c,'min_'+a,0)) for a in 'xyz'],
            'max_deg':[math.degrees(getattr(c,'max_'+a,0)) for a in 'xyz'],
            'enabled':[getattr(c,'use_limit_'+a,False) for a in 'xyz']} for c in p.constraints]}
        for p in rig.pose.bones}
actions=[{'name':a.name,'curves':[{'path':f.data_path,'index':f.array_index,
    'keys':[[k.co[0],k.co[1]] for k in f.keyframe_points]} for l in a.layers for st in l.strips for b in st.channelbags for f in b.fcurves]} for a in bpy.data.actions]
result={'frame_data':frames,'actions':actions,'scene_props':{k:s[k] for k in s.keys() if isinstance(s[k],(str,int,float))}}
(P/'inspection/user_stand_frame5.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for name,data in frames['5'].items():print(name,json.dumps({'q_deg':data['euler_deg'],'effective_y_deg':data['effective_y_deg'],'limits':data['limits']},ensure_ascii=False))
print('KEYS',json.dumps(actions,ensure_ascii=False))
