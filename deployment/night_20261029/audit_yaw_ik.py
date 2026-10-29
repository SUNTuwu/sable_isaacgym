"""CPU-only kinematic diagnostic; never creates a sim or changes trainer state."""
import json,sys,math,time,xml.etree.ElementTree as ET
from pathlib import Path
import torch
torch.set_num_threads(1)
P=Path('/root/gpufree-data/projects/sable-isaacgym')
r=json.loads((P/'deployment/yaw_sine_20261029/launch_receipt.json').read_text());run=Path(r['run'])
sys.path.insert(0,str(run/'runtime'))
from model_mapping import ModelMapping,quat_apply,quat_multiply
from action_schema import ACTION_NAMES
mapping=ModelMapping(run/'runtime/assets/sable.xml',run/'runtime/assets/physical_manifest.json','cpu')
m=mapping.manifest
bank=torch.load(run/'reset_poses.pt',map_location='cpu')
pose_ids=[0,len(bank['q'])//4,len(bank['q'])//2,3*len(bank['q'])//4]
angles=[-45,-30,-15,0,15,30,45]
cases=[(i,a,s) for i in pose_ids for a in angles for s in range(2)]
q0=torch.stack([bank['q'][i] for i,a,s in cases]);h0=torch.stack([bank['height'][i] for i,a,s in cases]);target=torch.stack([bank['toe'][i] for i,a,s in cases])
names=list(m['default_angles']);indices=torch.tensor([names.index(n) for n in ACTION_NAMES]);hip=names.index('hip_roll');dep=names.index('R_hip_roll_mimic')
nodes=ET.parse(run/'runtime/assets/sable.xml').getroot();ranges={j.get('name'):[float(x) for x in j.get('range').split()] for j in nodes.iter('joint') if j.get('name') in ACTION_NAMES}
limits=torch.tensor([ranges[n] for n in ACTION_NAMES]);n=len(cases)
yaw=torch.tensor([math.radians(a) for i,a,s in cases]);yq=torch.stack((torch.zeros(n),torch.zeros(n),torch.sin(yaw/2),torch.cos(yaw/2)),1)
r0=torch.tensor(m['init_quat_xyzw']).repeat(n,1);root_quat=quat_multiply(yq,r0)
tips=[]
for _,pts,rad in mapping.capsules:
    dist=int(pts.square().sum(-1).argmax());axis=pts[dist]-pts[1-dist]
    tips.append(pts[dist]+rad*axis/axis.norm())
tips=torch.stack(tips)
def measure(q,h):
    p=torch.tensor(m['init_position']).repeat(n,1)+torch.stack((torch.zeros(n),torch.zeros(n),h),1)
    full=torch.tensor([m['default_angles'][name] for name in names]).repeat(n,1);full[:,indices]=q;full[:,dep]=-full[:,hip]
    pos,rot=mapping.forward(p,root_quat,full,names)
    ids=[v[0] for v in mapping.capsules]
    toe=pos[:,ids]+quat_apply(rot[:,ids],tips[None].expand(n,-1,-1))
    z=[]
    for index,points,radius in mapping.capsules:
        end=pos[:,index,None]+quat_apply(rot[:,index,None].expand(-1,2,-1),points[None].expand(n,-1,-1))
        z.append(end[:,:,2].min(1).values-radius)
    return toe,torch.stack(z,1)
torch.manual_seed(9703)
q=q0.clone();mask=torch.tensor([s for i,a,s in cases])[:,None];q[:,:14]+=.2*torch.randn(n,14)*mask
q.requires_grad_(True);h=h0.clone().requires_grad_(True)
optimizer=torch.optim.Adam([{'params':[q],'lr':.02},{'params':[h],'lr':.001}])
t0=time.time()
for k in range(700):
    if k==400:optimizer.param_groups[0]['lr']=.005;optimizer.param_groups[1]['lr']=.0003
    toe,z=measure(q,h);loss=(toe[:,:,:2]-target[:,:,:2]).square().sum()+z.square().sum()+1e-8*(q-q0).square().sum()
    optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():q.clamp_(limits[:,0],limits[:,1]);q[:,-1]=q0[:,-1];h.clamp_(-.04,.005)
toe,z=measure(q,h);xy=(toe[:,:,:2]-target[:,:,:2]).norm(dim=2);score=xy.square().sum(1)+z.square().sum(1)
rows=[]
for i in pose_ids:
    for angle in angles:
        choices=[k for k,c in enumerate(cases) if c[0]==i and c[1]==angle];k=min(choices,key=lambda k:float(score[k]))
        rows.append(dict(pose=i,yaw_deg=angle,max_xy_error_mm=float(xy[k].max()*1000),xy_error_mm=(xy[k]*1000).tolist(),max_ground_error_mm=float(z[k].abs().max()*1000),root_height_adjustment_mm=float((h[k]-h0[k])*1000),joint_delta_deg=((q[k]-q0[k])*180/math.pi).detach().tolist()))
report=dict(task='fixed_initial_feet_yaw_kinematics',source_run=str(run),cpu_threads=1,elapsed_s=time.time()-t0,assumptions=['RootXY fixed, root roll/pitch fixed, yaw prescribed','Waist pitch locked at physical standard -40deg, single opposed hip-roll respected','XY distal toe fixed, lowest capsule surface at ground; rootZ and other14 joints optimized within asset limits','This is local numerical IK evidence, not global infeasibility proof or dynamic stability validation'],rows=rows)
out=P/'deployment/night_20261029/yaw_ik_audit.json';out.write_text(json.dumps(report,indent=2))
for angle in angles:
    group=[v for v in rows if v['yaw_deg']==angle];print(angle,'maxXY range mm',min(v['max_xy_error_mm'] for v in group),max(v['max_xy_error_mm'] for v in group),'ground max mm',max(v['max_ground_error_mm'] for v in group),flush=True)
print('saved',out,'elapsed',time.time()-t0,flush=True)
