"""Read-only diagnostic around a frozen evaluator; captures poses before resets."""
import sys, argparse, json, math, time
from pathlib import Path
import xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();run=a.checkpoint.resolve().parent.parent
sys.path.insert(0,str(run/'runtime'))
from isaacgym import gymapi
from isaacgym.torch_utils import quat_mul,quat_from_angle_axis,quat_rotate_inverse
import torch,numpy as np
from sable_env import SableRobot as SableEnv
from run_io import atomic_json,sha256,read_json
import evaluate

xml=ET.parse(run/'runtime/assets/sable.xml')
for name,parent in [('spine_pitch','ROOT'),('spine_roll','spine_pitch')]:
    body=xml.find(".//body[@name='%s']/body[@name='%s']"%(parent,name))
    assert body is not None and body.get('quat') is None and body.get('euler') is None
    assert np.allclose(np.fromstring(body.find('joint').get('pos','0 0 0'),sep=' '),0)
axis=lambda name:np.fromstring(xml.find(".//joint[@name='%s']"%name).get('axis'),sep=' ')
records=[];meta={};original=SableEnv.check_termination
def pitch(q):
    x,y,z,w=q.unbind(-1)
    return torch.asin((2*(w*y-z*x)).clamp(-1,1))*180/math.pi
def wrapped(env):
    original(env)
    if not getattr(env,'capture_eval',False):return
    if not hasattr(env,'composed_audit_step'):
        env.composed_audit_step=0;env.composed_audit_active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        env.composed_axes={name:torch.tensor(axis(name),dtype=torch.float,device=env.device).expand(env.num_envs,3) for name in ['spine_pitch','spine_roll']}
    env.composed_audit_step+=1
    valid=env.composed_audit_active & ~env.true_fall
    env.composed_audit_active &= ~env.reset_buf
    t=env.composed_audit_step*env.dt
    if t<1.:return
    q=env.base_quat;reference=env.base_init_state[3:7].expand(env.num_envs,4)
    waist=[]
    for name in ['spine_pitch','spine_roll']:
        index=env.dof_names.index(name);axis_=env.composed_axes[name]
        q=quat_mul(q,quat_from_angle_axis(env.dof_pos[:,index],axis_))
        reference=quat_mul(reference,quat_from_angle_axis(env.default_dof_pos[:,index].expand(env.num_envs),axis_))
        waist.append(env.dof_pos[:,index]*180/math.pi)
    physical=env.rigid_states[:,env.body_names.index('spine_roll'),3:7]
    predicted=q/q.norm(dim=1,keepdim=True);physical=physical/physical.norm(dim=1,keepdim=True)
    distance=torch.minimum((predicted-physical).norm(dim=1),(predicted+physical).norm(dim=1))
    closure=4*torch.asin((distance/2).clamp(0,1))*180/math.pi
    def gravity_error(actual,target):
        ga=quat_rotate_inverse(actual,env.gravity_vec);gt=quat_rotate_inverse(target,env.gravity_vec)
        # atan2 avoids acos numerical error near zero.
        return torch.atan2(torch.cross(ga,gt,dim=1).norm(dim=1),(ga*gt).sum(1))*180/math.pi
    def height(names):return env.rigid_states[:,[env.body_names.index(n) for n in names],2].mean(1)
    thigh=[]
    for side in ['L','R']:
        v=env.rigid_states[:,env.body_names.index(side+'_knee_pitch'),:3]-env.rigid_states[:,env.body_names.index(side+'_hip_pitch'),:3]
        thigh.append(torch.atan2(v[:,0],-v[:,2])*180/math.pi)
    columns=[pitch(env.base_quat),waist[0],waist[1],pitch(q),gravity_error(env.base_quat,env.base_init_state[3:7].expand(env.num_envs,4)),gravity_error(q,reference),closure,
             height(['L_shoulder_pitch','R_shoulder_pitch']),height(['L_hip_pitch','R_hip_pitch']),*thigh]
    columns += [env.dof_pos[:,env.dof_names.index(name)]*180/math.pi for name in ['L_hip_pitch','R_hip_pitch','L_knee_pitch','R_knee_pitch']]
    values=torch.stack(columns,1);values[~valid]=float('nan');records.append(values.cpu().numpy())
    meta.update(standard_root_pitch_deg=float(pitch(env.base_init_state[3:7].unsqueeze(0))[0]),standard_rear_pitch_deg=float(pitch(reference)[0]),commands_m_s=env.commands[:,0].cpu().tolist())
SableEnv.check_termination=wrapped
base=a.output.with_name(a.output.stem+'_evaluation.json')
assert not a.output.exists() and not base.exists(),'Refusing to overwrite diagnostic evidence'
sys.argv=['evaluate.py','--checkpoint',str(a.checkpoint.resolve()),'--output',str(base.resolve())]
evaluate.main()
values=np.stack(records);per_env=np.nanmedian(values,axis=0)
columns=['root_pitch_deg','waist_pitch_deg','waist_roll_deg','rear_pitch_deg','root_gravity_error_deg','rear_gravity_error_deg','composition_vs_physical_error_deg','shoulder_height_m','hip_height_m','left_thigh_sideview_deg','right_thigh_sideview_deg','left_hip_joint_deg','right_hip_joint_deg','left_knee_joint_deg','right_knee_joint_deg']
ev=read_json(base)
report={'created_unix':time.time(),'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':sha256(a.checkpoint),'runtime_manifest_sha256':sha256(run/'runtime_manifest.json'),'diagnostic_source_sha256':sha256(__file__),'wrapped_evaluator_sha256':sha256(run/'runtime/evaluate.py'),'baseline_report':str(base),'baseline_report_sha256':sha256(base),'summary':ev['summary'],'columns':columns,'reference':meta,'per_environment_median':per_env.tolist(),'median_across_environments':dict(zip(columns,np.nanmedian(per_env,axis=0).tolist())),'maximum_composition_error_deg':float(np.nanmax(values[:,:,6])),'by_command':[],'scope':'24 deterministic environments,16s,seed947,after first1s and before first failure only; uses actual ROOT quaternion and measured waist pitch/roll composed about frozen XML axes. No PPO or reward change. Sideview thigh angle includes heading influence.'}
for command in [.03,.045,.06]:
    mask=np.isclose(meta['commands_m_s'],command)
    report['by_command'].append({'command_m_s':command,'num_envs':int(mask.sum()),'median':dict(zip(columns,np.nanmedian(per_env[mask],axis=0).tolist()))})
atomic_json(a.output,report);print('COMPOSED_BODY_AUDIT',json.dumps({k:v for k,v in report.items() if k not in ['per_environment_median','columns','by_command']},ensure_ascii=False))
