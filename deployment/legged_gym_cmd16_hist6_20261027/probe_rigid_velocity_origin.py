"""Isolated off-center COM test of this installed GPU Gym velocity convention."""
from pathlib import Path
from isaacgym import gymapi,gymtorch
import torch
from model_mapping import quat_apply
from contact_force_analysis import material_point_velocity_from_poses,material_point_velocity
from run_io import atomic_json


def main():
    gym=gymapi.acquire_gym();p=gymapi.SimParams();p.dt=.0005;p.up_axis=gymapi.UP_AXIS_Z
    p.gravity=gymapi.Vec3(0,0,0);p.use_gpu_pipeline=True;p.physx.use_gpu=True
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,p);o=gymapi.AssetOptions();o.angular_damping=0.;o.linear_damping=0.
    a=gym.create_box(sim,.02,.02,.02,o);env=gym.create_env(sim,gymapi.Vec3(-1,-1,-1),gymapi.Vec3(1,1,1),1)
    actor=gym.create_actor(env,a,gymapi.Transform(),'offset',0,0);props=gym.get_actor_rigid_body_properties(env,actor)
    props[0].com=gymapi.Vec3(.02,0,0);gym.set_actor_rigid_body_properties(env,actor,props,False);gym.prepare_sim(sim)
    root=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim));state=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim);root[:,7]=.1;root[:,8]=.2;root[:,12]=10.;gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(root))
    com=torch.tensor([[.02,0,0]],device='cuda');local_point=torch.tensor([[0.,0.,-.01]],device='cuda');old=None;values=[]
    for tick in range(500):
        gym.simulate(sim);gym.fetch_results(sim,True);gym.refresh_rigid_body_state_tensor(sim);s=state.clone();c=s[:,:3]+quat_apply(s[:,3:7],com)
        if old is not None:
            prev,pc=old;v=(s[:,7:10]+prev[:,7:10])*.5
            point=s[:,:3]+quat_apply(s[:,3:7],local_point)
            analytic=material_point_velocity(s,point,com)
            finite=material_point_velocity_from_poses(prev,s,point,p.dt)
            values.append(torch.stack((((s[:,:3]-prev[:,:3])/p.dt-v).norm(dim=-1),((c-pc)/p.dt-v).norm(dim=-1),(analytic-finite).norm(dim=-1)),dim=-1).cpu()[0])
        old=s,c
    values=torch.stack(values)[5:];r=dict(origin_fd_vs_reported_rms_m_s=float(values[:,0].square().mean().sqrt()),com_fd_vs_reported_rms_m_s=float(values[:,1].square().mean().sqrt()),material_point_com_formula_vs_pose_secant_rms_m_s=float(values[:,2].square().mean().sqrt()),inferred_linear_velocity_reference='body COM',no_learning=True)
    r['passed']=r['origin_fd_vs_reported_rms_m_s']>.1 and r['com_fd_vs_reported_rms_m_s']<.001 and r['material_point_com_formula_vs_pose_secant_rms_m_s']<.001
    atomic_json(Path('results/cmd16_hist6/contact_force_probe_20261027/velocity_origin.json'),r);print(r);gym.destroy_sim(sim);assert r['passed']


if __name__=='__main__':main()
