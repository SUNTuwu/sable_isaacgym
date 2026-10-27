"""Non-learning GPU sphere test: sliding force balance and true rolling contact."""
from pathlib import Path
from isaacgym import gymapi,gymtorch
import torch
from contact_force_analysis import material_point_velocity_from_poses
from run_io import atomic_json


def main():
    gym=gymapi.acquire_gym();p=gymapi.SimParams();p.dt=.0005;p.substeps=1;p.up_axis=gymapi.UP_AXIS_Z
    p.gravity=gymapi.Vec3(0,0,-9.81);p.use_gpu_pipeline=True;p.physx.use_gpu=True;p.physx.solver_type=1
    p.physx.num_position_iterations=8;p.physx.num_velocity_iterations=2;p.physx.contact_collection=gymapi.ContactCollection.CC_ALL_SUBSTEPS
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,p)
    plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);plane.static_friction=0.;plane.dynamic_friction=0.;gym.add_ground(sim,plane)
    opt=gymapi.AssetOptions();opt.linear_damping=0.;opt.angular_damping=0.;opt.density=1000.
    asset=gym.create_sphere(sim,.02,opt);shapes=gym.get_asset_rigid_shape_properties(asset)
    for shape in shapes:shape.friction=2.;shape.restitution=0.;shape.rolling_friction=0.;shape.torsion_friction=0.
    gym.set_asset_rigid_shape_properties(asset,shapes)
    env=gym.create_env(sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,2),1);pose=gymapi.Transform();pose.p.z=.0201
    actor=gym.create_actor(env,asset,pose,'sphere',0,0);gym.prepare_sim(sim)
    root=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim));state=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim));force=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim);root[:,7]=2.;gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(root))
    mass=gym.get_actor_rigid_body_properties(env,actor)[0].mass;previous=None;rows=[]
    for tick in range(1800):
        gym.simulate(sim);gym.fetch_results(sim,True);gym.refresh_rigid_body_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
        current=state.clone()
        if previous is not None:
            point=current[:,:3].clone();point[:,2]-=.02
            vp=material_point_velocity_from_poses(previous,current,point,p.dt)
            reconstructed=(current[:,7:10]-previous[:,7:10])*mass/p.dt;reconstructed[:,2]+=mass*9.81
            analytic=current[:,7:10]+torch.cross(current[:,10:13],point-current[:,:3],dim=-1)
            rows.append(torch.cat((reconstructed,force.clone(),vp,analytic,current[:,7:10]),-1).cpu()[0])
        previous=current
    values=torch.stack(rows);normal=values[:,5];slip=values[:,6:8].norm(dim=-1)
    sliding=(normal>mass*8)&(values[:,6]>.1)&(torch.arange(len(rows))>20)
    rolling=(normal>mass*8)&(slip<.005)&(values[:,12]>.5)&(torch.arange(len(rows))>800)
    relative=float(((values[sliding,0]+normal[sliding])/normal[sliding]).square().mean().sqrt())
    point_error=float((values[:,6:9]-values[:,9:12]).square().sum(-1)[100:].mean().sqrt())
    horizontal_error=float((values[:,6:8]-values[:,9:11]).square().sum(-1)[100:].mean().sqrt())
    result=dict(physics_dt_s=p.dt,mass_kg=mass,sliding_samples=int(sliding.sum()),rolling_samples=int(rolling.sum()),
        coulomb_horizontal_relative_rms_error=relative,material_point_secant_vs_endpoint_velocity_rms_m_s=point_error,
        horizontal_material_point_velocity_rms_error_m_s=horizontal_error,
        rolling_center_speed_mean_m_s=float(values[rolling,12].mean()),rolling_point_speed_mean_m_s=float(slip[rolling].mean()),
        normal_only_tensor_horizontal_rms_n=float(values[:,3:5].square().sum(-1).mean().sqrt()),
        no_learning=True,scope='Isolated sphere, same GPU PhysX dt/solver/mean-friction recipe. Checks horizontal momentum force against Coulomb sliding and finite-pose point velocity against rigid rolling. The reward uses horizontal sliding: vertical secant displacement includes rotational curvature and is not an endpoint vertical velocity. Not robot locomotion acceptance.')
    result['passed']=result['sliding_samples']>40 and result['rolling_samples']>40 and relative<.05 and horizontal_error<.005
    atomic_json(Path('results/cmd16_hist6/contact_force_probe_20261027/backend_validation.json'),result)
    print(result);gym.destroy_sim(sim);assert result['passed']


if __name__=='__main__':main()
