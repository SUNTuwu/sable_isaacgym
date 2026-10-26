import argparse,json,time
from pathlib import Path
from isaacgym import gymapi
import torch,numpy as np
from sable_config import SableCfg,MODEL
from sable_env import SableRobot

def create_env(num=8,fixed=False,noise=False,cfg=None):
    cfg=cfg or SableCfg();cfg.env.num_envs=num;cfg.asset.fix_base_link=fixed
    cfg.noise.add_noise=noise;cfg.domain_rand.randomize_friction=noise
    params=gymapi.SimParams();params.dt=cfg.sim.dt;params.substeps=cfg.sim.substeps;params.up_axis=gymapi.UP_AXIS_Z
    params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=True
    params.physx.use_gpu=True;params.physx.num_threads=4;params.physx.solver_type=1
    params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=2
    params.physx.contact_offset=.0005;params.physx.rest_offset=0.;params.physx.max_depenetration_velocity=.5
    params.physx.bounce_threshold_velocity=.1;params.physx.contact_collection=gymapi.ContactCollection.CC_ALL_SUBSTEPS
    params.physx.max_gpu_contact_pairs=2**22;params.physx.default_buffer_size_multiplier=5.
    env=SableRobot(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
    return env

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fixed',action='store_true');parser.add_argument('--seconds',type=float,default=16);parser.add_argument('--num-envs',type=int,default=4);parser.add_argument('--front-friction-scale',type=float,default=1.);parser.add_argument('--output',type=Path);args=parser.parse_args()
    cfg=SableCfg();cfg.material_friction.front_scale=args.front_friction_scale
    env=create_env(args.num_envs,args.fixed,cfg=cfg);env.reset_noise=0.;env.fixed_command=0.
    env.reset();env.last_episode_records=[]
    print('FLOOR_PLACEMENT',json.dumps({'ground':'native_plane_z0','requested_origins':env.env_origins.tolist()}),flush=True)
    history=[];t=time.monotonic()
    for k in range(int(args.seconds/env.dt)):
        actions=torch.zeros(env.num_envs,env.num_actions,device=env.device)
        if args.fixed:actions[:,MODEL['policy_joint_names'].index('hip_roll')]=.7*np.sin(2*np.pi*.5*k*env.dt)
        obs,_,reward,done,info=env.step(actions)
        if k%5==0:
            history.append({'t':(k+1)*env.dt,'height':env.root_states[0,2].item(),'xyz':env.root_states[0,:3].tolist(),
                'q':env.dof_pos[0].tolist(),'qd':env.dof_vel[0].tolist(),'torque':env.torques[0].tolist(),
                'feet_force':env.contact_forces[0,env.feet_indices,2].tolist(),'contacts':env.contact_forces[0].norm(dim=1).tolist()})
        if not torch.isfinite(obs).all():raise RuntimeError('Nonfinite physics')
        if k%50==0:
            print('PROBE',json.dumps({'t':k*env.dt,'xyz':env.root_states[:,:3].tolist(),'gravity':env.projected_gravity.tolist(),
                'feet_force':env.contact_forces[:,env.feet_indices,2].tolist(),'fallen':env.true_fall.tolist(),
                'waist':env.dof_pos[:,env.waist].tolist(),'q':env.dof_pos.tolist(),'max_qd':env.dof_vel.abs().max(1).values.tolist()}),flush=True)
    falls=sum(int(r[1]) for r in env.last_episode_records)
    coupling=max(abs(h['q'][env.hip]+h['q'][env.dependent]) for h in history)
    excursion=max(h['q'][env.hip] for h in history)-min(h['q'][env.hip] for h in history)
    passed=falls==0 and coupling<.02 and abs(env.import_mass-1.5)<.02 and (not args.fixed or excursion>.05)
    report={'passed':passed,'fixed':args.fixed,'seconds':args.seconds,'wall_seconds':time.monotonic()-t,'falls':falls,
        'coupling_max_abs_rad':coupling,'hip_roll_excursion_rad':excursion,'mass_kg':env.import_mass,'dof_names':env.dof_names,'body_names':env.body_names,
        'history':history,'episode_records':env.last_episode_records,'scope':'Simulation baseline only, not hardware validation'}
    out=Path(__file__).resolve().parent/'verification';out.mkdir(exist_ok=True)
    report['front_friction_scale']=cfg.material_friction.front_scale
    target=args.output or out/('fixed.json' if args.fixed else 'standing.json')
    target.write_text(json.dumps(report,indent=2))
    print('PREFLIGHT',json.dumps({k:v for k,v in report.items() if k not in ['history','episode_records','dof_names','body_names']}),flush=True)
    env.gym.destroy_sim(env.sim)
    if not passed:raise SystemExit(2)
