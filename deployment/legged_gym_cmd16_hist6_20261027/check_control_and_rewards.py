"""Direct old-controller torque comparison and post-clipping command-cost check."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env, MigrationRobot
from sable_env import SableRobot
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,sha256


def method_tree(path, class_name, name):
    tree=ast.parse(Path(path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    teacher=verify_teacher_manifest(a.teacher)
    source=Path(__file__).resolve().parent
    old=Path(teacher['run'])/'runtime/sable_env.py'
    assert method_tree(old,'SableRobot','_compute_torques')==method_tree(source/'sable_env.py','SableRobot','_compute_torques')
    cfg=student_config(teacher['run'],num_envs=6)
    env=create_migration_env(cfg,noise=False)
    env.reset()
    # Test the real aggregator with a negative aggregate, not a replica formula.
    probe=SimpleNamespace(device=env.device,rew_buf=torch.zeros(6,device=env.device),
        reward_names=['gait_clearance','velocity_error'],
        reward_functions=[lambda:torch.ones(6,device=env.device),lambda:torch.ones(6,device=env.device)],
        reward_scales={'gait_clearance':-32*env.dt,'velocity_error':-2*env.dt},
        episode_sums={n:torch.zeros(6,device=env.device) for n in ('gait_clearance','velocity_error')},cfg=cfg)
    SableRobot.compute_reward(probe)
    torch.testing.assert_close(probe.rew_buf,torch.full((6,),-2*env.dt,device=env.device))
    # Command error must be active for both signs, lateral-only and zero commands.
    motion=SimpleNamespace(omni=True,cfg=cfg,commands=torch.zeros(6,4,device=env.device),
        heading_linear_velocity=torch.tensor([[.05,0],[-.05,0],[0,.05],[0,-.05],[0,0],[.03,.04]],device=env.device))
    cost=MigrationRobot._reward_velocity_error(motion)
    torch.testing.assert_close(cost,torch.tensor([1.,1.,1.,1.,0.,1.],device=env.device))
    env.attach_teacher(teacher['checkpoint'],teacher['sha256'],0.)
    keys=['num_envs','default_dof_pos','cfg','hard_limits','p_gains','d_gains','torque_limits','max_speed','large','dependent']
    legacy=SimpleNamespace(**{k:getattr(env,k) for k in keys})
    legacy.active=env.active[:14];legacy.motion_reference=env.teacher.reference
    maximum=0.
    torch.manual_seed(947)
    env.episode_length_buf.fill_(100)
    env.commands[:,0]=.12
    for case in range(3):
        env.dof_pos.copy_(env.default_dof_pos+torch.randn_like(env.dof_pos)*.2)
        env.dof_vel.copy_((torch.rand_like(env.dof_vel)*2-1)*env.max_speed*(case/1.))
        legacy.dof_pos=env.dof_pos;legacy.dof_vel=env.dof_vel
        env.teacher.prepare();env.substep_index=0
        for _ in range(cfg.control.decimation):
            actual=env._compute_torques(torch.zeros(6,15,device=env.device))
            expected=SableRobot._compute_torques(legacy,env.teacher.actions)
            maximum=max(maximum,float((actual-expected).abs().max()))
            assert not actual[:,env.dependent].any()
    assert maximum < 1e-6, maximum
    result={'passed':True,'old_controller_function_ast_identical':True,'legacy_source':str(old),
        'max_torque_error_nm':maximum,'environments':6,'state_cases':3,'physics_substeps_per_case':40,
        'post_positive_clip_velocity_cost':probe.rew_buf.cpu().tolist(),
        'velocity_error_cases':cost.cpu().tolist(),'source_sha256':{n:sha256(source/n)for n in
            ('sable_env.py','migration_env.py','teacher_adapter.py','motor_envelope.py','migration_config.py')},
        'scope':'Same-state direct comparison with frozen v44 control method at every substep; no survival claim'}
    atomic_json(a.output,result)
    print(json.dumps(result),flush=True)
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
