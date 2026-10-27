"""Physical shadow probe; rewards are measured but no policy is updated."""
import argparse,json
from pathlib import Path
from isaacgym import gymapi
import torch
from migration_config import student_config
from migration_env import create_migration_env
from history_policy import HistoryActorCritic
from model_mapping import quat_apply
from teacher_adapter import verify_teacher_manifest
from run_io import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(947)
    names=['forward','stop','reverse','left','right','turn_left','turn_right'];per=2;n=len(names)*per
    cfg=student_config(verify_teacher_manifest(a.teacher)['run'],'S3',n)
    cfg.migration.yaw_error_scale_rad_s=.35;cfg.commands.resampling_time=1000.;cfg.env.episode_length_s=20.
    cfg.reference.file='/intentionally_absent_force_probe.json'
    env=create_migration_env(cfg,noise=False);env.capture_eval=True
    policy=HistoryActorCritic().to(env.device).eval();policy.load_state_dict(torch.load(a.checkpoint,map_location=env.device)['model_state_dict'])
    env.reset();command=env.commands16.clone()
    command[:,:3]=command.new_tensor([[.105,0,0],[0,0,0],[-.04,0,0],[0,.04,0],[0,-.04,0],[0,0,.35],[0,0,-.35]]).repeat_interleave(per,dim=0)
    env.set_commands(command)
    props=env.gym.get_actor_rigid_body_properties(env.envs[0],env.actor_handles[0])
    com=command.new_tensor([[props[int(i)].com.x,props[int(i)].com.y,props[int(i)].com.z]for i in env.feet_indices])
    alive=torch.ones(n,dtype=torch.bool,device=env.device);rows=[];comparisons=[];previous=None;valid_rows=[]
    timing_error=0.;aggregate_error=0.;finite=True
    with torch.no_grad():
        for tick in range(200):
            if tick==100:
                env.reset_idx(torch.tensor([1],device=env.device));env._refresh_proprio_state();env.compute_observations();previous=None
            actual_error=command[:,2]-env.heading_yaw_rate
            _,_,reward,done,_=env.step(policy.act_inference(env.get_observations()))
            s=env.eval_snapshot;d=s['contact_force_analysis'];timing_error=max(timing_error,float((s['contact_error_yaw']-actual_error).abs().max()))
            finite=finite and all(bool(torch.isfinite(value).all())for value in d.values()if value.dtype!=torch.bool)
            parts=s['weighted_rewards'];aggregate_error=max(aggregate_error,float((sum(v for k,v in parts.items()if k!='total')-reward).abs().max()))
            state=env.rigid_states[:,env.feet_indices].clone();offset=quat_apply(state[...,3:7],com[None].expand(n,-1,-1));center=state[...,:3]+offset
            if previous is not None:
                before,before_center,mask=previous;valid=alive&mask&~done
                average_velocity=(state[...,7:10]+before[...,7:10])*.5
                origin_difference=(state[...,:3]-before[...,:3])/env.dt
                com_difference=(center-before_center)/env.dt
                comparisons.append(torch.stack(((origin_difference-average_velocity).square().sum(-1),(com_difference-average_velocity).square().sum(-1)),-1)[valid].cpu())
            previous=(state,center,alive&~done)
            rows.append({k:v.clone()for k,v in d.items()});valid_rows.append(alive.clone());alive&=~done
    errors=torch.cat(comparisons).mean((0,1)).sqrt().tolist()
    balance=torch.stack([r['normal_balance_error_n']for r in rows]);valid=torch.stack([r['resultant_valid']for r in rows])&torch.stack(valid_rows)
    balance_rms=float(balance[valid].square().mean().sqrt())
    summaries=[]
    for g,name in enumerate(names):
        sl=slice(g*per,(g+1)*per)
        summaries.append(dict(name=name,survival_fraction=float(alive[sl].float().mean()),
            mean_costs={key:float(torch.stack([r[key][sl]for r in rows]).mean())for key in ['wrong_cost','cancellation_cost','loaded_slip_cost']},
            mean_force_n=torch.stack([r['sum_force_world'][sl]for r in rows]).mean((0,1)).cpu().tolist()))
    result=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),horizon_s=4.,finite=finite,normal_balance_rms_error_n=balance_rms,
        preaction_yaw_error_max_difference=timing_error,reward_aggregation_max_error=aggregate_error,
        nominal_weight_n=env.contact_nominal_weight_n,foot_local_COM_m=com.cpu().tolist(),
        reported_linear_velocity_vs_origin_FD_RMSE_m_s=errors[0],reported_linear_velocity_vs_COM_FD_RMSE_m_s=errors[1],
        groups=summaries,reference_execution_calls=env.reference_execution_calls,task_acceptance=False,
        scope='Four-second shadow/reset/aggregation and velocity-origin diagnostic. Endpoint forces sampled50Hz; physical state unchanged by reward analysis. Not16s stability or task acceptance.')
    atomic_json(a.output,result);print(json.dumps(result));env.gym.destroy_sim(env.sim)
    assert finite and timing_error<1e-6 and aggregate_error<1e-4 and balance_rms<.02


if __name__=='__main__':main()
