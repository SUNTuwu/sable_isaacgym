"""Controller preservation, single-foot sacrifice, reset and dense-mask contracts."""
import math
from types import SimpleNamespace
import numpy as np
import torch
from teacher_pace_stability import PaceStability,tracking_scores,dense_report,acceptance


def main():
    ids=['02_STRAFE_LEFT','03_STRAFE_RIGHT','01_FORWARD_POLICY','04_TURN_LEFT']
    env=SimpleNamespace(num_envs=4,device='cpu',dt=.02,teacher_ids=torch.arange(4),
                        cfg=SimpleNamespace(teacher_bank={'clips':[{'id':x} for x in ids]}))
    c=PaceStability(env,True)
    old=torch.full((4,15),-math.radians(20));new=-old
    first=torch.zeros(4,dtype=torch.bool)
    assert torch.equal(c.filter(old,new,first),new),'Strength0 preserves all previous actions'
    c.set_strength(1.);limited=c.filter(old,new,first)
    assert torch.equal(limited[2:],new[2:]),'Other skills must remain exact'
    assert torch.equal(limited[:2,[0,2,3,4,7,8,9,14]],new[:2,[0,2,3,4,7,8,9,14]])
    assert torch.allclose((limited-old)[:2,c.hind],c.rates()[:2]*env.dt,atol=1e-7)
    assert torch.equal(c.filter(old,new,torch.ones_like(first)),new),'Reset warm-start'
    sigma=torch.full((4,),.020);error=torch.zeros(4,4,3)
    reward,cost=tracking_scores(error,sigma)
    assert torch.allclose(reward,torch.full((4,),12.)) and not cost.any()
    error[:,3,0]=.05;bad,badcost=tracking_scores(error,sigma)
    assert (bad<6).all() and (badcost<-3).all(),'A sacrificed rear foot must remove worst-foot reward'
    terms={'reference_endpoint_reward':torch.ones(4)*7}
    # Constant tracking error has zero error velocity even if reference travels.
    for _ in range(3):out=c.apply(dict(terms),error,torch.zeros(4,15))
    assert not out['pace_endpoint_error_velocity_cost'].any()
    error[:,2,2]+=.02
    out=c.apply(dict(terms),error,torch.ones(4,15)*.1)
    assert (out['pace_endpoint_error_velocity_cost'][:2]<0).all()
    assert not out['pace_endpoint_error_velocity_cost'][2:].any()
    assert torch.equal(out['reference_endpoint_reward'][2:],terms['reference_endpoint_reward'][2:])
    assert (out['pace_endpoint_position_cost']>=-6).all()
    c.reset(torch.arange(4));out=c.apply(dict(terms),error,torch.zeros(4,15))
    assert not out['pace_endpoint_error_velocity_cost'].any() and not out['pace_residual_acceleration_cost'].any()
    records=[]
    for t in range(6):
        e=np.zeros((4,4,3));o=np.zeros((4,15));alive=np.ones(4,dtype=bool)
        if t>=3:alive[0]=False;e[0]=99;o[0]=99
        records.append((e,o,alive))
    dense=dense_report(records,.02,1.)
    assert max(dense['tip_rmse_mm_per_foot'])==0 and dense['hind_reference_relative_jump_p95_mm']==0
    report=dict(horizon_s=16.,executed_simulation_s=16.,survival_fraction=1.,xy_rmse_m_s=.01,yaw_rmse_rad_s=.01)
    assert acceptance(report,dense)['passed']
    dense['curriculum_strength']=.99;assert not acceptance(report,dense)['passed']
    print('pace_stability_contract_passed: nonpace_preservation, worst_foot, residual_rate, reset, reference_relative_jitter, first_episode_mask, acceptance')


if __name__=='__main__':main()
