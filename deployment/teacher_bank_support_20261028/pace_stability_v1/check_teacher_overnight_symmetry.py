"""Signed symmetry scores, actual same-foot periods and soft pace lift costs."""
import torch
from teacher_event_symmetry import FootEventPairSymmetry,signed_position_scores,signed_period_scores
from teacher_lift_limits import LiftCountGuard,turn_acceptance


def run_contract():
    error=torch.tensor([0.,.002,.005,.010,.030,.050])
    reward,cost=signed_position_scores(error)
    assert reward[0]==1 and cost[0]==0 and (reward[1:]<reward[:-1]).all()
    assert (cost[1:]<cost[:-1]).all() and (cost>=-6).all()
    total=reward+cost
    assert total[1]>0 and (total[2:]<0).all(),'Close symmetry rewarded, asymmetry penalized'
    assert cost[4]<-4 and cost[5]<cost[4],'Large errors must retain a slope, no old flat cap'
    tr,tc=signed_period_scores(torch.tensor([0.,.02,.1,.2]),torch.ones(4))
    assert tr[0]==2 and tc[0]==0 and tr[1]+tc[1]>0 and (tr[2:]+tc[2:]<0).all()
    s=FootEventPairSymmetry(1,'cpu',.02)
    xy=torch.tensor([[[.04,.03],[.04,-.03],[-.02,.03],[-.02,-.03]]])
    yes=torch.ones(1,4,dtype=torch.bool);none=~yes
    left=none.clone();left[:,[0,2]]=True;right=none.clone();right[:,[1,3]]=True
    age=torch.tensor([0.]);period=torch.ones(1);on=torch.ones(1,dtype=torch.bool)
    s.update(xy,yes,none,none,age,period,on)
    for t,event in [(.1,left),(.6,right),(1.1,left),(1.6,right)]:
        out,_,_=s.update(xy,yes,none,event,age+t,period,on)
    assert out['paired_period_symmetry_reward'][0]>0 and abs(float(out['paired_period_squared_cost'][0]))<1e-5
    assert torch.allclose(s.pair_periods[:,0],torch.ones(1,2,2),atol=1e-6),'Half-cycle phase offset is allowed'
    s.update(xy,yes,none,left,age+2.1,period,on)
    out,_,_=s.update(xy,yes,none,right,age+2.9,period,on)
    assert out['paired_period_squared_cost'][0]<0 and torch.allclose(s.period_delta[:,0],torch.ones(1,2)*-.3,atol=1e-6)
    s.reset(torch.tensor([0]));out,_,_=s.update(xy,yes,none,none,age,period,on)
    assert out['paired_period_symmetry_reward'][0]==0 and out['paired_period_squared_cost'][0]==0
    # The pace counter includes brief/airborne shaking but has soft costs only.
    config={'teachers':{'02_STRAFE_LEFT':{'per_foot_max_hz':1.,'total_max_hz':4.,'hard_limit':False,'frequency_weight':2.,'jitter_weight':.5}}}
    g=LiftCountGuard(1,'cpu',.02,torch.zeros(1,dtype=torch.long),['02_STRAFE_LEFT'],config)
    for j in range(20):
        hit=g.update(torch.ones(1,4)*(.003 if j%2 else 0),torch.tensor([j*.02]))
        assert not hit.any()
    assert g.count.min()>4 and g.frequency_cost()[0]<0 and not g.violation.any()
    assert turn_acceptance({'teacher_id':'02_STRAFE_LEFT'},config) is None
    print('overnight_signed_symmetry_contract_passed: position_square_signed_scores, stronger_large_error_slope, true_period_not_phase_difference, missing_data, soft_pace_lift_counting')


if __name__=='__main__':run_contract()
