"""Physical-event and integral-boundary CPU checks, no training workload."""
import math
import torch
from teacher_gait_balance import BilateralCycleIntegrals,turn_scores


def run_contract():
    times=torch.tensor([[.06,.10,.20,.50,1.]])
    distance=torch.ones_like(times)*.30;yes=torch.ones_like(times,dtype=torch.bool)
    reward,cost=turn_scores(times,distance,yes,yes,.02)
    assert torch.allclose(cost,torch.tensor([[-10/3,-2.,-1.,-.4,-.2]]),atol=1e-5)
    assert (cost[0,1:]>cost[0,:-1]).all()
    d=torch.tensor([[-.4,0,.199,.20,.30,.40,.50]])
    r,c=turn_scores(torch.ones_like(d)*.2,d,torch.ones_like(d,dtype=torch.bool),torch.ones_like(d,dtype=torch.bool),.02)
    assert (r[0,:4]==0).all() and 0<r[0,4]<r[0,5] and torch.allclose(r[0,5:],torch.tensor([3.,3.]))
    assert not turn_scores(times,distance,~yes,~yes,.02)[0].any()
    assert not turn_scores(times,distance,~yes,~yes,.02)[1].any()

    def cycle(dt,width_fn,air_fn=None,enabled=True,cycles=1):
        tracker=BilateralCycleIntegrals(1,'cpu',dt);out=[]
        for j in range(round(cycles/dt)):
            k=j%round(1/dt);width=torch.tensor([width_fn(k,round(1/dt))])
            airborne=torch.tensor([air_fn(k,round(1/dt)) if air_fn else [False]*4])
            v=tracker.update(~airborne,airborne.float()*.02,width,torch.tensor([1.+j*dt]),torch.ones(1),torch.tensor([enabled]))
            if v['complete'].item():out.append(v)
        assert len(out)==(cycles if enabled else 0)
        return tracker,out

    # Correct side signs and phase alternation do not demand simultaneous swing.
    w=lambda k,n:[.03,-.03,.03,-.03]
    a=lambda k,n:[k<n//2,k>=n//2,k>=n//2,k<n//2]
    tracker,out=cycle(.02,w,a,cycles=2)
    assert all(abs(v['air_cost'].item())<1e-6 for v in out)
    assert not out[0]['width_scored'].any() and out[1]['width_scored'].all()
    assert out[1]['width_cost']==0
    # Integrate first and square afterwards: opposing width deviations cancel.
    w=lambda k,n:[.02 if k<n//2 else .04,-.03,.03,-.03]
    _,out=cycle(.02,w);assert abs(out[0]['width_cost'].item())<1e-6
    # Same physical asymmetry must produce the same event cost at10/20ms.
    for dt in [.01,.02]:
        _,out=cycle(dt,lambda k,n:[.03,-.04,.03,-.03])
        assert abs(out[0]['width_cost'].item()+math.expm1(1)/2)<1e-4
        _,out=cycle(dt,lambda k,n:[.03,-.03,.03,-.03],lambda k,n:[k<round(.4/dt),k<round(.3/dt),False,False])
        assert abs(out[0]['air_cost'].item()+math.expm1(1)/2)<1e-4
    _,out=cycle(.02,w,enabled=False)
    # Missing feet are never zero-filled as a falsely symmetric placement.
    _,out=cycle(.02,w,lambda k,n:[True,False,False,False])
    assert not out[0]['width_scored'][0,0]
    assert out[0]['width_scored'][0,1]
    tracker.reset(torch.tensor([0]));assert not tracker.air.any() and not tracker.width_valid.any() and not tracker.elapsed.any()
    v=tracker.update(torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4),torch.ones(1,4)*.03,
                     torch.tensor([.5]),torch.ones(1),torch.ones(1,dtype=torch.bool))
    assert not v['complete'].any() and not tracker.elapsed.any()
    print('gait_balance_contract_passed: reciprocal60_100_200_500_1000ms, stride200_400_cap, full_cycle_integral_before_square, phase_shift,10_20ms_invariance, missing_pair, reset, warmup')


if __name__=='__main__':run_contract()
