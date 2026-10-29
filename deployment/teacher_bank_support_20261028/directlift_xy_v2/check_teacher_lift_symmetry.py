"""Count20ms jitter, reject over-budget immediately; mirrored event geometry."""
import math
import torch
from teacher_lift_limits import DirectLiftCounter,LiftCountGuard,limit_turn_offset,turn_acceptance
from teacher_event_symmetry import FootEventPairSymmetry,relative_xy


def run_contract():
    counter=DirectLiftCounter(1,'cpu');events=[]
    for z in [0,.003,0,.003,0,.003]:events.append(counter.update(torch.ones(1,4)*z).clone())
    assert sum(int(e[0,0]) for e in events)==3,'20ms upstrokes must not be hidden by60ms gate'
    counter.reset(torch.tensor([0]));assert not counter.update(torch.ones(1,4)*.03).any()
    assert counter.update(torch.ones(1,4)*.034).all(),'Held-air shaking also counts'
    counter=DirectLiftCounter(1,'cpu')
    assert not counter.update(torch.zeros(1,4)).any()
    assert not counter.update(torch.ones(1,4)*.0001).any()
    assert counter.update(torch.ones(1,4)*.002).all()
    assert not counter.update(torch.ones(1,4)*.004).any(),'One continuous upstroke counts once'
    config={'teachers':{'04_TURN_LEFT':{'per_foot_max_hz':.5,'total_max_hz':2.,'window_s':4.,
        'baseline_total_count':396.,'baseline_horizon_s':4.}}}
    g=LiftCountGuard(1,'cpu',.02,torch.zeros(1,dtype=torch.long),['04_TURN_LEFT'],config)
    for j,z in enumerate([0,.003,0,.003]):hit=g.update(torch.ones(1,4)*z,torch.tensor([j*.02]))
    assert hit.all() and g.interval_failure.all()
    g.reset(torch.tensor([0]));assert not g.count.any() and not g.violation.any()
    assert g.observations(torch.zeros(1)).shape==(1,5)
    for j in range(202):
        z=.003 if j in (1,101) else 0
        assert not g.update(torch.ones(1,4)*z,torch.tensor([j*.02])).any(),'Exactly2s spacing is allowed'
    before=torch.tensor([[.2,-.2],[.2,-.2],[.2,-.2]])
    request=-before;enabled=torch.tensor([True,False,True]);first=torch.tensor([False,False,True])
    result=limit_turn_offset(before,request,.01,enabled,first)
    assert torch.allclose(result[0],torch.tensor([.19,-.19])) and torch.equal(result[1:],request[1:])
    r=dict(teacher_id='04_TURN_LEFT',quality_direct_lifts_per_foot=[8]*4,horizon_s=16.,num_envs=4,
        survival_fraction=1.,yaw_rmse_rad_s=.1,mean_command=[0,0,.785],mean_actual=[0,0,.78])
    assert turn_acceptance(r,config)['passed']
    r['survival_fraction']=0.;assert not turn_acceptance(r,config)['passed'],'Early reset cannot fake reduced counts'
    r['survival_fraction']=1.;r['mean_actual'][2]=0.;assert not turn_acceptance(r,config)['passed'],'Standing still cannot pass'
    r['mean_actual'][2]=.78;r['quality_direct_lifts_per_foot']=[100]*4
    assert not turn_acceptance(r,config)['passed']

    xy=torch.tensor([[[.04,.03],[.04,-.03],[-.02,.03],[-.02,-.03]]])
    yes=torch.ones(1,4,dtype=torch.bool);no=~yes;age=torch.tensor([0.]);period=torch.ones(1);on=torch.ones(1,dtype=torch.bool)
    s=FootEventPairSymmetry(1,'cpu',.02);s.update(xy,yes,no,no,age,period,on)
    left=no.clone();left[:,[0,2]]=True;right=no.clone();right[:,[1,3]]=True
    c,_,w=s.update(xy,~left,left,no,age+.1,period,on);assert not w.any(),'No missing-side zero fill'
    c,d,w=s.update(xy,~right,right,no,age+.6,period,on)
    assert all(abs(float(v[0]))<1e-7 for v in c.values()) and (w[:,1]>0).all()
    c,_,_=s.update(xy,yes,no,yes,age+.8,period,on);assert all(v[0]==0 for v in c.values())
    asymmetric=xy.clone();asymmetric[:,0,0]+=.02;asymmetric[:,0,1]+=.01
    c,_,_=s.update(asymmetric,yes,no,left,age+1.,period,on)
    assert c['paired_touchdown_x_exp_cost'][0]<0 and c['paired_touchdown_y_exp_cost'][0]<0
    assert c['paired_liftoff_x_exp_cost'][0]==0
    # Takeoff uses the last actual stance record, not an airborne/current body point.
    s=FootEventPairSymmetry(1,'cpu',.02);s.update(xy,yes,no,no,age,period,on)
    c,_,_=s.update(xy+10,no,yes,no,age+.1,period,on)
    assert all(v[0]==0 for v in c.values())
    c,_,w=s.update(xy,yes,no,left,age+4,period,on);assert not w[:,0].any(),'Stale counterpart is excluded'
    s.reset(torch.tensor([0]));assert not s.valid.any()
    # Global translation and yaw do not change chest/pelvis-local XY geometry.
    points=torch.cat((xy,torch.zeros(1,4,1)),-1);anchor=torch.zeros(1,2,3)
    q=torch.tensor([[[0.,0.,0.,1.],[0.,0.,0.,1.]]]);base=relative_xy(points,anchor,q)
    yaw=.8;rot=torch.tensor([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    points[:,:,:2]=points[:,:,:2]@rot.T+torch.tensor([3.,-2.]);anchor[:,:,:2]=torch.tensor([3.,-2.])
    q[:,:,2]=math.sin(yaw/2);q[:,:,3]=math.cos(yaw/2)
    assert torch.allclose(base,relative_xy(points,anchor,q),atol=1e-6)
    print('literal_lift_and_event_symmetry_contract_passed:20ms_jitter,held_air,hard_interval,reset,offset_slew,acceptance_no_early_death_or_standing,mirrorXY,asynchronous_events,latched_takeoff,stale_pair,rigid_frame_invariance')


if __name__=='__main__':run_contract()
