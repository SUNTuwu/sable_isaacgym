import json
from pathlib import Path
import torch
from leg_propulsion import propulsion_terms
from run_io import sha256,atomic_json

force=torch.tensor([[[1.,0.,5.],[-1.,0.,5.],[0.,1.,5.],[0.,0.,5.]]])
velocity=torch.tensor([[.05,0.,0.]])
contact=torch.ones(1,4,dtype=torch.bool)
gate=torch.ones(1);eligible=torch.ones(1,dtype=torch.bool)
power,positive,negative,reward=propulsion_terms(force,velocity,contact,gate,eligible)
checks={'forward_positive_backward_negative':torch.allclose(power,torch.tensor([[.05,-.05,0.,0.]])),
        'signed_leg_reward':torch.allclose(reward,torch.tensor([[1.,-2.,0.,0.]])),
        'braking_pair_cannot_farm_bonus':reward.sum()<0}
rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
p2,*_=propulsion_terms(force@rotation.T,velocity@rotation.T,contact,gate,eligible)
checks['common_world_rotation_invariant']=torch.allclose(power,p2)
p3,*_=propulsion_terms(force,-velocity,contact,gate,eligible)
checks['actual_reverse_motion_reverses_sign']=torch.allclose(power,-p3)
checks['vertical_motion_no_reward']=not propulsion_terms(force,torch.tensor([[0.,0.,1.]]),contact,gate,eligible)[3].any()
checks['flight_no_reward']=not propulsion_terms(force,velocity,~contact,gate,eligible)[3].any()
checks['stationary_no_reward']=not propulsion_terms(force,velocity*0,contact,gate,eligible)[3].any()
checks['parking_or_startup_no_reward']=not propulsion_terms(force,velocity,contact,gate,~eligible)[3].any()
checks['tracking_failure_removes_positive_only']=torch.allclose(propulsion_terms(force,velocity,contact,gate*0,eligible)[3],torch.tensor([[0.,-2.,0.,0.]]))
strong=propulsion_terms(force*10,velocity,contact,gate,eligible)[3]
checks['positive_bounded_negative_uncapped']=strong[0,0]==1 and strong[0,1]==-20
checks['mirror_exchange']=torch.allclose(propulsion_terms(force[:,[1,0,3,2]]*torch.tensor([1.,-1.,1.]),velocity*torch.tensor([1.,-1.,1.]),contact,gate,eligible)[3],reward[:,[1,0,3,2]])
# A sign-changing force within one control interval must not be averaged away.
backward=propulsion_terms(-force,velocity,contact,gate,eligible)
checks['substep_positive_negative_separate']=torch.allclose((power+backward[0])/2,torch.zeros_like(power)) and bool(((reward+backward[3])/2).sum()<0)
integrated_20ms=(reward+backward[3])*.01
integrated_500us=sum([reward*.0005]*20+ [backward[3]*.0005]*20)
checks['integrated_reward_independent_of_sampling_dt']=torch.allclose(integrated_20ms,integrated_500us,atol=1e-7)
here=Path(__file__).resolve().parent
report={'passed':all(checks.values()),'checks':{k:bool(v) for k,v in checks.items()},
        'sources_sha256':{n:sha256(here/n) for n in ['leg_propulsion.py','sable_env.py','sable_config.py','check_leg_propulsion.py']}}
atomic_json(here/'verification/leg_propulsion.json',report)
print(json.dumps(report));assert report['passed']
