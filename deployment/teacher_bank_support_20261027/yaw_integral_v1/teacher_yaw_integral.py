"""Signed task-relative yaw integral, exponential cost, and86->87 migration."""
import copy
import math
import torch

SCALE_RAD=math.radians(15.)
METADATA={'definition':'I(t)=integral(actual heading yaw rate-commanded yaw rate)dt, signed and unwrapped; reset each episode',
 'scale_deg':15.,'weight_per_second':2.,'maximum_cost_per_second':8.,
 'formula':'-2*expm1(min(abs(I)/radians(15),log(5))); convex exponential up to24.14deg, numerical cap8/s',
 'observation':'Append one scalar clamp(I/radians(15),-10,10); actor/critic87D,15actions',
 'migration':'Zero-pad actor/critic first-layer weights and their Adam moments. Original86 columns and all other parameters remain exact.'}


class YawIntegral:
    def __init__(self,n,device,dt):
        self.dt=dt;self.value=torch.zeros(n,device=device)
    def reset(self,ids):self.value[ids]=0.
    def update(self,actual,desired):
        self.value+=(actual-desired)*self.dt
        return -2*torch.expm1((self.value.abs()/SCALE_RAD).clamp(max=math.log(5)))
    def observation(self):return (self.value/SCALE_RAD).clamp(-10,10)


def expand_checkpoint(parent,parameter_names):
    state=parent['model_state_dict'];old=state['actor.0.weight'].shape[1]
    if old==87:return parent,None
    assert old==86 and state['critic.0.weight'].shape[1]==86
    adapted=copy.deepcopy(parent);model=adapted['model_state_dict'];optimizer=adapted['optimizer_state_dict']
    names=list(parameter_names);assert len(optimizer['param_groups'])==1
    pids=optimizer['param_groups'][0]['params'];assert len(pids)==len(names)
    widened={'actor.0.weight','critic.0.weight'}
    for name in widened:
        value=model[name];model[name]=torch.cat((value,torch.zeros_like(value[:,:1])),1)
    for pid,name in zip(pids,names):
        if name not in widened:continue
        for key,value in optimizer['state'].get(pid,{}).items():
            if torch.is_tensor(value) and value.ndim==2:
                assert value.shape[1]==86
                optimizer['state'][pid][key]=torch.cat((value,torch.zeros_like(value[:,:1])),1)
    for name,value in state.items():
        new=model[name]
        assert torch.equal(value,new[:,:86] if name in widened else new)
        if name in widened:assert not new[:,86:].any()
    adapted['observations']=87
    return adapted,dict(source_observations=86,target_observations=87,original_columns_equal=True,
                        appended_columns_zero=True,adam_moments_zero_padded=True,initial_outputs_preserved=True)
