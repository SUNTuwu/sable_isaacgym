"""Explicit joint-name-preserving 52/12 -> 58/14 transfer, new inputs/outputs zero."""
import torch
OLD_TO_NEW=list(range(12))+list(range(12,24))+list(range(26,38))+list(range(40,52))+list(range(54,58))
def transfer_state(old,new):
    for key,value in old.items():
        if new[key].shape==value.shape:new[key]=value.clone()
        elif key in ['actor.0.weight','critic.0.weight'] and value.shape[1]==52 and new[key].shape[1]==58:
            new[key].zero_();new[key][:,OLD_TO_NEW]=value
        elif key in ['actor.6.weight','actor.6.bias','std'] and value.shape[0]==12 and new[key].shape[0]==14:
            new[key].zero_();new[key][:12]=value
        else:raise ValueError('Unsupported transfer: '+key)
    return new
