"""World-frame geometry, literal boundaries and reset/turn exclusions."""
import math
import torch
from teacher_translation_guard import TranslationGuard


def roots(yaw,n):
    r=torch.zeros(n,13);r[:,5]=math.sin(yaw/2);r[:,6]=math.cos(yaw/2);return r


def run_contract():
    command=torch.tensor([[.1,0,0],[-.1,0,0],[0,.1,0],[0,-.1,0],[0,0,.785]])
    n=len(command);r=roots(.7,n);r[:,:2]=torch.tensor([123.,-456.])
    g=TranslationGuard(n,'cpu',command);g.reset(torch.arange(n),r)
    assert not g.evaluate(r).any() and not g.observations(r).any()
    r[:,:2]+=g.axis*10.;assert not g.evaluate(r).any(),'Along-path progress must not trigger'
    normal=torch.stack((-g.axis[:,1],g.axis[:,0]),1)
    r[:,:2]+=normal*.099;assert not g.evaluate(r).any()
    r[:,:2]+=normal*.002;assert g.evaluate(r)[:4].all() and not g.evaluate(r)[4]
    assert torch.allclose(g.cross_track[:4],torch.ones(4)*.101,atol=5e-5)
    # Exactly the bound is allowed; next control-sample excess fails.
    r=roots(0,n);g.reset(torch.arange(n),r)
    r[:,1]=.10;assert not g.evaluate(r)[:2].any()
    r[:,1]=.1002;assert g.evaluate(r)[:2].all()
    r=roots(0,n);g.reset(torch.arange(n),r);r=roots(math.radians(10),n)
    assert not g.evaluate(r).any()
    r=roots(math.radians(10.01),n);assert g.evaluate(r)[:4].all() and not g.evaluate(r)[4]
    r=roots(math.radians(-10.01),n);assert g.evaluate(r)[:4].all()
    # Wrap across +/-pi represents a2deg change, not a358deg change.
    r=roots(math.radians(179),n);g.reset(torch.arange(n),r)
    assert not g.evaluate(roots(math.radians(-179),n)).any()
    assert abs(math.degrees(float(g.heading_error[0]))-2)<1e-4
    # Pitch/roll and vertical height are not planar drift limits.
    r=roots(0,n);g.reset(torch.arange(n),r);r[:,3]=math.sin(.2);r[:,6]=math.cos(.2);r[:,2]=2
    assert not g.evaluate(r).any()
    g=TranslationGuard(n,'cpu',command,enabled=False);g.reset(torch.arange(n),r)
    r[:,:2]+=2;assert not g.evaluate(r).any() and not g.observations(r).any()
    print('translation_guard_contract_passed: all4XYdirections, fixed_initial_axis, absolute10deg_and10cm_boundaries, along_track_exempt, angle_wrap, height_roll_exempt, reset, turns_exempt')


if __name__=='__main__':run_contract()
