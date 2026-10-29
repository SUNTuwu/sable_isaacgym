"""Exercise the real reward adapter with overshoot before and at timeout."""
import sys,unittest
sys.path.insert(0,sys.argv[1])
import torch
from types import SimpleNamespace
import check_reference_v2 as suite
original=suite.check_env
checked=0
def verify(e):
    global checked
    original(e)
    e.turn_boundary=SimpleNamespace(evaluate=lambda env:torch.ones(env.num_envs,dtype=torch.bool),angle=torch.ones(e.num_envs),displacement=torch.ones(e.num_envs,2))
    saved=e.episode_length_buf.clone()
    e.episode_length_buf.fill_(1)
    original(e)
    assert not e.reset_buf.any(),'Transient overshoot terminated an episode'
    assert not e.teacher_terms['v2_task_boundary_event'].any(),'Boundary charged before natural end'
    e.episode_length_buf.fill_(e.max_episode_length)
    original(e)
    assert e.reset_buf.all() and e.time_out_buf.all() and not e.true_fall.any()
    assert torch.allclose(e.teacher_terms['v2_task_boundary_event']*e.dt,torch.full((e.num_envs,),-10.))
    e.episode_length_buf.copy_(saved);del e.turn_boundary;original(e);checked+=1
suite.check_env=verify
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(suite.RewardsTest))
assert result.wasSuccessful()
assert checked>=8
print('Verified terminal-only overshoot settlement on',checked,'physical-reference adapter cases')
