import math,unittest,importlib.util
from pathlib import Path
import torch
from teacher_turn_cycles import signed_tracking,TurnCycles
from teacher_reference_rewards_v2 import ReferenceRewards

def state(n=6):
    return dict(normal_force=torch.ones(n,4)*2,clearance=torch.zeros(n,4),phase=torch.zeros(n),
        command=torch.tensor([[.075,0,0],[-.075,0,0],[0,.038,0],[0,-.038,0],[.075,0,0],[-.075,0,0]])[:n],
        velocity=torch.zeros(n,3),foot_local=torch.zeros(n,4,3),reference_local=torch.zeros(n,4,3),
        foot_sigma=torch.tensor([[.04,.04,.02]]*n),velocity_sigma=torch.ones(n)*.03,yaw_sigma=torch.ones(n)*.1,
        target=torch.zeros(n,15),contact_probability=torch.ones(n,4),contact_velocity=torch.zeros(n,4,3),
        gravity=torch.tensor([[0.,0,-1]]*n),reference_gravity=torch.tensor([[0.,0,-1]]*n),height_error=torch.zeros(n),
        limit_excess=torch.zeros(n,15),joint_acceleration=torch.zeros(n,15),normalized_torque=torch.zeros(n,15),fall=torch.zeros(n,dtype=torch.bool))

class Checks(unittest.TestCase):
    def test_signed_reward(self):
        for direction in (1,-1):
            target=torch.tensor([direction*.087266])
            self.assertAlmostEqual(float(signed_tracking(torch.zeros(1),target)),0.,places=5)
            self.assertAlmostEqual(float(signed_tracking(target,target)),6.,places=5)
            self.assertGreater(float(signed_tracking(target*.5,target)),0.)
            self.assertLess(float(signed_tracking(-target,target)),0.)
            self.assertLess(float(signed_tracking(target*3,target)),0.)
    def test_six_translations_identical(self):
        spec=importlib.util.spec_from_file_location('previous_rewards',Path(__file__).with_name('teacher_reference_rewards_before.py'))
        previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
        args=(6,'cpu',.02,torch.ones(6),torch.ones(6,4)*.3,torch.tensor([True,True,False,False,True,True]))
        old=previous.ReferenceRewards(*args);new=ReferenceRewards(*args);torch.manual_seed(20261029)
        for tick in range(100):
            s=state();s.update(velocity=torch.randn(6,3)*.1,foot_local=torch.randn(6,4,3)*.02,
                target=torch.randn(6,15)*.05,normal_force=torch.rand(6,4)*3,clearance=torch.rand(6,4)*.02,
                contact_probability=torch.rand(6,4),phase=torch.full((6,),(tick%50)/50),fall=torch.rand(6)>.9)
            a=old.update(s);b=new.update(s)
            self.assertEqual(set(a),set(b))
            for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)
            if tick==55:old.reset(torch.tensor([1,3]));new.reset(torch.tensor([1,3]))
    def cycle(self,direction=1,motion='good'):
        dt=.02;goal=math.radians(10);command=torch.tensor([[0.,0,direction*goal]])
        c=TurnCycles(1,'cpu',torch.ones(1),torch.ones(1,4));events=[];reports=[]
        for tick in range(151):
            t=tick*dt;phase=(tick%50)/50
            yaw=direction*goal*t
            if motion=='reverse':yaw=-yaw
            if motion=='stationary':yaw=0
            if motion=='oscillate':yaw=goal*math.sin(2*math.pi*t)
            # Force wrap at +-pi to test unwrapped heading increments.
            yaw=math.atan2(math.sin(yaw+3.1),math.cos(yaw+3.1))
            xy=torch.tensor([[.1*t if motion=='drift' else 0.,0.]])
            lifts=torch.full((1,4),tick%50==25,dtype=torch.bool)
            if motion=='chatter':lifts[:]=tick%10==5
            reward=c.update(torch.tensor([yaw]),xy,torch.tensor([phase]),command,torch.ones(1),lifts,torch.tensor([False]),dt)
            if c.diagnostics['complete'][0]:events.append(float(reward)*dt);reports.append(c.diagnostics)
        return c,events,reports
    def test_cycles_and_reset(self):
        for direction in (1,-1):
            c,events,reports=self.cycle(direction)
            self.assertEqual(len(events),2)
            self.assertTrue(all(bool(r['passed'][0]) for r in reports))
            self.assertTrue(all(x>1.99 for x in events))
            c.reset(torch.tensor([0]));self.assertFalse(bool(c.armed[0]))
        for motion in ('reverse','stationary','oscillate','drift','chatter'):
            c,events,reports=self.cycle(motion=motion)
            self.assertTrue(all(not bool(r['passed'][0]) for r in reports),motion)
            if motion=='reverse':self.assertTrue(all(x<0 for x in events))
            if motion=='stationary':self.assertTrue(all(abs(x)<1e-5 for x in events))
    def test_partial_failure_no_bonus(self):
        c=TurnCycles(1,'cpu',torch.ones(1),torch.ones(1,4))
        for phase in (.5,.9,0.,.5,.9,0.):
            rate=c.update(torch.zeros(1),torch.zeros(1,2),torch.tensor([phase]),torch.tensor([[0.,0,.1]]),torch.ones(1),torch.zeros(1,4,dtype=torch.bool),torch.tensor([True]),.02)
            self.assertEqual(float(rate),0.)

if __name__=='__main__':unittest.main(verbosity=2)
