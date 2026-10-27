import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from command_foot_motion import desired_foot_velocity,FootCenterVelocity,foot_velocity_cost


class CommandFootMotionTests(unittest.TestCase):
    def test_cycle_displacement_follows_command_and_frequency(self):
        n=10000;phase=(torch.arange(n)+.5)/n
        command=torch.tensor([.04,-.02,0.]).expand(n,-1)
        for duty in (.35,.5,.75):
            target=desired_foot_velocity(command,torch.tensor([1.,0.]).expand(n,-1),torch.zeros(n,2),
                torch.zeros(n,4,2),phase[:,None].expand(-1,4),torch.full((n,1),duty),torch.ones(n))
            for frequency in (.8,1.,1.2):
                torch.testing.assert_close(target.mean(0)/frequency,command[0,:2].expand(4,-1)/frequency,atol=1e-7,rtol=1e-5)
            self.assertTrue((target[phase<duty]==0.).all())

    def test_yaw_direction_reflection_and_stop(self):
        feet=torch.tensor([[[.1,.03],[.1,-.03],[-.06,.03],[-.06,-.03]]])
        cmd=torch.tensor([[0.,0.,.3]]);phase=torch.full((1,4),.75);duty=torch.full((1,1),.5)
        f=torch.tensor([[1.,0.]]);root=torch.zeros(1,2)
        target=desired_foot_velocity(cmd,f,root,feet,phase,duty,torch.ones(1))
        self.assertLess(float(target[0,0,0]),0);self.assertGreater(float(target[0,1,0]),0)
        self.assertGreater(float(target[0,0,1]),0);self.assertLess(float(target[0,2,1]),0)
        perm=[1,0,3,2];sign=torch.tensor([1.,-1.]);mirror=desired_foot_velocity(cmd*torch.tensor([1.,-1.,-1.]),f*sign,root*sign,
            feet[:,perm]*sign,phase[:,perm],duty,torch.ones(1))
        torch.testing.assert_close(mirror,target[:,perm]*sign)
        stop=desired_foot_velocity(cmd,f,root,feet,phase,duty,torch.zeros(1))
        self.assertTrue((stop==0).all())
        self.assertTrue((foot_velocity_cost(target,target)==0).all())

    def test_finite_difference_excludes_teleport_and_is_per_environment(self):
        velocity=FootCenterVelocity(2,'cpu',.02);centers=torch.zeros(2,4,2)
        self.assertTrue((velocity.update(centers)==0).all());self.assertFalse(velocity.valid.any())
        centers[0,:,0]=.002;centers[1,:,1]=-.001
        measured=velocity.update(centers)
        torch.testing.assert_close(measured[0,:,0],torch.full((4,),.1))
        torch.testing.assert_close(measured[1,:,1],torch.full((4,),-.05))
        velocity.reset(torch.tensor([0]));centers[0]+=100.
        measured=velocity.update(centers)
        self.assertFalse(velocity.valid[0]);self.assertTrue(velocity.valid[1]);self.assertTrue((measured==0).all())


if __name__=='__main__':unittest.main()
