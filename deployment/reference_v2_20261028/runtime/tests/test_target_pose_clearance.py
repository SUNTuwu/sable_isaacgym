import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from target_pose_clearance import target_pose_clearance_cost,executed_target_coordinates


class TargetPoseClearanceTests(unittest.TestCase):
    def test_ground_clamped_pose_still_has_a_local_signal(self):
        z=torch.tensor([[-.003,-.002,-.001,0.]])
        desired=torch.full_like(z,.023)
        cost=target_pose_clearance_cost(z,desired,torch.ones_like(z),desired)
        self.assertTrue((cost[:,:-1]>cost[:,1:]).all())
        self.assertTrue((cost>=1.).all())

    def test_stance_stop_and_excess_height_do_not_pay(self):
        z=torch.tensor([[.0,.03,.023,-.01]])
        desired=torch.full_like(z,.023)
        weight=torch.tensor([[0.,1.,1.,0.]])
        self.assertTrue((target_pose_clearance_cost(z,desired,weight,desired)==0).all())

    def test_scale_mirror_and_finite_cap(self):
        z=torch.tensor([[-.001,.007,.011,.03]])
        h=torch.tensor([[.023]]);w=torch.tensor([[.2,.7,.6,1.]])
        cost=target_pose_clearance_cost(z,h,w,h)
        torch.testing.assert_close(target_pose_clearance_cost(z*2,h*2,w,h*2),cost)
        p=[1,0,3,2]
        torch.testing.assert_close(target_pose_clearance_cost(z[:,p],h,w[:,p],h),cost[:,p])
        self.assertTrue((target_pose_clearance_cost(z-100,h,w,h)<=9.).all())

    def test_target_copy_preserves_physics_and_opposed_coupling(self):
        q=torch.arange(12,dtype=torch.float32).reshape(2,6);saved=q.clone()
        active=torch.tensor([0,2,4]);target=torch.tensor([[.1,.2,.4],[-.1,-.2,-.4]])
        hypothetical=executed_target_coordinates(q,active,target,2,3)
        torch.testing.assert_close(q,saved)
        torch.testing.assert_close(hypothetical[:,active],target)
        torch.testing.assert_close(hypothetical[:,3],-target[:,1])
        torch.testing.assert_close(hypothetical[:,[1,5]],saved[:,[1,5]])


if __name__=='__main__':unittest.main()
