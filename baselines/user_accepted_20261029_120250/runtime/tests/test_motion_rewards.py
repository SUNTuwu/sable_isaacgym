import unittest
import torch
from motion_rewards import yaw_error_cost,angular_tracking_reward


class AngularRewardTests(unittest.TestCase):
    def test_zero_correct_sign_and_overshoot_cost(self):
        command=torch.tensor([.35,-.35,.35,.35])
        actual=torch.tensor([.35,-.35,0.,-.35])
        self.assertTrue(torch.equal(yaw_error_cost(actual-command,.35),torch.tensor([0.,0.,1.,4.])))

    def test_cost_normalization_does_not_change_positive_tracking(self):
        error=torch.tensor([0.,.13125,-.13125,.35,-.7])
        expected=torch.exp(-error.square()/.25)
        self.assertTrue(torch.equal(angular_tracking_reward(error),expected))
        self.assertTrue(torch.equal(yaw_error_cost(error,1.),error.square()))
        self.assertTrue(torch.allclose(yaw_error_cost(error,.35)*(.35**2),error.square()))
        self.assertTrue(torch.equal(yaw_error_cost(error,.35),yaw_error_cost(-error,.35)))

    def test_invalid_scales_are_rejected(self):
        for scale in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):yaw_error_cost(torch.ones(2),scale)


if __name__=='__main__':unittest.main()
