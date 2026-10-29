import copy
import unittest
from queue_continuation import frontier_progress


class FrontierSelectionTests(unittest.TestCase):
    def setUp(self):
        self.baseline={'new_capability_limits':{'xy_rmse_m_s':.035,'yaw_rmse_rad_s':.3,'mean_command_gain_min':.5}}
        self.row=dict(actual_command=[.065625,0.,.13125],mean_actual=[.065625,0.,.02],
            xy_rmse_m_s=.025,yaw_rmse_rad_s=.2,four_foot_stepping_fraction=1.,
            actual_frequency_from_mean_interval_hz=[1.]*4,interval_count_per_foot=[20]*4)

    def score(self,row):
        return frontier_progress({'selected_rows':[row]},self.baseline)['normalized_unmet_sum']

    def test_real_response_improvement_matters_before_binary_gate_pass(self):
        earlier=self.score(self.row)
        improved=copy.deepcopy(self.row);improved['mean_actual'][2]=.04
        self.assertGreater(earlier,self.score(improved))
        self.assertGreater(self.score(improved),0.)
        improved['mean_actual'][2]=.066
        self.assertEqual(self.score(improved),0.)

    def test_missing_actual_cadence_never_becomes_zero_deficit(self):
        valid=copy.deepcopy(self.row);valid['mean_actual'][2]=.066
        missing=copy.deepcopy(valid);missing['actual_frequency_from_mean_interval_hz'][2]=None
        self.assertGreater(self.score(missing),self.score(valid))
        stopped=copy.deepcopy(missing);stopped['actual_command']=[0.,0.,0.];stopped['mean_actual']=[0.,0.,0.]
        self.assertEqual(self.score(stopped),0.)


if __name__=='__main__':unittest.main()
