import copy
import unittest
from evaluation_frontier import extend_groups, split_frontier
from evaluation_scenarios import scenarios


class FrontierBatchTests(unittest.TestCase):
    def test_optional_extension_preserves_all_endpoint_commands(self):
        base = scenarios('S3')
        saved = copy.deepcopy(base)
        expanded, specs = extend_groups(base, [.375, .5])
        self.assertEqual(base, saved)
        self.assertEqual(expanded[:14], saved)
        self.assertEqual(len(expanded), 24)
        self.assertEqual(len(specs), 10)
        command = dict(expanded)['frontier_reverse_0.375']
        self.assertAlmostEqual(command[0], .050625)
        self.assertEqual(command[1:], [0., 0., 1.])
        with self.assertRaises(ValueError):extend_groups(base, [.5, .5])

    def test_no_frontier_is_exact_report_identity(self):
        report = {'untouched': [1, 2, None]}
        expected = copy.deepcopy(report)
        split_frontier(report, {})
        self.assertEqual(report, expected)

    def test_partition_retains_endpoint_failures_and_missing_foot_events(self):
        extra = 'frontier_reverse_0.375'
        report = dict(checkpoint='model.pt', checkpoint_sha256='test', seed=947,
            horizon_s=16, environments_per_scenario=4, num_envs=8, survival_fraction=.5,
            reference_execution_calls=0,
            scenarios=[dict(name='reverse', survival_fraction=0.),
                dict(name=extra, survival_fraction=1., initial_command=[.050625, 0., 0., 1.],
                    xy_rmse_m_s=.02, yaw_rmse_rad_s=.1, mean_actual_xyz_command_units=[.04, 0., 0.],
                    steps_per_foot=[8, 0, 0, 0], minimum_completed_steps_for_motion_check=7,
                    four_feet_stepping_fraction=0.)],
            events=[dict(scenario='reverse', foot=0, interval_s=2.),
                dict(scenario=extra, foot=0, interval_s=None),
                dict(scenario=extra, foot=0, interval_s=1.),
                dict(scenario=extra, foot=0, interval_s=2.)],
            velocity_segments=[dict(name='reverse'), dict(name=extra)],
            weighted_rewards_per_second=[dict(name='reverse'), dict(name=extra)],
            diagnostics=dict(scenarios={'reverse': {'original': True}, extra:
                {'foot_clearance_p95_per_foot_m': [.01, 0., 0., 0.]}}))
        original = copy.deepcopy(report['scenarios'][0])
        split_frontier(report, {extra: ('reverse', .375)})
        self.assertEqual(report['scenarios'], [original])
        self.assertEqual(report['survival_fraction'], 0.)
        self.assertEqual(report['num_envs'], 4)
        self.assertEqual(report['simulation_environments'], 8)
        self.assertEqual(len(report['events']), 1)
        self.assertEqual(set(report['diagnostics']['scenarios']), {'reverse'})
        row = report['frontier_curriculum']['groups'][0]
        self.assertEqual(row['interval_count_per_foot'], [2, 0, 0, 0])
        self.assertEqual(row['actual_frequency_from_mean_interval_hz'], [2/3, None, None, None])
        self.assertFalse(report['frontier_curriculum']['task_acceptance'])


if __name__ == '__main__':unittest.main()
