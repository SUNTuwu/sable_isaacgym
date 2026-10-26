"""Stage changes are explicit and gated by measured independent evaluations."""
import torch

STAGES = ('S1', 'S2', 'S3', 'S5')


def sample_velocity_commands(commands, ids, stage, ranges):
    n = len(ids)
    if not n:
        return
    def uniform(low, high):
        return low + (high - low) * torch.rand(n, device=commands.device)
    if stage in ('S1', 'S2'):
        commands[ids, 0] = uniform(.09, .12)
        commands[ids, 1:3] = 0.
        commands[ids[torch.rand(n, device=commands.device) < .1], :3] = 0.
        commands[ids, 5] = 1.
        return
    # 25% retain teacher-domain forward motion, on-policy, for regression.
    family = torch.rand(n, device=commands.device)
    commands[ids, 0] = uniform(*ranges['vx'])
    commands[ids, 1] = uniform(*ranges['vy'])
    commands[ids, 2] = uniform(*ranges['yaw_rate'])
    keep = family < .25
    commands[ids[keep], 0] = uniform(.09, .12)[keep]
    commands[ids[keep], 1:3] = 0.
    commands[ids[(family >= .25) & (family < .35)], :3] = 0.
    lateral = (family >= .35) & (family < .5)
    commands[ids[lateral], 0] = 0.
    commands[ids[lateral], 2] = 0.
    turn = (family >= .5) & (family < .65)
    commands[ids[turn], :2] = 0.
    commands[ids, 5] = uniform(*ranges['frequency']) if stage == 'S5' else 1.


def stage_gate(stage, report, baseline):
    failures = []
    if report.get('survival_fraction', 0.) < .9:
        failures.append('survival_below90percent')
    if report.get('reference_execution_calls', 1) != 0:
        failures.append('reference_execution_dependency')
    if stage == 'S2':
        allowed = max(baseline['velocity_rmse_m_s'] * 1.1, baseline['velocity_rmse_m_s'] + .003)
        if report.get('velocity_rmse_m_s', float('inf')) > allowed:
            failures.append('old_domain_velocity_regression')
    elif stage in ('S3', 'S5'):
        # Predeclared SI thresholds; per-scenario pass is required, not only mean.
        for scenario in report.get('scenarios', []):
            if scenario['survival_fraction'] < .9:
                failures.append(scenario['name'] + ':survival')
            if scenario['xy_rmse_m_s'] > .035:
                failures.append(scenario['name'] + ':xy_tracking')
            if scenario['yaw_rmse_rad_s'] > .30:
                failures.append(scenario['name'] + ':yaw_tracking')
        if not report.get('scenarios'):
            failures.append('missing_velocity_scenarios')
        if stage == 'S5':
            for scenario in report.get('frequency_scenarios', []):
                if scenario.get('measured_frequency_hz') is None or abs(scenario['measured_frequency_hz'] - scenario['command_frequency_hz']) > .15:
                    failures.append(scenario['name'] + ':actual_touchdown_frequency')
            if not report.get('frequency_scenarios'):
                failures.append('missing_frequency_scenarios')
    return {'stage': stage, 'passed': not failures, 'failures': failures}
