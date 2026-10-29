"""Optional intermediate-command measurements in the endpoint simulation batch."""
from evaluate_frontier import GOALS


def extend_groups(groups, alphas):
    expanded = list(groups)
    specs = {}
    if len(set(alphas)) != len(alphas):raise ValueError('Duplicate frontier alpha')
    for alpha in alphas:
        if not 0 < alpha <= 1:raise ValueError('Frontier alpha must be in(0,1]')
        for family, goal in GOALS.items():
            name = 'frontier_%s_%g' % (family, alpha)
            command = [(1-alpha)*.105+alpha*goal[0], alpha*goal[1], alpha*goal[2], 1.]
            specs[name] = (family, alpha)
            expanded.append((name, command))
    return expanded, specs


def split_frontier(result, specs):
    """Keep the complete endpoint suite unchanged and expose a separate gate report.

    Only genuine measured events produce frequencies. No desired clock is used
    as a measurement, and missing feet retain None rather than a default1Hz.
    """
    if not specs:return
    groups = []
    rewards = {row['name']: row.get('values', {}) for row in result['weighted_rewards_per_second']}
    for row in result['scenarios']:
        name = row['name']
        if name not in specs:continue
        family, alpha = specs[name]
        frequency, counts = [], []
        for foot in range(4):
            intervals = [event['interval_s'] for event in result['events']
                if event['scenario'] == name and event['foot'] == foot and event.get('interval_s', 0)]
            counts.append(len(intervals))
            frequency.append(len(intervals)/sum(intervals) if intervals else None)
        groups.append(dict(family=family, alpha=alpha, actual_command=row['initial_command'][:3],
            survival_fraction=row['survival_fraction'], xy_rmse_m_s=row['xy_rmse_m_s'],
            yaw_rmse_rad_s=row['yaw_rmse_rad_s'], mean_actual=row['mean_actual_xyz_command_units'],
            steps_per_foot=row['steps_per_foot'], minimum_steps_per_foot=row['minimum_completed_steps_for_motion_check'],
            four_foot_stepping_fraction=row['four_feet_stepping_fraction'],
            actual_frequency_from_mean_interval_hz=frequency, interval_count_per_foot=counts,
            weighted_rewards_per_second=rewards.get(name, {}),
            clearance_p95_per_foot_m=result['diagnostics']['scenarios'][name]['foot_clearance_p95_per_foot_m']))
    frontier = dict(checkpoint=result['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'],
        seed=result['seed'], horizon_s=result['horizon_s'], environments_per_group=result['environments_per_scenario'],
        groups=groups, reference_execution_calls=result['reference_execution_calls'],
        task_acceptance=False, training_modified=False,
        scope='Intermediate direct-start commands in the same simulation batch as full endpoints. Original episodes and genuine2mm/60ms/5mm landing events only. Progression evidence, never full S3 acceptance.')
    result['simulation_environments'] = result['num_envs']
    result['scenarios'] = [row for row in result['scenarios'] if row['name'] not in specs]
    result['num_envs'] = len(result['scenarios'])*result['environments_per_scenario']
    result['survival_fraction'] = sum(row['survival_fraction'] for row in result['scenarios'])/len(result['scenarios'])
    result['events'] = [event for event in result['events'] if event['scenario'] not in specs]
    result['velocity_segments'] = [row for row in result['velocity_segments'] if row['name'] not in specs]
    result['weighted_rewards_per_second'] = [row for row in result['weighted_rewards_per_second'] if row['name'] not in specs]
    result['diagnostics']['scenarios'] = {name: value for name, value in result['diagnostics']['scenarios'].items() if name not in specs}
    result['frontier_curriculum'] = frontier
