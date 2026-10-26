"""Fail-closed, multi-seed acceptance from independent physical measurements."""
import argparse
import json
import math
from pathlib import Path
from evaluation_scenarios import scenarios
from run_io import atomic_json, sha256


def number(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def assess(stage, reports, baseline):
    failures = []
    limits = baseline['new_capability_limits']
    required_seeds = set(limits['evaluation_seeds'])
    if set(r.get('seed') for r in reports) != required_seeds or len(reports) != len(required_seeds):
        failures.append('exactly_three_declared_independent_seeds_required')
    if len({r.get('checkpoint_sha256') for r in reports}) != 1:
        failures.append('reports_must_use_one_checkpoint')
    expected = {name for name, _ in scenarios(stage)}
    for report in reports:
        prefix = 'seed%s:'%report.get('seed')
        if report.get('teacher_sha256') != baseline['teacher_sha256']:
            failures.append(prefix+'teacher_provenance_mismatch')
        if len(str(report.get('checkpoint_sha256',''))) != 64:
            failures.append(prefix+'missing_checkpoint_hash')
        if report.get('teacher_control') or report.get('reference_execution_calls', 1) != 0:
            failures.append(prefix+'reference_execution_dependency')
        if report.get('num_envs', 0) < limits['environments_per_seed']:
            failures.append(prefix+'insufficient_environments')
        if report.get('horizon_s', 0) < (28. if stage == 'S5' else 16.):
            failures.append(prefix+'insufficient_horizon')
        if report.get('suite') != stage:
            failures.append(prefix+'wrong_suite')
        rows = {s['name']:s for s in report.get('scenarios', [])}
        if set(rows) != expected:
            failures.append(prefix+'missing_or_unexpected_scenarios')
        for name, row in rows.items():
            key = prefix+name+':'
            if row.get('survival_fraction', 0) < .9:
                failures.append(key+'survival')
            xy = row.get('xy_rmse_m_s')
            limit = baseline['regression_limits']['xy_rmse_max_m_s'] if stage == 'S2' else limits['xy_rmse_m_s']
            if not number(xy) or xy > limit:
                failures.append(key+'xy_tracking')
            if stage == 'S2' and name.startswith('forward'):
                value = row.get('forward_rmse_m_s')
                if not number(value) or value > baseline['regression_limits']['forward_rmse_max_m_s']:
                    failures.append(key+'forward_regression')
            if stage != 'S2':
                yaw = row.get('yaw_rmse_rad_s')
                if not number(yaw) or yaw > limits['yaw_rmse_rad_s']:
                    failures.append(key+'yaw_tracking')
                if name not in ('stop','start_stop'):
                    if row.get('four_feet_stepping_fraction',0.) < .9:
                        failures.append(key+'insufficient_actual_four_foot_steps')
                    # New velocity skills must walk, not pass by sliding with
                    # planted feet. Frequency-switch segments have their own gate.
                    if not name.startswith('frequency'):
                        measured = row.get('steady_frequency_per_foot_hz',[])
                        if len(measured)!=4 or any(not number(f) or abs(f-1.)>.15 for f in measured):
                            failures.append(key+'default_actual_cadence')
            desired, actual = row.get('mean_command'), row.get('mean_actual_xyz_command_units')
            if desired is None or actual is None:
                failures.append(key+'missing_motion_samples')
                continue
            for axis in range(3):
                # Heading correction in S2 is an internal controller, not a new yaw task.
                if stage == 'S2' and axis == 2:
                    continue
                if abs(desired[axis]) >= (.02 if axis < 2 else .1):
                    gain = actual[axis]/desired[axis]
                    if not number(gain) or gain < limits['mean_command_gain_min']:
                        failures.append(key+'axis%d_insufficient_directional_response'%axis)
        if stage == 'S5':
            freq = {(r['name'], r.get('segment_start_s')):r for r in report.get('frequency_scenarios', [])}
            wanted = [('frequency08',0.,.8), ('frequency12',0.,1.2),
                      ('frequency_switch',0.,.8), ('frequency_switch',8.,1.2), ('frequency_switch',20.,.8)]
            for name, segment, target in wanted:
                row = freq.get((name,segment), {})
                by_foot = {f['foot']:f for f in row.get('feet',[])}
                for foot in range(4):
                    f = by_foot.get(foot,{})
                    measured = f.get('measured_frequency_hz')
                    if f.get('interval_count',0) < 3 or f.get('environments',0) < 2 or not number(measured) or abs(measured-target)>limits['frequency_error_hz']:
                        failures.append(prefix+'%s_segment%g_foot%d:actual_frequency'%(name,segment,foot))
            for foot in range(4):
                measured = []
                for segment in (0.,8.,20.):
                    by_foot = {f['foot']:f for f in freq.get(('frequency_switch',segment),{}).get('feet',[])}
                    measured.append(by_foot.get(foot,{}).get('measured_frequency_hz'))
                if all(number(x) for x in measured):
                    if min(measured[1]-measured[0],measured[1]-measured[2]) < limits['frequency_switch_difference_min_hz']:
                        failures.append(prefix+'foot%d:insufficient_actual_frequency_switch'%foot)
    return {'stage':stage, 'passed':not failures, 'failures':failures,
            'checkpoint_sha256':reports[0].get('checkpoint_sha256') if reports else None,
            'evaluation_seeds':sorted(r.get('seed',-1) for r in reports),
            'scope':'Three independent seeds, every requested scenario, actual motion and all four foot events'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['S2','S3','S5'], required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--reports', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = assess(a.stage,[json.loads(f.read_text()) for f in a.reports],json.loads(a.baseline.read_text()))
    result['reports'] = [{'path':str(f.resolve()),'sha256':sha256(f)} for f in a.reports]
    atomic_json(a.output,result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
