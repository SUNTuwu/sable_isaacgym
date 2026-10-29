"""Read-only, stdlib summary of the final evaluator's manifest or one final report.

Usage: python final_acceptance_summary_20261029.py /path/final_evaluation_manifest.json
Reports are read beside the supplied manifest: <teacher_id>/latest_evaluation.json.
Output is JSON on stdout. No files, training state, or acceptance flags are changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

FEET = ('FL', 'FR', 'HL', 'HR')  # action_schema.FOOT_LABELS; actual producer order
FIELDS = ('survival_fraction', 'mean_command', 'mean_actual', 'velocity_frame',
          'xy_rmse_m_s', 'yaw_rmse_rad_s', 'maximum_cross_track_m',
          'max_abs_final_heading_deg', 'max_turn_radius_m', 'pose_symmetry_rmse_m',
          'contact_phase_agreement', 'gravity_rmse_deg', 'foot_rmse_m_per_foot',
          'actual_liftoffs_per_foot', 'pair_liftoff_count_asymmetry',
          'mean_liftoff_period_s_per_foot', 'cycle_symmetry_scope', 'turn_cycles',
          'task_boundary_pass_fraction', 'natural_timeout_fraction',
          'first_episode_samples', 'num_envs', 'horizon_s', 'executed_simulation_s',
          'accepted', 'evaluation_limits')


def read(path):
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def summarize_report(path, selected=None):
    report, digest = read(path)
    if report.get('mode') != 'final_evaluation_only' or report.get('horizon_s') != 24.:
        raise ValueError('Refusing non-final or non-24-second report: ' + str(path))
    source = report.get('source_checkpoint')
    if selected is not None:
        if report.get('teacher_id') != selected['teacher_id'] or not isinstance(source, dict):
            raise ValueError('Report teacher/source checkpoint mismatch: ' + str(path))
        if any(source.get(k) != selected.get(k) for k in ('teacher_id', 'path', 'sha256', 'iteration')):
            raise ValueError('Report does not match manifest checkpoint: ' + str(path))
    result = {key: report.get(key) for key in FIELDS}
    result.update(teacher_id=report.get('teacher_id'), source_checkpoint=source,
                  report_path=str(path), report_sha256=digest,
                  checkpoint_matches_manifest=True if selected is not None else None,
                  missing_fields=[key for key in FIELDS if key not in report])
    periods = report.get('mean_liftoff_period_s_per_foot')
    counts = report.get('actual_liftoffs_per_foot')
    valid_periods = []
    for i in range(4):
        period = periods[i] if isinstance(periods, list) and len(periods) == 4 else None
        count = counts[i] if isinstance(counts, list) and len(counts) == 4 else None
        valid_periods.append(period if number(period) and period > 0 and number(count) and count > 0 else None)
    result['mean_liftoff_period_proxy_s_valid_per_foot'] = valid_periods
    result['left_right_mean_period_absolute_difference_s'] = {
        label: abs(valid_periods[a]-valid_periods[b]) if valid_periods[a] is not None and valid_periods[b] is not None else None
        for label, a, b in (('front_FL_FR', 0, 1), ('hind_HL_HR', 2, 3))}
    return result


def summarize(path):
    path = Path(path).resolve()
    source, digest = read(path)
    notes = [
        'accepted is copied unchanged; this summary does not implement a new acceptance gate.',
        'Foot order is FL, FR, HL, HR. Aggregate metrics cover first episodes in four held-out environments.',
        'Mean liftoff period is the producer proxy: total active environment-time / aggregate liftoff count, not measured successive event intervals.',
        'The producer clamps zero counts to one; the raw value is preserved, but derived valid periods/pair differences are null when either count is zero or missing.',
        'Matching mean periods or counts does not prove matched phase, within-cycle timing, or full gait waveform symmetry.',
        'Cross/radius fields are maximum path excursions; final heading is the maximum absolute terminal integrated yaw over environments.'
    ]
    if source.get('schema') == 'standalone-final-bank-evaluation/v2':
        if source.get('status') != 'completed':
            raise ValueError('Final manifest has not completed')
        if (source.get('horizon_s') != 24. or source.get('actual_simulation_steps') != 1200
                or source.get('first_episode_only') is not True):
            raise ValueError('Manifest is not the complete 24-second first-episode protocol')
        selected = source.get('selected_checkpoints')
        if not isinstance(selected, list) or not selected:
            raise ValueError('Manifest has no explicit selected checkpoints')
        teachers = []
        seen = set()
        for item in selected:
            ident = item['teacher_id']
            if not isinstance(ident, str) or not ident or '/' in ident or '\\' in ident or ident in ('.', '..') or ident in seen:
                raise ValueError('Invalid or duplicate teacher ID')
            seen.add(ident)
            teachers.append(summarize_report(path.parent / ident / 'latest_evaluation.json', item))
        provenance = dict(manifest_path=str(path), manifest_sha256=digest,
                          source_bank=source.get('source_bank'), status=source.get('status'),
                          actual_simulation_steps=source.get('actual_simulation_steps'),
                          total_envs=source.get('total_envs'), evaluation_seed=source.get('evaluation_seed'))
    else:
        teachers = [summarize_report(path)]
        provenance = dict(manifest_path=None, manifest_sha256=None,
                          standalone_report_path=str(path), checkpoint_selection_verification='No manifest supplied')
    return finite_json(dict(schema='final-acceptance-summary/v1', provenance=provenance,
                            foot_order=list(FEET), notes=notes, teachers=teachers))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Final manifest or one final_evaluation_only report')
    args = parser.parse_args()
    print(json.dumps(summarize(args.input), ensure_ascii=False, indent=2, allow_nan=False))
