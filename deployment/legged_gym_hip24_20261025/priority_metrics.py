"""Post-settling diagnostics, measured before reset, for the four user priorities."""
import math
import numpy as np


class PriorityMetrics:
    def __init__(self, dof_names, body_names, num_envs, dt):
        self.dof_names = dof_names
        self.body_names = body_names
        self.num_envs = num_envs
        self.dt = dt
        self.rows = [[] for _ in range(num_envs)]
        self.air = np.zeros((num_envs, 4))
        self.max_air = np.zeros_like(self.air)
        self.strides = [[[] for _ in range(4)] for _ in range(num_envs)]
        self.knees = [body_names.index(n) for n in ['L_knee_pitch', 'R_knee_pitch']]
        self.torso = [body_names.index(n) for n in ['ROOT', 'spine_pitch']]
        self.knee_housings = [body_names.index(n) for n in ['L_hip_pitch', 'R_hip_pitch']]
        self.elbow_housings = [body_names.index(n) for n in ['L_shoulder_pitch', 'R_shoulder_pitch']]

    def observe(self, t, active, q, body_xyz, body_force, clearance, foot_force,
                qualified_landing, travel):
        if t <= 1.:
            return
        self.air = np.where((clearance >= .002) & (foot_force <= 1.) & active[:, None],
                            self.air + self.dt, 0.)
        self.max_air = np.maximum(self.max_air, self.air)
        for i in np.flatnonzero(active):
            self.rows[i].append((q[i].copy(), body_xyz[i].copy(),
                                 body_force[i].copy(), clearance[i].copy()))
            for foot in np.flatnonzero(qualified_landing[i]):
                self.strides[i][foot].append(float(travel[i, foot]))

    def result(self):
        samples = []
        for i, rows in enumerate(self.rows):
            if not rows:
                samples.append({'env': i, 'samples_after_1s': 0})
                continue
            q = np.asarray([r[0] for r in rows]) * 180 / math.pi
            xyz = np.asarray([r[1] for r in rows])
            force = np.asarray([r[2] for r in rows])
            clearance = np.asarray([r[3] for r in rows])
            hip = [self.dof_names.index(n) for n in ['L_hip_pitch', 'R_hip_pitch']]
            elbow = [self.dof_names.index(n) for n in ['L_elbow_pitch', 'R_elbow_pitch']]
            samples.append({
                'env': i, 'samples_after_1s': len(rows),
                'knee_housing_contact_fraction': (force[:, self.knee_housings] > 1.).mean(0).tolist(),
                'elbow_housing_contact_fraction': (force[:, self.elbow_housings] > 1.).mean(0).tolist(),
                'torso_contact_fraction': (force[:, self.torso] > 2.).mean(0).tolist(),
                'knee_center_z_p05_m': np.percentile(xyz[:, self.knees, 2], 5, axis=0).tolist(),
                'hip_max_deg': q[:, hip].max(0).tolist(),
                'hip_near_upper_limit_fraction': (q[:, hip] >= 22.).mean(0).tolist(),
                'elbow_rom_p95_p05_deg': (np.percentile(q[:, elbow], 95, axis=0) - np.percentile(q[:, elbow], 5, axis=0)).tolist(),
                'foot_clearance_p95_m': np.percentile(clearance, 95, axis=0).tolist(),
                'foot_clearance_peak_m': clearance.max(0).tolist(),
                'max_continuous_clear_air_s': self.max_air[i].tolist(),
                'strict_step_travel_median_m': [float(np.median(s)) if s else None for s in self.strides[i]],
            })
        valid = [r for r in samples if r['samples_after_1s']]
        median_keys = ['knee_housing_contact_fraction', 'elbow_housing_contact_fraction', 'torso_contact_fraction', 'knee_center_z_p05_m',
                       'hip_near_upper_limit_fraction', 'elbow_rom_p95_p05_deg',
                       'foot_clearance_p95_m', 'foot_clearance_peak_m', 'max_continuous_clear_air_s']
        medians = {k: np.median([s[k] for s in valid], axis=0).tolist() for k in median_keys} if valid else {}
        return {'schema': 'sable-priority-diagnostics/1', 'settling_excluded_s': 1.,
                'joint_pair_order': ['L', 'R'], 'foot_order': ['LF', 'RF', 'LH', 'RH'],
                'body_contact_scope': 'Knee motor meshes are on hip_pitch rigid bodies; elbow meshes are on shoulder_pitch bodies. Net force may include self-contact; not a direct kneeling classifier.',
                'knee_contact_bodies': ['L_hip_pitch', 'R_hip_pitch'],
                'elbow_contact_bodies': ['L_shoulder_pitch', 'R_shoulder_pitch'],
                'range_scope': 'Per-environment p95-p05 over actual joint samples, then median across environments.',
                'median_across_environments': medians, 'samples': samples}
