"""Per-environment integration of substep-averaged propulsion diagnostics."""
import numpy as np


class PropulsionMetrics:
    def __init__(self, count, dt):
        self.dt = dt
        self.seconds = np.zeros(count)
        self.integrals = {name: np.zeros((count, 4)) for name in
                          ['power', 'positive', 'braking', 'reward_per_foot', 'contact']}
        self.trace = []

    def observe(self, t, active, values):
        if t < 1.:
            return
        self.seconds[active] += self.dt
        arrays = {name: value.detach().cpu().numpy() for name, value in values.items()}
        for name, accumulator in self.integrals.items():
            accumulator[active] += arrays[name][active] * self.dt
        if active[0]:
            self.trace.append(dict(t=t, **{name: value[0].tolist() for name, value in arrays.items()}))

    def result(self, commands, scale):
        samples = []
        for i, seconds in enumerate(self.seconds):
            samples.append(dict(command_m_s=float(commands[i]), observed_seconds=float(seconds),
                net_contribution_j=self.integrals['power'][i].tolist(),
                positive_contribution_j=self.integrals['positive'][i].tolist(),
                braking_contribution_j=self.integrals['braking'][i].tolist(),
                mean_power_w=(self.integrals['power'][i]/max(seconds,1e-9)).tolist(),
                mean_positive_w=(self.integrals['positive'][i]/max(seconds,1e-9)).tolist(),
                mean_braking_w=(self.integrals['braking'][i]/max(seconds,1e-9)).tolist(),
                reward_per_second_per_foot=(scale*self.integrals['reward_per_foot'][i]/max(seconds,1e-9)).tolist(),
                support_fraction=(self.integrals['contact'][i]/max(seconds,1e-9)).tolist()))
        groups = []
        for command in sorted(set(commands)):
            rows = [s for s in samples if s['command_m_s'] == command]
            groups.append({'command_m_s':float(command), 'count':len(rows),
                **{name: np.mean([s[name] for s in rows],axis=0).tolist() for name in
                   ['mean_power_w','mean_positive_w','mean_braking_w','reward_per_second_per_foot','support_fraction']}})
        return {'definition':'P_i = F_on_foot_world_xy dot ROOT_velocity_world_xy; positive along actual body velocity, negative opposed. This is a body-propulsion proxy, not exact friction/contact-point work or motor work.',
                'sampling':'40 physical steps at0.5ms per20ms control step; foot-only solver force sensors in world frame, forward dynamics excluded; positive/negative split before averaging; first1s excluded',
                'contact_gate':'upward force>1N and actual capsule bottom<=1mm',
                'foot_order':['LF','RF','LH','RH'], 'reward_weight_per_second':scale,
                'groups':groups,'samples':samples,'trace':self.trace}
