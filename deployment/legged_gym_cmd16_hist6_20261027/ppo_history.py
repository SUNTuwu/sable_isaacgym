"""Existing rsl_rl PPO with explicit terminal bootstrapping and history labels."""
import copy
import torch
from rsl_rl.algorithms import PPO
from observation_schema import mirror_raw, mirror_proprio


def timeout_bootstrap(rewards, timeouts, terminal_values, gamma):
    return rewards + gamma*terminal_values.reshape_as(rewards)*timeouts.to(rewards.dtype)


class HistoryPPO(PPO):
    def __init__(self, *args, auxiliary_learning_rate=5e-5, auxiliary_policy_kl_limit=.01, minimum_action_std=.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimum_action_std=minimum_action_std
        self.auxiliary_optimizer = torch.optim.Adam(self.actor_critic.history_encoder.parameters(), lr=auxiliary_learning_rate)
        self.auxiliary_learning_rate_max = auxiliary_learning_rate
        self.auxiliary_policy_kl_limit = auxiliary_policy_kl_limit
        self.auxiliary_rollout = []
        self.last_auxiliary_metrics = {}

    def observe_auxiliary(self, velocity_before_step, next_clean_proprio, done):
        self.auxiliary_rollout.append({'raw': self.transition.observations.detach().clone(),
            'velocity': velocity_before_step.detach().clone(),
            'next_proprio': next_clean_proprio.detach().clone(), 'done': done.detach().clone()})

    def process_env_step(self, rewards, dones, infos):
        if 'time_outs' not in infos or 'terminal_critic' not in infos:
            raise ValueError('Explicit timeouts and pre-reset terminal critic are required')
        with torch.no_grad():
            terminal_value = self.actor_critic.evaluate(infos['terminal_critic'])
        adjusted = timeout_bootstrap(rewards, infos['time_outs'], terminal_value, self.gamma)
        # Base PPO bootstraps using the pre-action value. Remove that shortcut
        # after adding the actual terminal-state value exactly once.
        safe_infos = {k:v for k,v in infos.items() if k != 'time_outs'}
        super().process_env_step(adjusted, dones, safe_infos)

    def update(self):
        values = super().update()
        with torch.no_grad():
            self.actor_critic.std.clamp_(self.minimum_action_std, .35)
            if self.actor_critic.freeze_waist:
                self.actor_critic.std[-1] = .04
        if not self.auxiliary_rollout:
            return values
        batch = {k:torch.cat([row[k] for row in self.auxiliary_rollout]) for k in self.auxiliary_rollout[0]}
        self.auxiliary_rollout.clear()
        count = min(4096, len(batch['raw']))
        ids = torch.randperm(len(batch['raw']), device=self.device)[:count]
        batch = {k:v[ids] for k,v in batch.items()}
        encoder = self.actor_critic.history_encoder
        weights_before = copy.deepcopy(encoder.state_dict())
        optimizer_before = copy.deepcopy(self.auxiliary_optimizer.state_dict())
        with torch.no_grad():
            mean_before = self.actor_critic.act_inference(batch['raw'])
        for ids in torch.randperm(count, device=self.device).split(1024):
            raw = batch['raw'][ids]
            loss, stats = encoder.auxiliary_loss(raw[:, :306], batch['velocity'][ids], batch['next_proprio'][ids], batch['done'][ids])
            reflected = mirror_raw(raw)
            mirror_loss, _ = encoder.auxiliary_loss(reflected[:, :306], batch['velocity'][ids]*raw.new_tensor([1.,-1.,1.]),
                                                   mirror_proprio(batch['next_proprio'][ids]), batch['done'][ids])
            self.auxiliary_optimizer.zero_grad(set_to_none=True)
            (.5*(loss+mirror_loss)).backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), .5)
            self.auxiliary_optimizer.step()
        with torch.no_grad():
            after = self.actor_critic.act_inference(batch['raw'])
            std = self.actor_critic.std.clamp(min=.02)
            shift = (((after-mean_before).square()/(2*std.square()))*self.actor_critic.execution_mask).sum(-1).mean()
            accepted = bool(torch.isfinite(shift) and shift <= self.auxiliary_policy_kl_limit)
        if not accepted:
            encoder.load_state_dict(weights_before, strict=True)
            self.auxiliary_optimizer.load_state_dict(optimizer_before)
            for group in self.auxiliary_optimizer.param_groups:
                group['lr'] = max(1e-7,group['lr']*.5)
        elif float(shift) < self.auxiliary_policy_kl_limit*.1:
            for group in self.auxiliary_optimizer.param_groups:
                group['lr'] = min(self.auxiliary_learning_rate_max,group['lr']*1.1)
        self.last_auxiliary_metrics = {'encoder_update_accepted': accepted, 'encoder_policy_kl': float(shift),
            'encoder_velocity_mse': float(stats['velocity_mse']), 'encoder_prediction_mse': float(stats['next_proprio_mse']),
            'auxiliary_examples': count, 'auxiliary_learning_rate':self.auxiliary_optimizer.param_groups[0]['lr'],
            'labels': 'same-time ROOT velocity and pre-reset next proprioception; done pairs masked'}
        return values
