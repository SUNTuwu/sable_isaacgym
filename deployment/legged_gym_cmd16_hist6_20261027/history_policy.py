"""Complete 326D raw-input projected policy, with a separate 92D critic."""
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from action_schema import ActionSpec
from observation_schema import mirror_raw, mirror_critic
from history_encoder import HistoryEncoder


class HistoryActorCritic(ActorCritic):
    def __init__(self, num_actor_obs=326, num_critic_obs=92, num_actions=15,
                 actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128),
                 init_noise_std=.04, freeze_waist=True, **kwargs):
        if (num_actor_obs, num_critic_obs, num_actions) != (326, 92, 15):
            raise ValueError('Expected raw326 / critic92 / actions15; internal actor input is90')
        super().__init__(90, 92, 15, actor_hidden_dims=list(actor_hidden_dims),
                         critic_hidden_dims=list(critic_hidden_dims), init_noise_std=init_noise_std,
                         **kwargs)
        self.history_encoder = HistoryEncoder()
        self.register_buffer('execution_mask', torch.tensor([1.] * 14 + [0. if freeze_waist else 1.]))
        self.freeze_waist = freeze_waist

    def branch(self, raw):
        velocity, latent = self.history_encoder(raw[:, :306])
        actor_input = torch.cat((raw[:, :51], raw[:, 306:322], raw[:, 322:326],
                                 velocity.detach() * 2., latent.detach()), -1)
        return self.actor(actor_input)

    def act_inference(self, observations):
        return .5 * (self.branch(observations) + ActionSpec.mirror(self.branch(mirror_raw(observations)))) * self.execution_mask

    def forward(self, observations):
        return self.act_inference(observations)

    def update_distribution(self, observations):
        mean = self.act_inference(observations)
        # Positive even for frozen dimensions. Frozen dimensions do not contribute
        # to policy likelihood, entropy, executed action, or exploration.
        permutation = [0, 1, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 13, 12, 14]
        std = ((self.std + self.std[permutation]) * .5).clamp(min=1e-4)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample() * self.execution_mask

    def get_actions_log_prob(self, actions):
        return (self.distribution.log_prob(actions) * self.execution_mask).sum(-1)

    @property
    def entropy(self):
        return (self.distribution.entropy() * self.execution_mask).sum(-1)

    def evaluate(self, critic_observations, **kwargs):
        return .5 * (self.critic(critic_observations) + self.critic(mirror_critic(critic_observations)))

    def set_waist_enabled(self, enabled):
        self.freeze_waist = not enabled
        self.execution_mask[-1] = float(enabled)

    def initialization_report(self, teacher_state):
        # The architecture and full-target semantics changed. Randomly initialize
        # all student tensors and transfer behavior via explicit supervised labels.
        return {'method': 'behavior_distillation_no_weight_reuse', 'reused': [],
                'skipped_teacher_keys': sorted(teacher_state),
                'new_student_keys': sorted(self.state_dict()),
                'reason': '58D residual teacher and326D history full-target student differ in inputs and outputs'}


class DeploymentPolicy(nn.Module):
    """Export includes history encoding, both mirror branches, and waist mask."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, raw):
        return self.policy.act_inference(raw)
