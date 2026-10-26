"""Complete 326D raw-input projected policy, with a separate 92D critic."""
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from action_schema import ActionSpec, ACTION_INDEX, ACTION_SIGN
from command_schema import COMMAND_SCALES
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
        self.novel_command_std_floor = 0.

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
        if self.novel_command_std_floor:
            # This is state-conditioned stochastic exploration only. The
            # deterministic exported actor is unchanged. Keep the learned
            # forward gait and true stop at their original noise level.
            vx,vy,yaw=observations[:,306]*.2,observations[:,307]*.1,observations[:,308]
            moving=(torch.sqrt(vx.square()+vy.square())>.01)|(yaw.abs()>.08)
            novel=(vx<.03)&moving
            std=torch.where(novel[:,None],std.clamp(min=self.novel_command_std_floor),std)
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
        self.actor = policy.actor
        self.encoder = policy.history_encoder
        # Registered buffers follow map_location / .to(device). Python
        # new_tensor constants in a traced function otherwise pin the CPU.
        self.register_buffer('execution_mask',policy.execution_mask.detach().clone())
        self.register_buffer('action_index',torch.tensor(ACTION_INDEX,dtype=torch.long))
        self.register_buffer('action_sign',torch.tensor(ACTION_SIGN,dtype=torch.float32))
        self.register_buffer('proprio_index',torch.tensor(list(range(6))+
            [offset+i for offset in (6,21,36) for i in ACTION_INDEX],dtype=torch.long))
        self.register_buffer('proprio_sign',torch.tensor([-1.,1.,-1.,1.,-1.,1.]+list(ACTION_SIGN)*3))
        self.register_buffer('command_scales',torch.tensor(COMMAND_SCALES))
        self.register_buffer('command_sign',torch.tensor([1.,-1.,-1.,1.,1.,1.,1.,1.,1.,1.,1.,-1.,-1.,1.,1.,1.]))
        self.register_buffer('clock_index',torch.tensor([1,0,3,2],dtype=torch.long))

    def branch(self,raw):
        velocity,latent=self.encoder(raw[:,:306])
        return self.actor(torch.cat((raw[:,:51],raw[:,306:322],raw[:,322:326],velocity*2.,latent),-1))

    def forward(self, raw):
        history=raw[:,:306].reshape(-1,6,51)
        reflected_history=history.index_select(2,self.proprio_index)*self.proprio_sign
        command=raw[:,306:322]*self.command_scales
        reflected_command=command*self.command_sign
        fr,hl,hr=command[:,6],command[:,7],command[:,8]
        reflected_command[:,6]=torch.remainder(-fr,1.)
        reflected_command[:,7]=torch.remainder(hr-fr,1.)
        reflected_command[:,8]=torch.remainder(hl-fr,1.)
        reflected=torch.cat((reflected_history.flatten(1),reflected_command/self.command_scales,
            raw[:,322:326].index_select(1,self.clock_index)),-1)
        return .5*(self.branch(raw)+self.branch(reflected).index_select(1,self.action_index)*self.action_sign)*self.execution_mask
