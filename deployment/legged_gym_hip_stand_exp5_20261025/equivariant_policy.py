"""Reflection-projected actor/critic using the existing official rsl_rl networks.

The mean is (f(s) + M_a f(M_s s))/2; the value is reflection invariant.
This is group averaging, not a reproduction of a paper's network architecture.
Export/inference must retain this projection: exporting .actor alone is invalid.
"""
import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from policy_symmetry import Mirror,NAMES
from sable_config import MODEL

class SymmetricActorCritic(ActorCritic):
    def __init__(self,num_actor_obs,num_critic_obs,num_actions,**kwargs):
        assert num_actor_obs==num_critic_obs==52 and num_actions==12
        assert MODEL['policy_joint_names']==NAMES
        super().__init__(num_actor_obs,num_critic_obs,num_actions,**kwargs)
        self._mirror=None

    def mirror(self,x):
        if self._mirror is None or self._mirror.oi.device!=x.device:
            # The first call is an inference-mode PPO rollout. Constants must
            # remain ordinary tensors so later training can save them for backward.
            with torch.inference_mode(False):
                self._mirror=Mirror(NAMES,x.shape[-1],x.device)
        return self._mirror

    def act_inference(self,observations):
        m=self.mirror(observations)
        return (self.actor(observations)+m.actions(self.actor(m.observations(observations))))*.5

    def update_distribution(self,observations):
        mean=self.act_inference(observations);m=self.mirror(observations)
        std=(self.std+self.std[m.ai])*.5
        self.distribution=Normal(mean,mean*0.+std)

    def evaluate(self,critic_observations,**kwargs):
        m=self.mirror(critic_observations)
        return (self.critic(critic_observations)+self.critic(m.observations(critic_observations)))*.5

def register_runner():
    import rsl_rl.runners.on_policy_runner as runner
    runner.SymmetricActorCritic=SymmetricActorCritic
