"""Frozen v44 query and complete substep targets, never residual-only labels."""
import json
from pathlib import Path
import torch
from equivariant_policy import SymmetricActorCritic
from guided_reference import GuidedReference
from action_schema import ActionSpec
from run_io import sha256


class FrozenTeacher:
    def __init__(self, env, checkpoint, expected_sha256):
        checkpoint = Path(checkpoint)
        if sha256(checkpoint) != expected_sha256:
            raise ValueError('Teacher checkpoint hash mismatch')
        self.env = env
        payload = torch.load(str(checkpoint), map_location=env.device)
        config = json.loads((checkpoint.parent.parent / 'config.json').read_text())
        self.policy = SymmetricActorCritic(58, 58, 14, **config['train']['policy']).to(env.device).eval()
        self.policy.load_state_dict(payload['model_state_dict'], strict=True)
        self.policy.requires_grad_(False)
        self.reference = GuidedReference(env)
        self.previous_residual = torch.zeros(env.num_envs, 14, device=env.device)
        self.actions = self.previous_residual.clone()
        self.query_count = 0
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = expected_sha256
        self.iteration = payload['iter']

    def observation(self):
        e = self.env
        idx = e.active[:14]
        phase = 2 * torch.pi * e.gait_signals.phase(e)
        return torch.cat((e.base_lin_vel * e.obs_scales.lin_vel,
                          e.base_ang_vel * e.obs_scales.ang_vel, e.projected_gravity,
                          e.commands[:, :3] * e.commands_scale,
                          (e.dof_pos[:, idx] - e.default_dof_pos[:, idx]) * e.obs_scales.dof_pos,
                          e.dof_vel[:, idx] * e.obs_scales.dof_vel, self.previous_residual,
                          phase.sin()[:, None], phase.cos()[:, None],
                          (self.reference.speed * e.obs_scales.lin_vel)[:, None],
                          self.reference.ramp(e.episode_length_buf * e.dt)[:, None]), -1)

    @torch.no_grad()
    def prepare(self):
        # Query before prepare updates filtered speed, matching the old controller.
        self.actions = self.policy.act_inference(self.observation()).clamp(-3., 3.)
        self.reference.prepare()
        self.query_count += 1
        return ActionSpec.offsets(self.target(1.)[:, self.env.active],
                                  self.env.default_dof_pos[:, self.env.active])

    def target(self, fraction):
        e = self.env
        target = e.default_dof_pos.expand(e.num_envs, -1).clone()
        target[:, e.active[:14]] += .35 * self.actions
        target += self.reference.start + fraction * (self.reference.end - self.reference.start)
        return torch.maximum(torch.minimum(target, e.hard_limits[:, 1]), e.hard_limits[:, 0])

    def finish(self, executed_target, pure_teacher=False):
        e = self.env
        if pure_teacher:
            self.previous_residual.copy_(self.actions)
        else:
            equivalent = (executed_target[:, :14] - e.default_dof_pos[:, e.active[:14]]
                          - self.reference.end[:, e.active[:14]]) / .35
            self.previous_residual.copy_(equivalent.clamp(-3., 3.))

    def reset(self, ids):
        self.reference.reset(ids)
        self.previous_residual[ids] = 0.
        self.actions[ids] = 0.


def verify_teacher_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if not manifest.get('completion_verified'):
        raise ValueError('Teacher must be complete and its archive verified')
    for item in manifest['files']:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Frozen teacher file changed: ' + item['path'])
    return manifest
