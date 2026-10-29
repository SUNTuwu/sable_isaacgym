"""Stage changes are explicit and gated by measured independent evaluations."""
import torch

STAGES = ('S1', 'S2', 'S3', 'S5')


def command_frontier(goals, alpha):
    """Training command path from learned forward motion to the full goal.

    Old forward and true stop remain unchanged. Alpha1 is exact identity.
    This changes the actual command sent to actor/reward together, never action
    targets. Independent evaluation always bypasses it at alpha1.
    """
    if not 0. <= alpha <= 1.:raise ValueError('Command frontier alpha must be in[0,1]')
    if alpha == 1.:return goals.clone()
    novel=(goals[:,0]<.08)|(goals[:,1:].abs().sum(1)>.01)
    moving=(goals[:,:2].norm(dim=1)>.01)|(goals[:,2].abs()>.08)
    start=torch.zeros_like(goals);start[:,0]=.105
    return torch.where((novel&moving)[:,None],torch.lerp(start,goals,alpha),goals)


class MovingStartCurriculum:
    """Training-only commands; all actions still come from the current student.

    Some new-direction samples begin with learned forward locomotion and then
    cross to their real target. A quarter always start at the requested command.
    Evaluation bypasses this object and retains direct starts from rest.
    """
    def __init__(self, num_envs, device, fraction=.75, hold_s=2., blend_s=2.):
        self.target = torch.zeros(num_envs, 3, device=device)
        self.start = self.target.clone()
        self.sample_time = torch.zeros(num_envs, device=device)
        self.enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.fraction, self.hold_s, self.blend_s = fraction, hold_s, blend_s

    def resample(self, ids, target, age, draws=None):
        self.target[ids] = target
        self.start[ids] = target
        draws = torch.rand(len(ids), device=target.device) if draws is None else draws
        novel = (target[:, 0] < .08) | (target[:, 1:].abs().sum(1) > .01)
        moving = (target[:, :2].norm(dim=1) > .01) | (target[:, 2].abs() > .08)
        selected = novel & moving & (draws < self.fraction)
        chosen = ids[selected]
        self.start[chosen, 0] = .105
        self.start[chosen, 1:] = 0.
        self.enabled[ids] = selected
        self.sample_time[ids] = age

    def value(self, age):
        alpha = ((age-self.sample_time-self.hold_s)/self.blend_s).clamp(0., 1.)
        alpha = alpha.square()*(3.-2.*alpha)
        return torch.lerp(self.start, self.target, alpha[:, None])


def sample_velocity_commands(commands, ids, stage, ranges):
    n = len(ids)
    if not n:
        return
    def uniform(low, high):
        return low + (high - low) * torch.rand(n, device=commands.device)
    if stage in ('S1', 'S2'):
        commands[ids, 0] = uniform(.09, .12)
        commands[ids, 1:3] = 0.
        commands[ids[torch.rand(n, device=commands.device) < .1], :3] = 0.
        commands[ids, 5] = 1.
        return
    # 25% retain teacher-domain forward motion, on-policy, for regression.
    family = torch.rand(n, device=commands.device)
    commands[ids, 0] = uniform(*ranges['vx'])
    commands[ids, 1] = uniform(*ranges['vy'])
    commands[ids, 2] = uniform(*ranges['yaw_rate'])
    keep = family < .25
    commands[ids[keep], 0] = uniform(.09, .12)[keep]
    commands[ids[keep], 1:3] = 0.
    commands[ids[(family >= .25) & (family < .35)], :3] = 0.
    lateral = (family >= .35) & (family < .5)
    commands[ids[lateral], 0] = 0.
    commands[ids[lateral], 1] = (torch.where(torch.rand(n,device=commands.device)<.5,-1.,1.)*uniform(.02,.04))[lateral]
    commands[ids[lateral], 2] = 0.
    turn = (family >= .5) & (family < .65)
    commands[ids[turn], :2] = 0.
    commands[ids[turn], 2] = (torch.where(torch.rand(n,device=commands.device)<.5,-1.,1.)*uniform(.15,.35))[turn]
    reverse = (family >= .65) & (family < .8)
    commands[ids[reverse], 0] = uniform(ranges['vx'][0],-.02)[reverse]
    commands[ids[reverse], 1:3] = 0.
    commands[ids, 5] = uniform(*ranges['frequency']) if stage == 'S5' else 1.


def stage_gate(stage, report, baseline):
    from acceptance import assess
    return assess(stage, report if isinstance(report,list) else [report], baseline)


def next_frequency_range(stage, current, completed_episodes, survival_fraction, elapsed_seconds, minimum_episodes=256):
    """Expand only the frequency range after actual completed training episodes.

    This is a training curriculum condition, never independent acceptance.
    """
    if stage != 'S5' or list(current) != [.9,1.1]:
        return list(current)
    if completed_episodes >= minimum_episodes and survival_fraction is not None and survival_fraction >= .9 and elapsed_seconds >= 240:
        return [.8,1.2]
    return list(current)
