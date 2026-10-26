"""Stage changes are explicit and gated by measured independent evaluations."""
import torch

STAGES = ('S1', 'S2', 'S3', 'S5')


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
    commands[ids[lateral], 2] = 0.
    turn = (family >= .5) & (family < .65)
    commands[ids[turn], :2] = 0.
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
