"""Turn-only reference clock without accumulated float32 phase drift.

The caller stores the reset phase before any physics step. episode_steps is the
already executed count, so the returned phase is the next PD target's phase.
Six translation rows use the exact original operation and dtype, bit for bit.
No policy, critic, optimizer, reference period or reward weight changes here.
"""
import torch


def advance_reference_phase(previous, reset_phase, episode_steps, periods, dt, turning):
    legacy = torch.remainder(previous + dt / periods, 1.)
    precise = torch.remainder(reset_phase.double() + (episode_steps.double() + 1.) * float(dt) / periods.double(), 1.)
    # A float64 value immediately below 1 can round to float32 1. Keep [0,1).
    precise = torch.remainder(precise.to(previous.dtype), 1.)
    return torch.where(turning, precise, legacy)
