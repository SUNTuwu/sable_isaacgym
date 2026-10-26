"""Behavioral checks of complete-swing comparison, reset and dt scaling."""
import json, math
from pathlib import Path
import torch
from swing_peak_balance import SwingPeakBalance
from run_io import atomic_json, sha256


def sample(dt, heights, swap=False):
    tracker = SwingPeakBalance(1, 'cpu', .8, dt)
    offset = torch.tensor([[.75, .25, 0., .5]])
    if swap:offset = offset[:, [1, 0, 3, 2]]
    total = 0.;done = 0
    for i in range(round(6.4 / dt)):
        t = i * dt
        phase = torch.remainder(t / .8 + offset, 1.)
        progress = ((phase - .75) / .25).clamp(0, 1)
        height = torch.tensor([heights]) * torch.sin(math.pi * progress).square()
        value = tracker.update(height, phase >= .75, torch.tensor([t >= .8]), torch.tensor([.012]))
        assert torch.isfinite(value).all()
        if t >= 3.2:total += float(value[0]) * dt;done += int(tracker.completed.sum())
    return total / 3.2, done, tracker


equal, count, _ = sample(.02, [.012] * 4)
mismatch, _, tracker = sample(.02, [.012, .006, .012, .012])
finer, _, _ = sample(.01, [.012, .006, .012, .012])
mirror, _, _ = sample(.02, [.006, .012, .012, .012], True)
checks = {'alternating_equal_peaks_zero_cost': abs(equal) < 1e-8,
          'mismatch_expected_per_second': abs(mismatch - .125) < .01,
          'dt_scaling': abs(mismatch - finer) < .01,
          'left_right_exchange': abs(mismatch - mirror) < 1e-6,
          'completed_swings_recorded': count == 16}
tracker.reset(torch.tensor([0]))
checks['reset_clears_history'] = not tracker.valid.any() and not tracker.last_peak.any()
tracker.update(torch.full((1, 4), .02), torch.ones(1, 4, dtype=torch.bool), torch.tensor([False]), torch.tensor([.012]))
tracker.update(torch.full((1, 4), .02), torch.ones(1, 4, dtype=torch.bool), torch.tensor([True]), torch.tensor([.012]))
tracker.update(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool), torch.tensor([True]), torch.tensor([.012]))
checks['partial_startup_swing_excluded'] = not tracker.completed.any() and not tracker.valid.any()
zero, _, _ = sample(.02, [0.] * 4)
checks['no_bonus_for_both_not_lifting'] = zero == 0
here = Path(__file__).resolve().parent
result = {'passed': all(checks.values()), 'checks': {k:bool(v) for k,v in checks.items()},
          'measured_per_second_cost': mismatch,
          'source_sha256': {n:sha256(here/n) for n in ['swing_peak_balance.py', 'check_swing_peak_balance.py']}}
atomic_json(here/'verification/swing_peak_balance.json', result)
print(json.dumps(result))
assert result['passed']
