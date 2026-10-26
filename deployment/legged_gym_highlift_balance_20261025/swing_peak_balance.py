"""Phase-completed left/right peak comparison, an experimental reward helper.

Compare each finished swing with the opposite foot's latest finished swing.
Never compare instantaneous alternating heights. Height tracking remains a
separate objective: two non-lifting feet have zero imbalance, not a bonus.
Inspired by open-source phase rewards; this particular adaptation is ours.
"""
import torch


class SwingPeakBalance:
    def __init__(self, count, device, period_s, dt):
        self.period_s, self.dt = period_s, dt
        self.peak = torch.zeros(count, 4, device=device)
        self.last_peak = torch.zeros_like(self.peak)
        self.previous_swing = torch.zeros_like(self.peak, dtype=torch.bool)
        self.started = torch.zeros_like(self.previous_swing)
        self.valid = torch.zeros_like(self.previous_swing)
        self.opposite = torch.tensor([1, 0, 3, 2], device=device)
        self.completed = torch.zeros_like(self.previous_swing)
        self.paired = torch.zeros_like(self.previous_swing)
        self.difference = torch.zeros_like(self.peak)

    def reset(self, ids):
        for value in (self.peak, self.last_peak, self.previous_swing,
                      self.started, self.valid, self.completed, self.paired, self.difference):
            value[ids] = 0

    def update(self, clearance, swing, eligible, target_height):
        # Clear stale peaks across resets, stopping commands, and startup ramp.
        ids = torch.nonzero(~eligible, as_tuple=False).flatten()
        for value in (self.peak, self.last_peak, self.started, self.valid, self.completed):
            value[ids] = 0
        # Track phase even while ineligible, excluding a partial startup swing.
        start = swing & ~self.previous_swing & eligible[:, None]
        self.peak[start] = 0
        self.started |= start
        self.peak = torch.where(swing & eligible[:, None], torch.maximum(self.peak, clearance.clamp(min=0)), self.peak)
        self.completed = self.previous_swing & ~swing & self.started & eligible[:, None]
        paired = self.completed & self.valid[:, self.opposite]
        difference = self.peak - self.last_peak[:, self.opposite]
        self.paired = paired
        self.difference = torch.where(paired, difference, torch.zeros_like(difference))
        cost = ((difference / target_height.clamp(min=.001)[:, None]) ** 2).clamp(max=9)
        # Four events per period; after the environment's dt reward scaling,
        # the coefficient has the same per-second meaning at different dt.
        result = (cost * paired).sum(1) * self.period_s / (4 * self.dt)
        self.last_peak = torch.where(self.completed, self.peak, self.last_peak)
        self.valid |= self.completed
        self.started &= ~self.completed
        self.previous_swing.copy_(swing)
        return result
