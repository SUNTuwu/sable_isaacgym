"""6x51 history -> ROOT velocity and learned predictive latent state.

The actor sees detached estimates. The latent receives an auxiliary next-state
prediction loss, with reset boundaries masked; this is not a copy of HIMLoco.
"""
import torch
from torch import nn
from torch.nn import functional as F


class HistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(306, 128), nn.ELU(), nn.Linear(128, 64),
                                     nn.ELU(), nn.Linear(64, 19))
        self.next_proprio = nn.Sequential(nn.Linear(19, 64), nn.ELU(), nn.Linear(64, 51))

    def forward(self, history):
        out = self.encoder(history.reshape(-1, 306))
        return out[:, :3], F.normalize(out[:, 3:], p=2, dim=-1, eps=1e-6)

    def auxiliary_loss(self, history, velocity, next_proprio, done):
        estimate, latent = self(history)
        velocity_loss = (estimate - velocity).square().mean()
        prediction = self.next_proprio(torch.cat((estimate, latent), -1))
        valid = (~done.bool()).float()
        # All fields are in their declared observation scales. Targets are clean.
        prediction_loss = ((prediction - next_proprio).square().mean(-1) * valid).sum() / valid.sum().clamp(min=1)
        return velocity_loss + .1 * prediction_loss, {'velocity_mse': velocity_loss.detach(),
                                                      'next_proprio_mse': prediction_loss.detach(),
                                                      'valid_pairs': valid.sum().detach()}
