"""KOALA's correction adapter: a per-token MLP from what the editor sees with the plain instruction to the teacher's
correction dv = v(x, sigma, teacher) - v(x, sigma, plain) at the same state (collect_adapter.py, train_adapter.py).

Token inputs: the reference token z_ref, x0_hat - z_ref (x0_hat = x - sigma v from the plain v), the plain v, the means
of z_ref and of x0_hat - z_ref over the subject mask M, M itself, sigma and Fourier features of the token position.
Inputs are standardised with the statistics of training tokens (mu, sd); the last layer starts at zero, so an untrained
adapter adds nothing. Load a head with Adapter().load_state_dict(torch.load(path)["state"]).
"""
import math

import torch
from torch import nn

FREQS = 4


def positions(side=32):
    """(side * side, 16) sin / cos features of the token's (row, col) scaled to [0, 1], tokens in raster order."""
    r = torch.arange(side * side)
    a = (torch.stack([r // side, r % side], -1) / (side - 1))[..., None] * (2.0 ** torch.arange(FREQS) * math.pi)
    return torch.cat([a.sin(), a.cos()], -1).flatten(1)


class Adapter(nn.Module):
    def __init__(self, channels=128, width=256):
        super().__init__()
        n = 5 * channels + 2 + 4 * FREQS
        self.register_buffer("mu", torch.zeros(n))
        self.register_buffer("sd", torch.ones(n))
        self.net = nn.Sequential(nn.Linear(n, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def feats(self, ref, x0, v, m, sigma, pos):
        """ref, x0, v (B, 1024, C) fp32; m (B, 1024, 1); sigma (B,) -> unstandardised inputs (B, 1024, 5C + 18)."""
        d = x0 - ref
        mean = lambda t: ((t * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1e-6)).expand_as(t)
        return torch.cat([ref, d, v, mean(ref), mean(d), m, sigma.view(-1, 1, 1).expand_as(m), pos.expand(len(ref), -1, -1)], -1)

    def forward(self, ref, x0, v, m, sigma, pos):
        return self.net((self.feats(ref, x0, v, m, sigma, pos) - self.mu) / self.sd)
