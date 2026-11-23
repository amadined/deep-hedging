"""
Deep hedging (Buehler, Gonon, Teichmann and Wood, 2019) with PyTorch.

A neural network chooses the hedge position at each time step. It is trained
by minimising the CVaR of the hedging losses on simulated paths, with the P&L
computed in PyTorch so gradients flow through the whole trading period.
"""

import numpy as np
import torch
from torch import nn


def hedge_pnl_torch(paths, deltas, payoff, p0, cost):
    """
    Same formula as hedging.hedge_pnl, written in PyTorch so gradients can flow through it.
    Input:
        paths       prices at t_0, ..., t_N for each path, shape (n_paths, N + 1)
        deltas      positions δ_0, ..., δ_{N-1} for each path, shape (n_paths, N)
        payoff      option payoffs, shape (n_paths,)
        p0          the premium
        cost        cost rate 
    Output:
        the P&L of each path, a tensor of shape (n_paths,)
    """
    n_paths = paths.shape[0]
    tr_gains = torch.sum(deltas * torch.diff(paths, dim=1), dim=1)
    positions = torch.cat((torch.zeros((n_paths, 1), dtype=deltas.dtype), deltas, torch.zeros((n_paths, 1), dtype=deltas.dtype)), dim=1)
    tr_costs = cost * torch.sum(torch.abs(torch.diff(positions, dim=1)) * paths, dim=1)
    pnl = p0 - payoff + tr_gains - tr_costs
    return pnl


def cvar_torch(pnl, alpha=0.95):
    """
    Empirical CVaR of the losses -pnl: the mean of the worst (1 - alpha) fraction. 
    Differentiable, used as the training loss.
    """
    n = pnl.shape[0]
    losses = -pnl
    k = max(1, int((1 - alpha) * n))
    return torch.topk(losses, k).values.mean()


class DeepHedger(nn.Module):
    """
    Neural network hedging strategy (Buehler et al., 2019).

    At each time t_j, a small MLP maps (log(S_j / K), time to maturity,
    previous position) to the new position delta_j. The same MLP is used
    at every time step. The previous position is an input so that the
    network can learn not to trade when trading is not worth the cost.

    Input of forward:
        paths       prices at t_0, ..., t_N, tensor of shape (n_paths, N + 1)
    Output of forward:
        deltas      positions, tensor of shape (n_paths, N)
    """

    def __init__(self, k, T, hidden=32):
        super().__init__()          
        self.k = k
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, paths):
        n_paths = paths.shape[0]             
        n_steps = paths.shape[1] - 1                  
        dt = self.T / n_steps

        prev = torch.zeros(n_paths, dtype=paths.dtype)   # no shares before t_0
        deltas = []

        for j in range(n_steps):
            log_moneyness = torch.log(paths[:, j] / self.k)                     # log(S_j / K) for all paths, shape (n_paths,)
            tau = torch.full((n_paths,), self.T - j, dtype=paths.dtype)         # time left at t_j

            x = torch.stack([log_moneyness, tau, prev], dim=1)                  # (n_paths, 3)
            delta = self.net(x).squeeze(1)                                      # (n_paths,)

            deltas.append(delta)
            prev = delta

        return torch.stack(deltas, dim=1)                                       # (n_paths, n_steps)



def train_hedger():
    ...   