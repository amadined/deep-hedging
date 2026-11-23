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
    ...   


def train_hedger():
    ...   