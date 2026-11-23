"""
Deep hedging (Buehler, Gonon, Teichmann and Wood, 2019) with PyTorch.

A neural network chooses the hedge position at each time step. It is trained
by minimising the CVaR of the hedging losses on simulated paths, with the P&L
computed in PyTorch so gradients flow through the whole trading period.
"""

import numpy as np
import torch
from torch import nn

from deep_hedging.black_scholes import call_price
from deep_hedging.models.gbm import simulate_gbm


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
            tau = torch.full((n_paths,), self.T - j * dt, dtype=paths.dtype)    # time left at t_j

            x = torch.stack([log_moneyness, tau, prev], dim=1)                  # (n_paths, 3)
            delta = self.net(x).squeeze(1)                                      # (n_paths,)

            deltas.append(delta)
            prev = delta

        return torch.stack(deltas, dim=1)                                       # (n_paths, n_steps)

def train_hedger(
    k,
    T,
    sigma,
    cost,
    n_steps,
    s0=100.0,
    r=0.0,
    n_iters=1500,
    batch_size=10_000,
    lr=1e-3,
    alpha=0.95,
    hidden=32,
    seed=0,
    print_every=100,
):
    """
    Train a DeepHedger on simulated Black-Scholes (GBM) paths by minimising
    the CVaR of the hedging losses of a short European call.

    Every iteration uses a fresh batch of paths, so the network never sees
    the same path twice. The premium is the Black-Scholes price, the same as
    for the classical strategies, so all strategies are compared fairly.

    Input:
        k, T, sigma     strike, maturity in years, volatility
        cost            proportional transaction cost rate c
        n_steps         number of rebalancing steps N
        s0, r           starting price and interest rate (paths use mu = r)
        n_iters         number of training iterations (optimiser steps)
        batch_size      number of paths per iteration
        lr              learning rate of Adam
        alpha           CVaR level of the loss
        hidden          neurons per hidden layer
        seed            seed for the initial weights and the training paths
        print_every     print the loss every this many iterations (0 = never)
    Output:
        model           the trained DeepHedger
        losses          list with the training loss of every iteration
    """
    torch.manual_seed(seed)
    model = DeepHedger(k, T, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    p0 = call_price(s0, k, T, r, sigma)

    losses = []
    for i in range(n_iters):
        # Fresh paths every iteration. The seeds start at 1_000_000 * (seed + 1),
        # so training never reuses the small seeds used for evaluation.
        path_seed = 1_000_000 * (seed + 1) + i
        paths = torch.tensor(
            simulate_gbm(s0, r, sigma, T, n_steps, batch_size, seed=path_seed),
            dtype=torch.float32,
        )
        payoff = torch.clamp(paths[:, -1] - k, min=0)

        deltas = model(paths)
        pnl = hedge_pnl_torch(paths, deltas, payoff, p0, cost)
        loss = cvar_torch(pnl, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if print_every and i % print_every == 0:
            print(f"iteration {i:5d}   CVaR loss {loss.item():.4f}")

    return model, losses


def deep_hedge_strategy(model, paths):
    """
    Positions of a trained DeepHedger for NumPy paths, as a NumPy array.

    Lets the network be evaluated with the same NumPy functions as the
    classical strategies (hedge_pnl, summarize, turnover).

    Input:
        model       a (trained) DeepHedger
        paths       NumPy array of prices, shape (n_paths, N + 1)
    Output:
        deltas      NumPy array (float64), shape (n_paths, N)
    """
    model.eval()
    with torch.no_grad():
        deltas = model(torch.tensor(paths, dtype=torch.float32))
    return deltas.numpy().astype(np.float64)