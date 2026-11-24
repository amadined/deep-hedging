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
from deep_hedging.heston_pricing import heston_call_price
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.models.heston import simulate_heston


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

    With use_variance=True, the current volatility sqrt(v_j) is a fourth
    input. This is meant for stochastic volatility models such as Heston,
    where the variance changes over time and can be observed by the hedger.

    Input of __init__:
        k               strike
        T               time to maturity in years at t_0
        hidden          neurons per hidden layer
        use_variance    if True, the volatility sqrt(v_j) is a fourth input
    Input of forward:
        paths           prices at t_0, ..., t_N, tensor of shape (n_paths, N + 1)
        variances       variances at t_0, ..., t_N, same shape as paths;
                        required if use_variance is True, ignored otherwise
    Output of forward:
        deltas          positions, tensor of shape (n_paths, N)
    """

    def __init__(self, k, T, hidden=32, use_variance=False):
        super().__init__()
        self.k = k
        self.T = T
        self.use_variance = use_variance
        n_inputs = 4 if use_variance else 3
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, paths, variances=None):
        if self.use_variance and variances is None:
            raise ValueError("this model needs the variance paths (use_variance=True)")

        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1
        dt = self.T / n_steps

        prev = torch.zeros(n_paths, dtype=paths.dtype)   # no shares before t_0
        deltas = []

        for j in range(n_steps):
            log_moneyness = torch.log(paths[:, j] / self.k)                     # log(S_j / K) for all paths, shape (n_paths,)
            tau = torch.full((n_paths,), self.T - j * dt, dtype=paths.dtype)    # time left at t_j
            features = [log_moneyness, tau, prev]

            if self.use_variance:
                # Volatility instead of variance: closer in scale to the other inputs.
                # Raw Heston variances can be slightly negative, hence the clamp.
                features.append(torch.sqrt(torch.clamp(variances[:, j], min=0)))

            x = torch.stack(features, dim=1)                                    # (n_paths, 3 or 4)
            delta = self.net(x).squeeze(1)                                      # (n_paths,)

            deltas.append(delta)
            prev = delta

        return torch.stack(deltas, dim=1)                                       # (n_paths, n_steps)


def _train_loop(model, sample_batch, p0, k, cost, n_iters, lr, alpha, print_every):
    """
    Training loop shared by train_hedger and train_hedger_heston.

    Input:
        model           the DeepHedger to train (changed in place)
        sample_batch    function i -> (paths, variances): the training batch of
                        iteration i as float32 tensors, variances may be None
        p0              premium received for the option
        k               strike
        cost            proportional transaction cost rate c
        n_iters         number of training iterations (optimiser steps)
        lr              learning rate of Adam
        alpha           CVaR level of the loss
        print_every     print the loss every this many iterations (0 = never)
    Output:
        losses          list with the training loss of every iteration
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for i in range(n_iters):
        paths, variances = sample_batch(i)
        payoff = torch.clamp(paths[:, -1] - k, min=0)

        deltas = model(paths, variances)
        pnl = hedge_pnl_torch(paths, deltas, payoff, p0, cost)
        loss = cvar_torch(pnl, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if print_every and i % print_every == 0:
            print(f"iteration {i:5d}   CVaR loss {loss.item():.4f}")

    return losses


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
    p0 = call_price(s0, k, T, r, sigma)

    def sample_batch(i):
        # Fresh paths every iteration. The seeds start at 1_000_000 * (seed + 1),
        # so training never reuses the small seeds used for evaluation.
        path_seed = 1_000_000 * (seed + 1) + i
        paths = simulate_gbm(s0, r, sigma, T, n_steps, batch_size, seed=path_seed)
        return torch.tensor(paths, dtype=torch.float32), None

    losses = _train_loop(model, sample_batch, p0, k, cost, n_iters, lr, alpha, print_every)
    return model, losses


def train_hedger_heston(
    k,
    T,
    heston,
    cost,
    n_steps,
    use_variance=True,
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
    Train a DeepHedger on simulated Heston paths by minimising the CVaR of
    the hedging losses of a short European call.

    Same procedure as train_hedger, but the paths come from the Heston
    simulator and the premium is the semi-analytic Heston price.

    Input:
        k, T            strike and maturity in years
        heston          dict with the Heston parameters v0, kappa, theta, xi, rho
        cost            proportional transaction cost rate c
        n_steps         number of rebalancing steps N
        use_variance    if True, the network also sees the current volatility
        other inputs    as in train_hedger
    Output:
        model           the trained DeepHedger
        losses          list with the training loss of every iteration
    """
    torch.manual_seed(seed)
    model = DeepHedger(k, T, hidden, use_variance=use_variance)
    p0 = heston_call_price(s0, k, T, r, **heston)

    def sample_batch(i):
        path_seed = 1_000_000 * (seed + 1) + i
        paths, variances = simulate_heston(
            s0=s0, mu=r, T=T, n_steps=n_steps, n_paths=batch_size, seed=path_seed, **heston
        )
        return (torch.tensor(paths, dtype=torch.float32),
                torch.tensor(variances, dtype=torch.float32))

    losses = _train_loop(model, sample_batch, p0, k, cost, n_iters, lr, alpha, print_every)
    return model, losses


def deep_hedge_strategy(model, paths, variances=None):
    """
    Positions of a trained DeepHedger for NumPy paths, as a NumPy array.

    Lets the network be evaluated with the same NumPy functions as the
    classical strategies (hedge_pnl, summarize, turnover).

    Input:
        model       a (trained) DeepHedger
        paths       NumPy array of prices, shape (n_paths, N + 1)
        variances   NumPy array of variances, same shape as paths; only
                    needed for a model with use_variance=True
    Output:
        deltas      NumPy array (float64), shape (n_paths, N)
    """
    model.eval()
    with torch.no_grad():
        paths_t = torch.tensor(paths, dtype=torch.float32)
        variances_t = None if variances is None else torch.tensor(variances, dtype=torch.float32)
        deltas = model(paths_t, variances_t)
    return deltas.numpy().astype(np.float64)