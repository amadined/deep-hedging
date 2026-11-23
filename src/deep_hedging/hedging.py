"""
Profit and loss of a hedged short option position.

The hedger sells one option for the premium p0, holds delta_k shares from
t_k to t_{k+1} (chosen using information up to t_k only), pays the payoff
at maturity and unwinds the position. Proportional transaction costs are
charged on every trade, including the initial purchase and the final sale.
Interest rates are zero.
"""

import numpy as np

def hedge_pnl(paths, deltas, payoff, p0, cost):
    """
    Input:
        paths       prices at t_0, ..., t_N for each path, shape (n_paths, N + 1)
        deltas      positions δ_0, ..., δ_{N-1} for each path, shape (n_paths, N)
        payoff      option payoffs, shape (n_paths,)
        p0          the premium
        cost        cost rate 
    Ouput:
        the P&L of each path, an array of shape (n_paths,)
    """
    n_paths = paths.shape[0]
    tr_gains = np.sum(deltas * np.diff(paths, axis=1), axis=1)
    positions = np.hstack((np.zeros((n_paths, 1)), deltas, np.zeros((n_paths, 1))))
    tr_costs = cost * np.sum(np.abs(np.diff(positions, axis=1)) * paths, axis=1)
    pnl = p0 - payoff + tr_gains - tr_costs
    return pnl

def turnover(deltas):
    """
    Total number of shares traded on each path, including the initial
    purchase and the final sale at maturity.

    Input:
        deltas      positions, shape (n_paths, N)
    Output:
        array of shape (n_paths,)
    """
    n_paths = deltas.shape[0]
    positions = np.hstack((np.zeros((n_paths, 1)), deltas, np.zeros((n_paths, 1))))
    return np.sum(np.abs(np.diff(positions, axis=1)), axis=1)