"""
Classical hedging strategies used as benchmarks for deep hedging.
"""

import numpy as np
from deep_hedging.black_scholes import call_delta

def bs_delta_strategy(paths, k, T, r, sigma):
    """ 
    Black-Scholes delta hedge: at each t_k hold N(d1) shares, computed with S_k and the remaining time T - t_k.

    Input:
        paths       prices at t_0, ..., t_N for each path, shape (n_paths, N + 1)
        k           strike 
        T           time to maturity in years at t_0
        r           risk-free rate
        sigma       volatility 
    Output:
        deltas, shape (n_paths, N)
    """
    N = paths.shape[1] - 1
    tau = T - np.arange(N) * T / N
    deltas = call_delta(paths[:, :-1], k, tau, r, sigma)
    return deltas

   