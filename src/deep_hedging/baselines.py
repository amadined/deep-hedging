"""
Classical hedging strategies used as benchmarks for deep hedging.
"""

import numpy as np
from deep_hedging.black_scholes import call_delta, call_gamma

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

def leland_sigma(sigma, cost, dt):
    """
    Leland (1985) adjusted volatility for a short option position.

    Input:
        sigma       volatility
        cost        proportional cost rate c per trade (as in hedge_pnl)
        dt          time between rebalancing dates
    Output:
        adjusted volatility sigma_hat (> sigma if cost > 0)
    """
    return sigma * np.sqrt(1 + np.sqrt(2 / np.pi) * 2 * cost / (sigma * np.sqrt(dt)))

def leland_strategy(paths, k, T, r, sigma, cost):
    """Leland (1985) delta hedge with transaction costs.

    Hedges with the Black-Scholes delta, but computed with the adjusted
    volatility sigma_hat from leland_sigma instead of sigma. The higher
    volatility accounts for the costs of rebalancing every dt = T / N:
    the delta curve becomes flatter, so the hedger trades somewhat less.
    The matching option premium is the Black-Scholes price with sigma_hat.

    Input:
        paths       prices at t_0, ..., t_N, shape (n_paths, N + 1)
        k           strike
        T           time to maturity in years at t_0
        r           risk-free rate
        sigma       volatility of the stock
        cost        proportional transaction cost rate c per trade
    Output:
        deltas, shape (n_paths, N): the number of shares held from t_j to t_{j+1}
    """
    n = paths.shape[1] - 1
    dt = T / n
    sigma_hat = leland_sigma(sigma, cost, dt)
    return bs_delta_strategy(paths, k, T, r, sigma_hat)


def whalley_wilmott_strategy(paths, k, T, r, sigma, cost, risk_aversion):
    """Whalley-Wilmott (1997) no-trade band around the Black-Scholes delta.

    At each t_j the hedger keeps the current position if it lies inside the
    band [delta - H, delta + H], otherwise it trades to the nearest edge of
    the band. The half-width of the band is

        H = (3/2 * exp(-r * tau) * cost * S * gamma^2 / risk_aversion)^(1/3),

    where delta and gamma are the Black-Scholes delta and gamma at S = S_j
    with remaining time tau = T - t_j.

    Input:
        paths           prices at t_0, ..., t_N, shape (n_paths, N + 1)
        k               strike
        T               time to maturity in years at t_0
        r               risk-free rate
        sigma           volatility used for delta and gamma
        cost            proportional transaction cost rate c
        risk_aversion   risk aversion of the hedger (> 0), larger values
                        give a narrower band and more trading
    Output:
        deltas, shape (n_paths, N): the number of shares held from t_j to t_{j+1}
    """
    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1
    dt = T / n_steps

    deltas = np.empty((n_paths, n_steps))
    position = np.zeros(n_paths)  # no shares before t_0

    for j in range(n_steps):
        s = paths[:, j]  # only information up to t_j: no look-ahead
        tau = T - j * dt
        delta = call_delta(s, k, tau, r, sigma)
        gamma = call_gamma(s, k, tau, r, sigma)
        half_width = (1.5 * np.exp(-r * tau) * cost * s * gamma**2 / risk_aversion) ** (1 / 3)

        # Inside the band: keep the position. Outside: move to the nearest edge.
        position = np.clip(position, delta - half_width, delta + half_width)
        deltas[:, j] = position

    return deltas