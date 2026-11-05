"""Closed-form Black-Scholes formulas for European options on a non-dividend-paying stock.

These are the "ground truth" of the project: the Monte Carlo engine and,
later, the hedging network are checked against them.

Notation (used throughout the project):
    s      current stock price
    k      strike
    t      time to maturity in years (must be > 0)
    r      continuously compounded risk-free rate
    sigma  volatility (annualised)
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(s, k, t, r, sigma):
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def call_price(s, k, t, r, sigma):
    """Black-Scholes price of a European call option."""
    d1, d2 = _d1_d2(s, k, t, r, sigma)
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def put_price(s, k, t, r, sigma):
    """Black-Scholes price of a European put option."""
    d1, d2 = _d1_d2(s, k, t, r, sigma)
    return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


def call_delta(s, k, t, r, sigma):
    """Delta of a European call: the number of shares held in the BS hedge."""
    d1, _ = _d1_d2(s, k, t, r, sigma)
    return norm.cdf(d1)
