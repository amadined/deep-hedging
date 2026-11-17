"""
Geometric Brownian motion 

Geometric Brownian motion is the stock model behind Black-scholes. Written as 
stochastic differential equation:

dS_t = mu * S_t * dt + sigma * S_t * dW_t

We simulate it in log space, so we only have to add increments and
exponentiate at the end. With X_t = ln(S_t), the exact solution gives

dX_t = X_0 + (mu - sigma^2 / 2) * dt + sigma * np.sqrt(dt) * Z

where Z ~ N(0, 1) is drawn independently for every step and path.
"""

import numpy as np

def simulate_gbm(s0, mu, sigma, T, n_steps, n_paths, seed=None):
    """
    Input:
        s0          starting price
        mu          drift per year
        sigma       volatility per year
        T           total time in years
        n_steps     number of time steps 
        n_paths     number of paths
        seed        random seed for reproducibility
    Output:
        np.ndarray of shape (n_paths, n_steps + 1). Each row is one path,
        column 0 equals s0 and column j is the price at time j * T / n_steps.
    """

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal(size=(n_paths, n_steps))
    log_inc = (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_inc, axis=1)
    res = np.hstack((np.zeros((n_paths,1)), log_paths))
    res = s0 * np.exp(res)
    return res
    


