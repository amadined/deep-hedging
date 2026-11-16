"""
Heston (1993) stochastic volatility model: the variance 
follows a mean-reverting square-root process, correlated with the stock price.
Full-truncation Euler scheme for the variance with a log-Euler step
for the stock price (Lord, Koekkoek and van Dijk, 2010).
"""
import numpy as np

def simulate_heston(s0, v0, mu, kappa, theta, xi, rho, T, n_steps, n_paths, seed=None):
    """
    Input: 
        s0          starting price
        v0          starting variance
        mu          drift per year
        kappa       speed of mean reversion of the variance
        theta       long-run variance the process is pulled towards
        xi          volatility of the variance
        rho         correlation between price shocks and variance shocks
        T           total time in years
        n_steps     number of time steps 
        n_paths     number of paths
        seed        random seed for reproducibility
    Output:
        two arrays prices, variances, both with shape (n_paths, n_steps + 1),
        column 0 is s0 and v0, raw variances can be slightly negative, use np.maximum(variances, 0).
    """

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    log_prices = np.empty((n_paths, n_steps + 1))
    log_prices[:, 0] = np.full(n_paths, np.log(s0))
    variances = np.empty((n_paths, n_steps + 1))
    variances[:, 0] = np.full(n_paths, v0)

    for j in range(n_steps):
        eps_1 = rng.standard_normal(size=n_paths)
        eps_2 = rng.standard_normal(size=n_paths)
        z_1 = eps_1
        z_2 = rho * eps_1 + np.sqrt(1 - rho**2) * eps_2
        v_plus = np.maximum(0, variances[:, j])
        log_prices[:, j + 1] = log_prices[:, j] + (mu - v_plus / 2) * dt + np.sqrt(v_plus * dt) * z_1
        variances[:, j + 1] = variances[:, j] + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus * dt) * z_2
    
    return np.exp(log_prices), variances
