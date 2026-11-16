import numpy as np
import pytest

from deep_hedging.models.heston import simulate_heston

# Typical parameters. The Feller condition 2*kappa*theta > xi^2 holds:
# 2 * 1.5 * 0.04 = 0.12 > 0.09 = 0.3^2
PARAMS = dict(s0=100, v0=0.04, mu=0.03, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)


# --- Basic properties -----------------------------------------------------

def test_shapes():
    prices, variances = simulate_heston(**PARAMS, T=30 / 252, n_steps=30, n_paths=1000, seed=0)
    assert prices.shape == (1000, 31)
    assert variances.shape == (1000, 31)


def test_start_values():
    prices, variances = simulate_heston(**PARAMS, T=1.0, n_steps=10, n_paths=500, seed=0)
    assert np.allclose(prices[:, 0], 100)
    assert np.all(variances[:, 0] == 0.04)


def test_prices_positive_and_finite_even_if_feller_is_violated():
    # xi = 1.0 strongly violates the Feller condition, so the raw variance
    # often goes negative. Thanks to the truncation v+ = max(v, 0), no square
    # root of a negative number is taken, so there are no NaNs.
    prices, variances = simulate_heston(
        s0=100, v0=0.04, mu=0.0, kappa=1.5, theta=0.04, xi=1.0, rho=-0.7,
        T=1.0, n_steps=252, n_paths=2000, seed=0,
    )
    assert np.all(np.isfinite(prices))
    assert np.all(np.isfinite(variances))
    assert np.all(prices > 0)


# --- Reproducibility --------------------------------------------------------

def test_same_seed_gives_same_paths():
    a = simulate_heston(**PARAMS, T=1.0, n_steps=10, n_paths=50, seed=42)
    b = simulate_heston(**PARAMS, T=1.0, n_steps=10, n_paths=50, seed=42)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_different_seeds_give_different_paths():
    a, _ = simulate_heston(**PARAMS, T=1.0, n_steps=10, n_paths=50, seed=1)
    b, _ = simulate_heston(**PARAMS, T=1.0, n_steps=10, n_paths=50, seed=2)
    assert not np.array_equal(a, b)


# --- Special case: no vol of vol ------------------------------------------

def test_constant_variance_without_vol_of_vol():
    # With xi = 0 and v0 = theta, the variance never moves:
    # v_{t+dt} = v_t + kappa * (theta - v_t) * dt = theta.
    # Heston then reduces to GBM with sigma = sqrt(theta).
    prices, variances = simulate_heston(
        s0=100, v0=0.04, mu=0.0, kappa=1.5, theta=0.04, xi=0.0, rho=-0.7,
        T=1.0, n_steps=252, n_paths=2000, seed=0,
    )
    assert np.allclose(variances, 0.04)
    # Daily log-returns then have standard deviation sigma * sqrt(dt)
    log_returns = np.diff(np.log(prices), axis=1)
    assert log_returns.std() == pytest.approx(0.2 * np.sqrt(1 / 252), rel=0.01)


# --- Statistical properties ------------------------

def test_terminal_price_mean():
    # E[S_T] = s0 * exp(mu * T). This holds exactly even for the discretised
    # scheme: each log-Euler step multiplies the expected price by exp(mu * dt).
    n = 100_000
    prices, _ = simulate_heston(**PARAMS, T=1.0, n_steps=100, n_paths=n, seed=1)
    s_T = prices[:, -1]
    standard_error = s_T.std(ddof=1) / np.sqrt(n)
    assert abs(s_T.mean() - 100 * np.exp(0.03)) < 4 * standard_error


def test_terminal_variance_mean():
    # Mean reversion: E[v_T] = theta + (v0 - theta) * exp(-kappa * T).
    # Start away from theta so the test checks something. Feller holds
    # (2 * 2 * 0.04 = 0.16 > 0.09) and enough steps keep the Euler bias small.
    v0, kappa, theta, T, n = 0.09, 2.0, 0.04, 1.0, 100_000
    _, variances = simulate_heston(
        s0=100, v0=v0, mu=0.0, kappa=kappa, theta=theta, xi=0.3, rho=-0.7,
        T=T, n_steps=250, n_paths=n, seed=2,
    )
    v_T = variances[:, -1]
    expected = theta + (v0 - theta) * np.exp(-kappa * T)
    standard_error = v_T.std(ddof=1) / np.sqrt(n)
    assert abs(v_T.mean() - expected) < 4 * standard_error


def test_correlation_of_price_and_variance_shocks():
    # Changes in log-price and changes in variance should have correlation rho
    rho = -0.7
    prices, variances = simulate_heston(**{**PARAMS, "rho": rho}, T=1.0, n_steps=250,
                                        n_paths=20_000, seed=3)
    d_log_s = np.diff(np.log(prices), axis=1).ravel()
    d_v = np.diff(variances, axis=1).ravel()
    assert np.corrcoef(d_log_s, d_v)[0, 1] == pytest.approx(rho, abs=0.02)


def test_zero_correlation():
    prices, variances = simulate_heston(**{**PARAMS, "rho": 0.0}, T=1.0, n_steps=250,
                                        n_paths=20_000, seed=4)
    d_log_s = np.diff(np.log(prices), axis=1).ravel()
    d_v = np.diff(variances, axis=1).ravel()
    assert np.corrcoef(d_log_s, d_v)[0, 1] == pytest.approx(0.0, abs=0.02)