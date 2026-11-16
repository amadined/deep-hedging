import numpy as np
import pytest

from deep_hedging.models.gbm import simulate_gbm


# --- Basic properties -----------------------------------------------------

def test_shape():
    paths = simulate_gbm(s0=100, mu=0.0, sigma=0.2, T=30 / 252, n_steps=30, n_paths=1000, seed=0)
    assert paths.shape == (1000, 31)


def test_start_value():
    paths = simulate_gbm(s0=100, mu=0.0, sigma=0.2, T=1.0, n_steps=10, n_paths=500, seed=0)
    assert np.all(paths[:, 0] == 100)


def test_prices_are_positive():
    # GBM prices can never become negative
    paths = simulate_gbm(s0=100, mu=-0.5, sigma=1.5, T=5.0, n_steps=500, n_paths=1000, seed=0)
    assert np.all(paths > 0)


# --- Reproducibility --------------------------------------------------------

def test_same_seed_gives_same_paths():
    a = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 50, seed=42)
    b = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 50, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_paths():
    a = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 50, seed=1)
    b = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 50, seed=2)
    assert not np.array_equal(a, b)


# --- Statistical properties ------------------------

def test_terminal_mean():
    # Theory: E[S_T] = s0 * exp(mu * T)
    s0, mu, sigma, T, n = 100, 0.08, 0.3, 1.0, 200_000
    s_T = simulate_gbm(s0, mu, sigma, T, n_steps=12, n_paths=n, seed=1)[:, -1]
    standard_error = s_T.std(ddof=1) / np.sqrt(n)
    assert abs(s_T.mean() - s0 * np.exp(mu * T)) < 4 * standard_error


def test_log_return_volatility():
    # Theory: log-returns over one step have standard deviation sigma * sqrt(dt)
    sigma, T, n_steps = 0.25, 1.0, 252
    paths = simulate_gbm(100, 0.0, sigma, T, n_steps, n_paths=2000, seed=2)
    log_returns = np.diff(np.log(paths), axis=1)
    dt = T / n_steps
    assert log_returns.std() == pytest.approx(sigma * np.sqrt(dt), rel=0.01)


def test_log_return_mean():
    # Theory: log-returns over one step have mean (mu - sigma^2 / 2) * dt
    mu, sigma, T, n_steps, n_paths = 0.1, 0.4, 1.0, 50, 20_000
    paths = simulate_gbm(100, mu, sigma, T, n_steps, n_paths, seed=3)
    log_returns = np.diff(np.log(paths), axis=1)
    dt = T / n_steps
    standard_error = log_returns.std(ddof=1) / np.sqrt(log_returns.size)
    assert abs(log_returns.mean() - (mu - 0.5 * sigma**2) * dt) < 4 * standard_error


def test_steps_are_independent():
    # Brownian increments are independent: consecutive log-returns should be uncorrelated
    paths = simulate_gbm(100, 0.0, 0.2, 1.0, n_steps=100, n_paths=5000, seed=4)
    log_returns = np.diff(np.log(paths), axis=1)
    corr = np.corrcoef(log_returns[:, :-1].ravel(), log_returns[:, 1:].ravel())[0, 1]
    # standard error of a correlation estimate is about 1 / sqrt(number of pairs)
    n_pairs = log_returns[:, :-1].size
    assert abs(corr) < 4 / np.sqrt(n_pairs)