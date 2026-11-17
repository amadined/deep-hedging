import numpy as np
import pytest

from deep_hedging.black_scholes import call_price
from deep_hedging.heston_pricing import heston_call_price, heston_char_func, heston_put_price
from deep_hedging.models.heston import simulate_heston
from deep_hedging.monte_carlo import call_payoff, mc_price

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

# ===========================================================================
# Semi-analytic Heston pricer
# ===========================================================================

# Pricing parameters (risk-neutral): S0, T, r and the Heston parameters
S0, T, R = 100, 1.0, 0.03
HESTON = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)


def _phi(u):
    return heston_char_func(u, S0, HESTON["v0"], HESTON["kappa"], HESTON["theta"],
                            HESTON["xi"], HESTON["rho"], R, T)


# --- Characteristic function ------------------------------------------------

def test_char_func_at_zero_is_one():
    # Every characteristic function satisfies phi(0) = E[exp(0)] = 1
    assert _phi(0.0) == pytest.approx(1.0)


def test_char_func_at_minus_i_is_forward_price():
    # phi(-i) = E[exp(ln S_T)] = E[S_T] = S0 * exp(r * T) under the risk-neutral measure
    assert _phi(-1j) == pytest.approx(S0 * np.exp(R * T))


def test_char_func_is_bounded_by_one():
    # |phi(u)| <= 1 for every real u
    for u in [0.1, 1.0, 5.0, 20.0, 100.0]:
        assert abs(_phi(u)) <= 1.0 + 1e-12


# --- Call and put prices ----------------------------------------------------

@pytest.mark.parametrize(
    "k, expected",
    [(80, 23.924892), (100, 8.802661), (120, 1.135868)],
)
def test_call_reference_values(k, expected):
    # Reference values from QuantLib's AnalyticHestonEngine (independent implementation)
    assert heston_call_price(S0, k, T, R, **HESTON) == pytest.approx(expected, abs=1e-4)


def test_black_scholes_limit():
    # With almost no vol of vol and v0 = theta, the variance stays at theta,
    # so Heston must reduce to Black-Scholes with sigma = sqrt(theta)
    heston = heston_call_price(100, 100, 1.0, 0.03, v0=0.04, kappa=1.5, theta=0.04, xi=1e-4, rho=-0.7)
    bs = call_price(100, 100, 1.0, 0.03, np.sqrt(0.04))
    assert heston == pytest.approx(bs, abs=1e-3)


def test_call_price_bounds_and_monotonicity():
    strikes = [60, 80, 100, 120, 140]
    prices = [heston_call_price(S0, k, T, R, **HESTON) for k in strikes]
    for k, c in zip(strikes, prices):
        # No-arbitrage bounds: max(S0 - K e^{-rT}, 0) <= C <= S0
        assert max(S0 - k * np.exp(-R * T), 0) - 1e-8 <= c <= S0
    # A higher strike can never make a call more valuable
    assert all(a > b for a, b in zip(prices, prices[1:]))


def test_put_price_is_positive_and_parity_holds():
    for k in [80, 100, 120]:
        call = heston_call_price(S0, k, T, R, **HESTON)
        put = heston_put_price(S0, k, T, R, **HESTON)
        assert put > 0
        assert call - put == pytest.approx(S0 - k * np.exp(-R * T))


# --- Simulator and pricer agree ---------------------------------------------

def test_monte_carlo_matches_semi_analytic_price():
    # Risk-neutral simulation (mu = r) with the full-truncation Euler scheme.
    # With 250 steps the discretisation bias is far below the Monte Carlo noise.
    prices, _ = simulate_heston(s0=S0, mu=R, T=T, n_steps=250, n_paths=100_000, seed=5, **HESTON)
    mc, se = mc_price(call_payoff(prices[:, -1], 100), R, T)
    exact = heston_call_price(S0, 100, T, R, **HESTON)
    assert abs(mc - exact) < 4 * se