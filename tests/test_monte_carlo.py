import numpy as np
import pytest

from deep_hedging.black_scholes import call_price, put_price
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.monte_carlo import call_payoff, mc_price, put_payoff


S0, K, T, R, SIGMA = 100, 105, 0.5, 0.05, 0.2


# --- Payoff functions -------------------------------------------------------

def test_call_payoff():
    s_T = np.array([80.0, 100.0, 120.0])
    assert np.array_equal(call_payoff(s_T, 100), [0.0, 0.0, 20.0])


def test_put_payoff():
    s_T = np.array([80.0, 100.0, 120.0])
    assert np.array_equal(put_payoff(s_T, 100), [20.0, 0.0, 0.0])


def test_payoffs_are_never_negative():
    s_T = np.linspace(1, 200, 500)
    assert np.all(call_payoff(s_T, 100) >= 0)
    assert np.all(put_payoff(s_T, 100) >= 0)


def test_put_call_parity_at_maturity():
    # (S_T - K)^+ - (K - S_T)^+ = S_T - K for every S_T
    s_T = np.linspace(1, 200, 500)
    diff = call_payoff(s_T, 100) - put_payoff(s_T, 100)
    assert np.allclose(diff, s_T - 100)


# --- mc_price on known data -----------------------------------------------

def test_constant_payoffs_without_discounting():
    # Also checks that a plain Python list is accepted
    price, se = mc_price([10, 10, 10], r=0.0, T=1.0)
    assert price == pytest.approx(10.0)
    assert se == pytest.approx(0.0)


def test_discounting():
    payoffs = np.array([0.0, 5.0, 10.0, 25.0])
    price, _ = mc_price(payoffs, r=0.05, T=2.0)
    assert price == pytest.approx(np.exp(-0.05 * 2.0) * payoffs.mean())


def test_standard_error_formula():
    # Standard error = sample standard deviation (ddof=1) / sqrt(n)
    payoffs = np.array([1.0, 2.0, 3.0, 4.0])
    _, se = mc_price(payoffs, r=0.0, T=1.0)
    assert se == pytest.approx(np.std(payoffs, ddof=1) / 2.0)


# --- Against Black-Scholes ------------------------------------------------

@pytest.mark.parametrize("n_steps", [1, 30])
def test_mc_call_matches_black_scholes(n_steps):
    # Simulate with the risk-neutral drift mu = r. Because the simulator is
    # exact, 1 step and 30 steps must both match.
    s_T = simulate_gbm(S0, R, SIGMA, T, n_steps, n_paths=200_000, seed=3)[:, -1]
    price, se = mc_price(call_payoff(s_T, K), R, T)
    assert abs(price - call_price(S0, K, T, R, SIGMA)) < 4 * se


def test_mc_put_matches_black_scholes():
    s_T = simulate_gbm(S0, R, SIGMA, T, 1, n_paths=200_000, seed=4)[:, -1]
    price, se = mc_price(put_payoff(s_T, K), R, T)
    assert abs(price - put_price(S0, K, T, R, SIGMA)) < 4 * se


# --- Convergence ------------------------------------------------------------

def test_standard_error_shrinks_like_one_over_sqrt_n():
    s_T_small = simulate_gbm(S0, R, SIGMA, T, 1, n_paths=10_000, seed=5)[:, -1]
    s_T_large = simulate_gbm(S0, R, SIGMA, T, 1, n_paths=160_000, seed=6)[:, -1]
    _, se_small = mc_price(call_payoff(s_T_small, K), R, T)
    _, se_large = mc_price(call_payoff(s_T_large, K), R, T)
    # 16x more paths -> standard error about sqrt(16) = 4x smaller
    assert se_small / se_large == pytest.approx(4.0, rel=0.1)