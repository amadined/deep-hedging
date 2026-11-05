import numpy as np
import pytest

from deep_hedging.black_scholes import call_delta, call_price, put_price


def test_textbook_values():
    # Calculated example: S=K=100, T=1, r=5%, sigma=20%
    assert call_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-4)
    assert put_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(5.5735, abs=1e-4)

    # textbook example from J.C. Hull
    assert call_price(42, 40, 0.5, 0.1, 0.2) == pytest.approx(4.76, abs=0.005)
    assert put_price(42, 40, 0.5, 0.1, 0.2) == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("s", [60, 90, 100, 110, 150])
@pytest.mark.parametrize("t", [0.05, 0.5, 2.0])
def test_put_call_parity(s, t):
    # C - P = S - K * exp(-rT) must hold for any model without arbitrage
    k, r, sigma = 100, 0.03, 0.25
    lhs = call_price(s, k, t, r, sigma) - put_price(s, k, t, r, sigma)
    rhs = s - k * np.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_delta_matches_finite_difference():
    # Delta is the derivative of the price with respect to S
    s, k, t, r, sigma, h = 105, 100, 0.25, 0.01, 0.2, 1e-4
    numerical = (call_price(s + h, k, t, r, sigma) - call_price(s - h, k, t, r, sigma)) / (2 * h)
    assert call_delta(s, k, t, r, sigma) == pytest.approx(numerical, abs=1e-6)


def test_delta_bounds_and_monotonicity():
    s = np.linspace(50, 150, 101)
    delta = call_delta(s, 100, 0.5, 0.0, 0.2)
    assert np.all((delta > 0) & (delta < 1))
    assert np.all(np.diff(delta) > 0)  # more shares as the option goes into the money
