import numpy as np
import pytest

from deep_hedging.baselines import (
    bs_delta_strategy,
    bs_delta_strategy_current_vol,
    leland_sigma,
    leland_strategy,
    whalley_wilmott_strategy,
)
from deep_hedging.black_scholes import call_delta, call_gamma, call_price
from deep_hedging.hedging import hedge_pnl
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.models.heston import simulate_heston
from deep_hedging.monte_carlo import call_payoff
from deep_hedging.risk import cvar, summarize, var


# ===========================================================================
# hedge_pnl
# ===========================================================================

# One path, two steps: prices 100 -> 110 -> 105, hold 0.5 then 1.0 shares
PATH = np.array([[100.0, 110.0, 105.0]])
DELTAS = np.array([[0.5, 1.0]])
PAYOFF = np.array([5.0])
P0 = 3.0


def test_pnl_without_hedging_is_premium_minus_payoff():
    paths = simulate_gbm(100, 0.0, 0.2, 1.0, 10, 1000, seed=0)
    payoff = call_payoff(paths[:, -1], 100)
    pnl = hedge_pnl(paths, np.zeros((1000, 10)), payoff, p0=8.0, cost=0.0)
    assert np.allclose(pnl, 8.0 - payoff)


def test_pnl_hand_example_without_costs():
    # Gains: 0.5 * (110 - 100) + 1.0 * (105 - 110) = 5 - 5 = 0
    # P&L:   3 - 5 + 0 = -2
    pnl = hedge_pnl(PATH, DELTAS, PAYOFF, P0, cost=0.0)
    assert pnl.shape == (1,)
    assert pnl[0] == pytest.approx(-2.0)


def test_pnl_hand_example_with_costs():
    # Trades: buy 0.5 at 100, buy 0.5 at 110, sell 1.0 at 105 
    # Costs:  0.01 * (0.5 * 100 + 0.5 * 110 + 1.0 * 105) = 0.01 * 210 = 2.10
    # P&L:    -2 - 2.10 = -4.10
    pnl = hedge_pnl(PATH, DELTAS, PAYOFF, P0, cost=0.01)
    assert pnl[0] == pytest.approx(-4.10)


def test_short_positions_also_cost():
    # Costs depend on |trade|, so selling short costs the same as buying.
    # In this example the trading gains are 0 for both, so the P&Ls must match.
    pnl_long = hedge_pnl(PATH, DELTAS, PAYOFF, P0, cost=0.01)
    pnl_short = hedge_pnl(PATH, -DELTAS, PAYOFF, P0, cost=0.01)
    assert pnl_long[0] == pytest.approx(pnl_short[0])


def test_costs_never_increase_pnl():
    rng = np.random.default_rng(1)
    paths = simulate_gbm(100, 0.0, 0.2, 1.0, 20, 500, seed=2)
    deltas = rng.uniform(-1, 1, size=(500, 20))
    payoff = call_payoff(paths[:, -1], 100)
    pnl_free = hedge_pnl(paths, deltas, payoff, 8.0, cost=0.0)
    pnl_costly = hedge_pnl(paths, deltas, payoff, 8.0, cost=0.005)
    assert np.all(pnl_costly <= pnl_free)
    assert np.any(pnl_costly < pnl_free)


# ===========================================================================
# Risk measures
# ===========================================================================

def test_var_and_cvar_hand_example():
    # P&L = -1, ..., -100, i.e. losses 1, ..., 100.
    # The worst 5% of losses are 96, ..., 100, with mean 98.
    pnl = -np.arange(1, 101, dtype=float)
    assert var(pnl, 0.95) == pytest.approx(95.05)
    assert cvar(pnl, 0.95) == pytest.approx(98.0)


def test_cvar_can_be_negative_if_even_the_worst_paths_make_money():
    # P&L = 1, ..., 100: the worst 5% are the profits 1, ..., 5 (losses -1, ..., -5)
    pnl = np.arange(1, 101, dtype=float)
    assert cvar(pnl, 0.95) == pytest.approx(-3.0)


def test_cvar_of_standard_normal():
    # Known value: CVaR_95 of a standard normal is pdf(1.645) / 0.05 = 2.0627
    rng = np.random.default_rng(3)
    pnl = rng.standard_normal(1_000_000)
    assert cvar(pnl, 0.95) == pytest.approx(2.0627, abs=0.01)


def test_cvar_at_least_var_and_sensitive_to_tail():
    rng = np.random.default_rng(4)
    pnl = rng.standard_normal(10_000)
    assert cvar(pnl) >= var(pnl)
    # Make the single worst outcome much worse: CVaR must increase, VaR need not
    worse = pnl.copy()
    worse[np.argmin(worse)] -= 100
    assert cvar(worse) > cvar(pnl)


def test_risk_measures_accept_lists():
    assert cvar([-1.0, -2.0, -3.0, -4.0], alpha=0.5) == pytest.approx(3.5)


def test_summarize():
    pnl = -np.arange(1, 101, dtype=float)
    s = summarize(pnl)
    assert s["mean"] == pytest.approx(-50.5)
    assert s["std"] == pytest.approx(np.std(pnl, ddof=1))
    assert s["var"] == pytest.approx(95.05)
    assert s["cvar"] == pytest.approx(98.0)


# ===========================================================================
# Black-Scholes delta hedging
# ===========================================================================

S0, K, T, SIGMA = 100, 100, 30 / 252, 0.2


def _delta_hedge_pnl(n_steps, cost, n_paths=50_000, seed=10):
    # Risk-neutral world with r = mu = 0, premium = Black-Scholes price
    paths = simulate_gbm(S0, 0.0, SIGMA, T, n_steps, n_paths, seed=seed)
    deltas = bs_delta_strategy(paths, K, T, 0.0, SIGMA)
    p0 = call_price(S0, K, T, 0.0, SIGMA)
    return hedge_pnl(paths, deltas, call_payoff(paths[:, -1], K), p0, cost)


def test_delta_strategy_shape_and_first_delta():
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 100, seed=0)
    deltas = bs_delta_strategy(paths, K, T, 0.0, SIGMA)
    assert deltas.shape == (100, 30)
    # At t_0 every path is at S0, so every delta equals the BS delta at S0
    assert np.allclose(deltas[:, 0], call_delta(S0, K, T, 0.0, SIGMA))


def test_delta_strategy_has_no_look_ahead():
    # Changing the price at maturity must not change any decision
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 100, seed=0)
    changed = paths.copy()
    changed[:, -1] *= 2.0
    assert np.array_equal(bs_delta_strategy(paths, K, T, 0.0, SIGMA),
                          bs_delta_strategy(changed, K, T, 0.0, SIGMA))


def test_delta_strategy_uses_remaining_time():
    # Delta at step j must use time to maturity T - j * dt
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 10, seed=0)
    deltas = bs_delta_strategy(paths, K, T, 0.0, SIGMA)
    j, dt = 17, T / 30
    assert np.allclose(deltas[:, j], call_delta(paths[:, j], K, T - j * dt, 0.0, SIGMA))


def test_mean_pnl_is_zero_without_costs():
    pnl = _delta_hedge_pnl(n_steps=30, cost=0.0)
    standard_error = pnl.std(ddof=1) / np.sqrt(pnl.size)
    assert abs(pnl.mean()) < 4 * standard_error


def test_hedging_error_shrinks_like_one_over_sqrt_n():
    # Four times more rebalancing -> hedging error about half as large
    std_100 = _delta_hedge_pnl(n_steps=100, cost=0.0).std()
    std_400 = _delta_hedge_pnl(n_steps=400, cost=0.0).std()
    assert std_100 / std_400 == pytest.approx(2.0, rel=0.1)


def test_hedge_reduces_risk_compared_to_no_hedge():
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 50_000, seed=11)
    payoff = call_payoff(paths[:, -1], K)
    p0 = call_price(S0, K, T, 0.0, SIGMA)
    hedged = hedge_pnl(paths, bs_delta_strategy(paths, K, T, 0.0, SIGMA), payoff, p0, 0.0)
    unhedged = hedge_pnl(paths, np.zeros((50_000, 30)), payoff, p0, 0.0)
    assert hedged.std() < 0.2 * unhedged.std()
    assert cvar(hedged) < cvar(unhedged)


def test_costs_make_mean_pnl_negative():
    pnl = _delta_hedge_pnl(n_steps=30, cost=0.001)
    standard_error = pnl.std(ddof=1) / np.sqrt(pnl.size)
    assert pnl.mean() < -4 * standard_error

# ===========================================================================
# Leland
# ===========================================================================

def test_leland_sigma_hand_value():
    # sigma = 0.2, c = 0.005 (round trip 0.01), dt = 1/252:
    # Leland number = sqrt(2/pi) * 0.01 / (0.2 * sqrt(1/252)) = 0.6334
    # sigma_hat = 0.2 * sqrt(1 + 0.6334) = 0.2556
    assert leland_sigma(0.2, 0.005, 1 / 252) == pytest.approx(0.25561, abs=1e-4)


def test_leland_sigma_without_costs_is_sigma():
    assert leland_sigma(0.2, 0.0, 1 / 252) == pytest.approx(0.2)


def test_leland_sigma_increases_with_costs_and_rebalancing_frequency():
    assert leland_sigma(0.2, 0.001, 1 / 252) > 0.2
    assert leland_sigma(0.2, 0.005, 1 / 252) > leland_sigma(0.2, 0.001, 1 / 252)
    # Smaller dt = more frequent rebalancing = more costs = larger adjustment
    assert leland_sigma(0.2, 0.001, 1 / 1000) > leland_sigma(0.2, 0.001, 1 / 252)


def test_leland_strategy_without_costs_equals_delta_hedge():
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 200, seed=0)
    assert np.allclose(
        leland_strategy(paths, K, T, 0.0, SIGMA, cost=0.0),
        bs_delta_strategy(paths, K, T, 0.0, SIGMA),
    )


def test_leland_strategy_uses_adjusted_sigma_with_correct_dt():
    # With N = 30 steps, dt must be T / 30 (not T / 31)
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 200, seed=0)
    sigma_hat = leland_sigma(SIGMA, 0.005, T / 30)
    assert np.allclose(
        leland_strategy(paths, K, T, 0.0, SIGMA, cost=0.005),
        bs_delta_strategy(paths, K, T, 0.0, sigma_hat),
    )


# ===========================================================================
# Whalley-Wilmott
# ===========================================================================

def _turnover(deltas):
    """Average total number of shares traded per path, including entry and exit."""
    n_paths = deltas.shape[0]
    positions = np.hstack((np.zeros((n_paths, 1)), deltas, np.zeros((n_paths, 1))))
    return np.abs(np.diff(positions, axis=1)).sum(axis=1).mean()


def test_ww_without_costs_equals_delta_hedge():
    # With cost = 0 the band has width 0
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 200, seed=0)
    assert np.allclose(
        whalley_wilmott_strategy(paths, K, T, 0.0, SIGMA, cost=0.0, risk_aversion=1.0),
        bs_delta_strategy(paths, K, T, 0.0, SIGMA),
    )


def test_ww_shape():
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 200, seed=0)
    deltas = whalley_wilmott_strategy(paths, K, T, 0.0, SIGMA, cost=0.005, risk_aversion=1.0)
    assert deltas.shape == (200, 30)


def test_ww_positions_stay_inside_the_band():
    cost, risk_aversion, n_steps = 0.005, 1.0, 30
    paths = simulate_gbm(S0, 0.0, SIGMA, T, n_steps, 500, seed=1)
    deltas = whalley_wilmott_strategy(paths, K, T, 0.0, SIGMA, cost, risk_aversion)
    dt = T / n_steps
    for j in range(n_steps):
        s, tau = paths[:, j], T - j * dt
        delta = call_delta(s, K, tau, 0.0, SIGMA)
        gamma = call_gamma(s, K, tau, 0.0, SIGMA)
        h = (1.5 * cost * s * gamma**2 / risk_aversion) ** (1 / 3)
        assert np.all(deltas[:, j] >= delta - h - 1e-12)
        assert np.all(deltas[:, j] <= delta + h + 1e-12)


def test_ww_only_trades_when_outside_the_band():
    # Whenever the position changes, the new position must lie on a band edge
    cost, risk_aversion, n_steps = 0.005, 1.0, 30
    paths = simulate_gbm(S0, 0.0, SIGMA, T, n_steps, 500, seed=2)
    deltas = whalley_wilmott_strategy(paths, K, T, 0.0, SIGMA, cost, risk_aversion)
    dt = T / n_steps
    for j in range(1, n_steps):
        s, tau = paths[:, j], T - j * dt
        delta = call_delta(s, K, tau, 0.0, SIGMA)
        h = (1.5 * cost * s * call_gamma(s, K, tau, 0.0, SIGMA) ** 2 / risk_aversion) ** (1 / 3)
        traded = deltas[:, j] != deltas[:, j - 1]
        on_edge = np.isclose(deltas[:, j], delta - h) | np.isclose(deltas[:, j], delta + h)
        assert np.all(on_edge[traded])


def test_ww_trades_less_than_delta_hedge():
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 100, 5000, seed=3)
    ww = whalley_wilmott_strategy(paths, K, T, 0.0, SIGMA, cost=0.005, risk_aversion=3.0)
    bs = bs_delta_strategy(paths, K, T, 0.0, SIGMA)
    assert _turnover(ww) < 0.6 * _turnover(bs)


def test_ww_with_huge_risk_aversion_approaches_delta_hedge():
    # A very risk-averse hedger has an almost zero band
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 200, seed=4)
    ww = whalley_wilmott_strategy(paths, K, T, 0.0, SIGMA, cost=0.005, risk_aversion=1e9)
    assert np.allclose(ww, bs_delta_strategy(paths, K, T, 0.0, SIGMA), atol=1e-3)


def test_ww_has_no_look_ahead():
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 100, seed=5)
    changed = paths.copy()
    changed[:, -1] *= 2.0
    assert np.array_equal(
        whalley_wilmott_strategy(paths, K, T, 0.0, SIGMA, 0.005, 1.0),
        whalley_wilmott_strategy(changed, K, T, 0.0, SIGMA, 0.005, 1.0),
    )


# ===========================================================================
# Both strategies beat plain delta hedging when costs are high
# ===========================================================================

def test_cost_aware_strategies_reduce_risk():
    cost, n_steps = 0.005, 100
    paths = simulate_gbm(S0, 0.0, SIGMA, T, n_steps, 50_000, seed=6)
    payoff = call_payoff(paths[:, -1], K)
    p0 = call_price(S0, K, T, 0.0, SIGMA)

    def risk(deltas):
        return cvar(hedge_pnl(paths, deltas, payoff, p0, cost))

    cvar_bs = risk(bs_delta_strategy(paths, K, T, 0.0, SIGMA))
    cvar_leland = risk(leland_strategy(paths, K, T, 0.0, SIGMA, cost))
    cvar_ww = risk(whalley_wilmott_strategy(paths, K, T, 0.0, SIGMA, cost, risk_aversion=3.0))
    assert cvar_leland < cvar_bs
    assert cvar_ww < cvar_bs


# ===========================================================================
# Black-Scholes delta with the current volatility
# ===========================================================================

def test_current_vol_delta_with_constant_variance_equals_delta_hedge():
    # If the variance never changes, using the current volatility is the same
    # as using the fixed volatility sqrt(v)
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 200, seed=0)
    variances = np.full_like(paths, SIGMA**2)
    assert np.allclose(
        bs_delta_strategy_current_vol(paths, variances, K, T, 0.0),
        bs_delta_strategy(paths, K, T, 0.0, SIGMA),
    )


def test_current_vol_delta_uses_the_variance_at_each_step():
    heston = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
    paths, variances = simulate_heston(s0=S0, mu=0.0, T=T, n_steps=30, n_paths=100, seed=1, **heston)
    deltas = bs_delta_strategy_current_vol(paths, variances, K, T, 0.0)
    assert deltas.shape == (100, 30)
    j, dt = 12, T / 30
    sigma_j = np.sqrt(np.maximum(variances[:, j], 1e-8))
    assert np.allclose(deltas[:, j], call_delta(paths[:, j], K, T - j * dt, 0.0, sigma_j))


def test_current_vol_delta_handles_zero_and_negative_variances():
    paths = simulate_gbm(S0, 0.0, SIGMA, T, 30, 50, seed=0)
    variances = np.full_like(paths, SIGMA**2)
    variances[:, 5] = 0.0
    variances[:, 6] = -0.001
    assert np.all(np.isfinite(bs_delta_strategy_current_vol(paths, variances, K, T, 0.0)))


def test_current_vol_delta_has_no_look_ahead():
    heston = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
    paths, variances = simulate_heston(s0=S0, mu=0.0, T=T, n_steps=30, n_paths=100, seed=2, **heston)
    changed = variances.copy()
    changed[:, -1] *= 3.0
    assert np.array_equal(
        bs_delta_strategy_current_vol(paths, variances, K, T, 0.0),
        bs_delta_strategy_current_vol(paths, changed, K, T, 0.0),
    )

