import numpy as np
import pytest
import torch

from deep_hedging.baselines import bs_delta_strategy
from deep_hedging.black_scholes import call_price
from deep_hedging.deep_hedger import (
    DeepHedger,
    cvar_torch,
    deep_hedge_strategy,
    hedge_pnl_torch,
    train_hedger,
)
from deep_hedging.hedging import hedge_pnl
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.monte_carlo import call_payoff
from deep_hedging.risk import cvar


# ===========================================================================
# hedge_pnl_torch
# ===========================================================================

def test_pnl_torch_hand_example():
    # Same example as for the NumPy version:
    # prices 100 -> 110 -> 105, hold 0.5 then 1.0 shares, payoff 5, premium 3
    paths = torch.tensor([[100.0, 110.0, 105.0]])
    deltas = torch.tensor([[0.5, 1.0]])
    payoff = torch.tensor([5.0])
    assert hedge_pnl_torch(paths, deltas, payoff, 3.0, 0.0).item() == pytest.approx(-2.0)
    assert hedge_pnl_torch(paths, deltas, payoff, 3.0, 0.01).item() == pytest.approx(-4.10)


@pytest.mark.parametrize("cost", [0.0, 0.001, 0.005])
def test_pnl_torch_matches_numpy_version(cost):
    # The PyTorch P&L must give the same numbers as the tested NumPy hedge_pnl
    rng = np.random.default_rng(0)
    paths = simulate_gbm(100, 0.0, 0.2, 30 / 252, 30, 1000, seed=1)
    deltas = rng.uniform(-1, 1, size=(1000, 30))
    payoff = call_payoff(paths[:, -1], 100)

    expected = hedge_pnl(paths, deltas, payoff, 2.75, cost)
    result = hedge_pnl_torch(
        torch.tensor(paths, dtype=torch.float32),
        torch.tensor(deltas, dtype=torch.float32),
        torch.tensor(payoff, dtype=torch.float32),
        2.75,
        cost,
    )
    assert result.shape == (1000,)
    # float32 has about 7 significant digits, so allow a small tolerance
    assert np.allclose(result.numpy(), expected, atol=1e-3)


def test_pnl_torch_lets_gradients_flow_to_the_deltas():
    # Training only works if the P&L is differentiable with respect to the positions
    paths = torch.tensor([[100.0, 110.0, 105.0]])
    deltas = torch.tensor([[0.5, 1.0]], requires_grad=True)
    pnl = hedge_pnl_torch(paths, deltas, torch.tensor([5.0]), 3.0, 0.0)
    pnl.sum().backward()
    # Without costs, d(P&L)/d(delta_k) is the price change S_{k+1} - S_k
    assert torch.allclose(deltas.grad, torch.tensor([[10.0, -5.0]]))


def test_pnl_torch_keeps_float32():
    paths = torch.tensor([[100.0, 110.0, 105.0]])
    deltas = torch.tensor([[0.5, 1.0]])
    pnl = hedge_pnl_torch(paths, deltas, torch.tensor([5.0]), 3.0, 0.01)
    assert pnl.dtype == torch.float32


# ===========================================================================
# cvar_torch
# ===========================================================================

def test_cvar_torch_hand_example():
    # P&L = -1, ..., -100, i.e. losses 1, ..., 100: the worst 5 losses average 98
    pnl = -torch.arange(1.0, 101.0)
    assert cvar_torch(pnl, 0.95).item() == pytest.approx(98.0)


def test_cvar_torch_matches_numpy_version():
    rng = np.random.default_rng(2)
    pnl = rng.standard_normal(10_000)
    result = cvar_torch(torch.tensor(pnl, dtype=torch.float32), 0.95).item()
    assert result == pytest.approx(cvar(pnl, 0.95), rel=1e-4)


def test_cvar_torch_with_few_paths_is_not_nan():
    # With fewer than 20 paths, (1 - alpha) * n < 1; at least the worst path must be used
    pnl = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    result = cvar_torch(pnl, 0.95)
    assert not torch.isnan(result)
    assert result.item() == pytest.approx(4.0)


def test_cvar_torch_gradient_only_reaches_the_worst_paths():
    # The loss only depends on the worst 5% of paths, so only they get a gradient
    pnl = (-torch.arange(1.0, 101.0)).requires_grad_()
    cvar_torch(pnl, 0.95).backward()
    worst_five = slice(95, 100)  # P&L -96, ..., -100
    assert torch.all(pnl.grad[worst_five] != 0)
    assert torch.all(pnl.grad[:95] == 0)

# ===========================================================================
# DeepHedger
# ===========================================================================

K, T, N_STEPS = 100, 30 / 252, 30


def _paths(n_paths, seed=0):
    paths = simulate_gbm(100, 0.0, 0.2, T, N_STEPS, n_paths, seed=seed)
    return torch.tensor(paths, dtype=torch.float32)


def test_model_output_shape_and_dtype():
    torch.manual_seed(0)
    model = DeepHedger(k=K, T=T)
    deltas = model(_paths(50))
    assert deltas.shape == (50, N_STEPS)
    assert deltas.dtype == torch.float32
    assert torch.all(torch.isfinite(deltas))


def test_model_has_expected_number_of_weights():
    # 3 -> 32 -> 32 -> 1: (3*32 + 32) + (32*32 + 32) + (32*1 + 1) = 128 + 1056 + 33 = 1217
    model = DeepHedger(k=K, T=T, hidden=32)
    assert sum(p.numel() for p in model.parameters()) == 1217


def test_first_decision_is_the_same_on_every_path():
    # At t_0 all paths have the same price, time and previous position (0)
    torch.manual_seed(0)
    model = DeepHedger(k=K, T=T)
    deltas = model(_paths(50))
    assert torch.allclose(deltas[:, 0], deltas[0, 0].expand(50))


def test_model_is_reproducible_with_manual_seed():
    paths = _paths(20)
    torch.manual_seed(42)
    a = DeepHedger(k=K, T=T)(paths)
    torch.manual_seed(42)
    b = DeepHedger(k=K, T=T)(paths)
    assert torch.equal(a, b)


def test_model_has_no_look_ahead():
    # Changing the price at maturity must not change any decision
    torch.manual_seed(0)
    model = DeepHedger(k=K, T=T)
    paths = _paths(20)
    changed = paths.clone()
    changed[:, -1] *= 2.0
    with torch.no_grad():
        assert torch.equal(model(paths), model(changed))


def test_model_only_uses_past_prices():
    # Changing the price at t_10 may change decisions from t_10 on, but never before
    torch.manual_seed(0)
    model = DeepHedger(k=K, T=T)
    paths = _paths(20)
    changed = paths.clone()
    changed[:, 10] *= 1.1
    with torch.no_grad():
        a, b = model(paths), model(changed)
    assert torch.equal(a[:, :10], b[:, :10])
    assert not torch.equal(a[:, 10:], b[:, 10:])


def test_previous_position_is_an_input():
    # The decision at t_11 depends on the price at t_10 only through the previous
    # position: t_11's own inputs (its price and time) are unchanged.
    # An untrained network reacts only weakly to its inputs, so use a large price
    # change and compare exactly: any dependence at all makes the outputs differ.
    torch.manual_seed(0)
    model = DeepHedger(k=K, T=T)
    paths = _paths(20)
    changed = paths.clone()
    changed[:, 10] *= 1.5
    with torch.no_grad():
        a, b = model(paths), model(changed)
    assert not torch.equal(a[:, 11], b[:, 11])


def test_gradients_reach_all_weights():
    # The CVaR of the hedging P&L must be differentiable with respect to every weight
    torch.manual_seed(0)
    model = DeepHedger(k=K, T=T)
    paths = _paths(1000)
    payoff = torch.clamp(paths[:, -1] - K, min=0)
    loss = cvar_torch(hedge_pnl_torch(paths, model(paths), payoff, 2.75, 0.001))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.any(p.grad != 0), f"zero gradient for {name}"

def test_model_inputs_are_correct():
    # Record what the MLP receives at every time step and check each input column:
    # 0 = log(S_j / K), 1 = time to maturity T - j * dt, 2 = previous position
    torch.manual_seed(0)
    model = DeepHedger(k=K, T=T)
    paths = _paths(10)
    inputs = []
    hook = model.net.register_forward_hook(lambda module, args, output: inputs.append(args[0]))
    with torch.no_grad():
        deltas = model(paths)
    hook.remove()

    assert len(inputs) == N_STEPS
    dt = T / N_STEPS
    for j, x in enumerate(inputs):
        assert torch.allclose(x[:, 0], torch.log(paths[:, j] / K))
        assert torch.allclose(x[:, 1], torch.full((10,), T - j * dt))
        previous = torch.zeros(10) if j == 0 else deltas[:, j - 1]
        assert torch.allclose(x[:, 2], previous)

# ===========================================================================
# Training
# ===========================================================================

# Small settings so the tests run in a few seconds (real training uses more)
FAST = dict(k=100, T=30 / 252, sigma=0.2, n_steps=10, n_iters=200,
            batch_size=2000, lr=5e-3, print_every=0)


def _cvar_on_test_paths(deltas_fn, cost):
    """CVaR of a strategy on fixed test paths, computed with the NumPy tools."""
    paths = simulate_gbm(100, 0.0, 0.2, T, FAST["n_steps"], 5000, seed=0)
    p0 = call_price(100, 100, T, 0.0, 0.2)
    pnl = hedge_pnl(paths, deltas_fn(paths), call_payoff(paths[:, -1], 100), p0, cost)
    return cvar(pnl)


def test_train_returns_model_and_losses():
    model, losses = train_hedger(cost=0.0, **{**FAST, "n_iters": 20})
    assert isinstance(model, DeepHedger)
    assert len(losses) == 20
    assert all(np.isfinite(losses))


def test_training_reduces_the_risk_on_new_paths():
    # Untrained, the network hedges randomly. After a short training its CVaR on
    # paths it has never seen must be far lower.
    torch.manual_seed(0)
    untrained = DeepHedger(k=100, T=T)
    trained, _ = train_hedger(cost=0.001, **FAST)
    cvar_untrained = _cvar_on_test_paths(lambda p: deep_hedge_strategy(untrained, p), 0.001)
    cvar_trained = _cvar_on_test_paths(lambda p: deep_hedge_strategy(trained, p), 0.001)
    assert cvar_trained < 0.5 * cvar_untrained


def test_short_training_gets_close_to_delta_hedging():
    # Even a short training run should come close to the Black-Scholes delta hedge.
    # This also catches a sign error in the loss (minimising the CVaR of the P&L
    # instead of the losses), which trains to a clearly worse strategy.
    trained, _ = train_hedger(cost=0.001, **FAST)
    cvar_deep = _cvar_on_test_paths(lambda p: deep_hedge_strategy(trained, p), 0.001)
    cvar_delta = _cvar_on_test_paths(lambda p: bs_delta_strategy(p, 100, T, 0.0, 0.2), 0.001)
    assert cvar_deep < 1.15 * cvar_delta


def test_training_is_reproducible_with_seed():
    model_a, losses_a = train_hedger(cost=0.001, seed=3, **{**FAST, "n_iters": 5})
    model_b, losses_b = train_hedger(cost=0.001, seed=3, **{**FAST, "n_iters": 5})
    assert losses_a == losses_b
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)


def test_different_seeds_give_different_models():
    _, losses_a = train_hedger(cost=0.001, seed=1, **{**FAST, "n_iters": 5})
    _, losses_b = train_hedger(cost=0.001, seed=2, **{**FAST, "n_iters": 5})
    assert losses_a != losses_b


def test_deep_hedge_strategy_returns_numpy_deltas():
    torch.manual_seed(0)
    model = DeepHedger(k=K, T=T)
    paths = simulate_gbm(100, 0.0, 0.2, T, N_STEPS, 40, seed=0)
    deltas = deep_hedge_strategy(model, paths)
    assert isinstance(deltas, np.ndarray)
    assert deltas.shape == (40, N_STEPS)
    assert deltas.dtype == np.float64
    # Same numbers as calling the model directly
    with torch.no_grad():
        direct = model(torch.tensor(paths, dtype=torch.float32)).numpy()
    assert np.allclose(deltas, direct)