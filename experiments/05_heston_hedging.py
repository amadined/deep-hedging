"""Experiment 05: hedging under Heston stochastic volatility.

In the Heston model the volatility moves randomly, so the Black-Scholes
delta is the wrong hedge, and with the stock as the only hedging instrument
no strategy can remove all risk (incomplete market). A short one-month
at-the-money call, rebalanced daily (N = 30), is hedged with:

  - Black-Scholes delta with the fixed starting volatility sqrt(v0),
  - Black-Scholes delta with the current volatility sqrt(v_t),
  - the deep hedger trained on GBM paths in experiment 04 (model risk),
  - a deep hedger trained on Heston paths, without the variance as input,
  - a deep hedger trained on Heston paths, with the volatility as input.

All strategies receive the semi-analytic Heston price as premium.

The GBM-trained models are loaded from experiment 04 (models/); if they are
missing they are trained here with the same settings. The Heston-trained
models are saved in models/ and reused on the next run.

Run from the repository root:
    python experiments/05_heston_hedging.py

Saves figures/heston_comparison.png, figures/heston_incompleteness.png and
figures/heston_skew_adjustment.png, and prints a Markdown table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_hedging.baselines import bs_delta_strategy, bs_delta_strategy_current_vol
from deep_hedging.black_scholes import call_price
from deep_hedging.deep_hedger import (
    DeepHedger,
    deep_hedge_strategy,
    train_hedger,
    train_hedger_heston,
)
from deep_hedging.hedging import hedge_pnl, turnover
from deep_hedging.heston_pricing import heston_call_price
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.models.heston import simulate_heston
from deep_hedging.monte_carlo import call_payoff
from deep_hedging.risk import summarize

# Option (same as experiments 02-04)
S0, K, T, R = 100, 100, 30 / 252, 0.0
N_STEPS = 30
COSTS = [0.0, 0.005]
ALPHA = 0.95

# Heston parameters: 20% volatility on average, strongly random volatility
# (xi = 0.5) and the typical negative correlation between price and volatility
HESTON = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
SIGMA_GBM = np.sqrt(HESTON["v0"])   # volatility of the GBM world in experiment 04

# Training settings (same as experiment 04)
N_ITERS = 1500
BATCH_SIZE = 10_000
TRAIN_SEED = 0

# Test paths
N_TEST, TEST_SEED = 50_000, 2

STRATEGIES = [
    "BS delta, fixed vol",
    "BS delta, current vol",
    "Deep hedge trained on GBM",
    "Deep hedge (Heston), no vol input",
    "Deep hedge (Heston), vol input",
]
FIGURES = Path("figures")
MODELS = Path("models")


def load_or_train(model_file, train_fn, use_variance):
    """Load a saved DeepHedger, or train it with train_fn() and save it."""
    if model_file.exists():
        model = DeepHedger(K, T, use_variance=use_variance)
        model.load_state_dict(torch.load(model_file))
        return model
    print(f"Training {model_file.name} ...")
    model, _ = train_fn()
    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_file)
    return model


def gbm_model(cost):
    """The deep hedger trained on GBM paths (experiment 04)."""
    return load_or_train(
        MODELS / f"deep_hedger_cost{cost}.pt",
        lambda: train_hedger(K, T, SIGMA_GBM, cost, N_STEPS, s0=S0, r=R, n_iters=N_ITERS,
                             batch_size=BATCH_SIZE, seed=TRAIN_SEED),
        use_variance=False,
    )


def heston_model(cost, use_variance):
    """A deep hedger trained on Heston paths, with or without the volatility input."""
    return load_or_train(
        MODELS / f"heston_hedger_cost{cost}_vol{use_variance}.pt",
        lambda: train_hedger_heston(K, T, HESTON, cost, N_STEPS, use_variance=use_variance,
                                    s0=S0, r=R, n_iters=N_ITERS, batch_size=BATCH_SIZE,
                                    seed=TRAIN_SEED),
        use_variance=use_variance,
    )


def main():
    p0 = heston_call_price(S0, K, T, R, **HESTON)
    print(f"Heston premium p0 = {p0:.4f} (Black-Scholes with sigma = {SIGMA_GBM}: "
          f"{call_price(S0, K, T, R, SIGMA_GBM):.4f})\n")

    paths, variances = simulate_heston(s0=S0, mu=R, T=T, n_steps=N_STEPS, n_paths=N_TEST,
                                       seed=TEST_SEED, **HESTON)
    payoff = call_payoff(paths[:, -1], K)

    results = {}    # results[cost][strategy] = summary dict
    deltas = {}     # deltas[cost][strategy] = positions on the test paths
    for cost in COSTS:
        deltas[cost] = {
            "BS delta, fixed vol": bs_delta_strategy(paths, K, T, R, SIGMA_GBM),
            "BS delta, current vol": bs_delta_strategy_current_vol(paths, variances, K, T, R),
            "Deep hedge trained on GBM": deep_hedge_strategy(gbm_model(cost), paths),
            "Deep hedge (Heston), no vol input": deep_hedge_strategy(heston_model(cost, False), paths),
            "Deep hedge (Heston), vol input": deep_hedge_strategy(heston_model(cost, True), paths,
                                                                 variances),
        }
        results[cost] = {}
        for name, d in deltas[cost].items():
            summary = summarize(hedge_pnl(paths, d, payoff, p0, cost), ALPHA)
            summary["turnover"] = turnover(d).mean()
            results[cost][name] = summary

    # For comparison: the same option in the Black-Scholes (GBM) world of experiment 04
    gbm_paths = simulate_gbm(S0, R, SIGMA_GBM, T, N_STEPS, N_TEST, seed=TEST_SEED)
    gbm_payoff = call_payoff(gbm_paths[:, -1], K)
    gbm_p0 = call_price(S0, K, T, R, SIGMA_GBM)
    gbm_cvar = {
        "BS delta": summarize(hedge_pnl(gbm_paths, bs_delta_strategy(gbm_paths, K, T, R, SIGMA_GBM),
                                        gbm_payoff, gbm_p0, 0.0), ALPHA)["cvar"],
        "Deep hedge": summarize(hedge_pnl(gbm_paths, deep_hedge_strategy(gbm_model(0.0), gbm_paths),
                                          gbm_payoff, gbm_p0, 0.0), ALPHA)["cvar"],
    }

    # --- Table -----------------------------------------------------------
    print("| Cost | Strategy | Mean P&L | Std P&L | CVaR 95% | Turnover |")
    print("|---|---|---|---|---|---|")
    for cost in COSTS:
        for name in STRATEGIES:
            s = results[cost][name]
            print(f"| {cost:.1%} | {name} | {s['mean']:.3f} | {s['std']:.3f} "
                  f"| {s['cvar']:.3f} | {s['turnover']:.2f} |")
    print(f"\nBlack-Scholes world, no costs: CVaR of BS delta {gbm_cvar['BS delta']:.3f}, "
          f"deep hedge {gbm_cvar['Deep hedge']:.3f}")

    FIGURES.mkdir(exist_ok=True)

    # --- Figure 1: CVaR of all strategies --------------------------------
    width = 0.16
    x = np.arange(len(COSTS))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(STRATEGIES):
        values = [results[c][name]["cvar"] for c in COSTS]
        ax.bar(x + (i - 2) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cost {c:.1%}" for c in COSTS])
    ax.set_ylabel(f"CVaR {ALPHA:.0%} of loss")
    ax.set_title("Hedging risk under Heston stochastic volatility (N = 30, Heston premium)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "heston_comparison.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: the price of incompleteness (no costs) -----------------
    labels = ["BS delta", "Deep hedge"]
    bs_world = [gbm_cvar["BS delta"], gbm_cvar["Deep hedge"]]
    heston_world = [results[0.0]["BS delta, fixed vol"]["cvar"],
                    results[0.0]["Deep hedge (Heston), vol input"]["cvar"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, bs_world, 0.4, label="Black-Scholes world (constant vol)")
    ax.bar(x + 0.2, heston_world, 0.4, label="Heston world (random vol)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"CVaR {ALPHA:.0%} of loss")
    ax.set_title("Without costs: risk that hedging with the stock cannot remove")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "heston_incompleteness.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: how the network adjusts the delta (no costs) -----------
    # Difference between the deep hedge and the BS delta with the current
    # volatility, at day 15, against the stock price, coloured by volatility.
    j = 15
    n_show = 4000
    diff = (deltas[0.0]["Deep hedge (Heston), vol input"][:n_show, j]
            - deltas[0.0]["BS delta, current vol"][:n_show, j])
    vol_j = np.sqrt(np.maximum(variances[:n_show, j], 0))
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(paths[:n_show, j], diff, c=vol_j, s=4, cmap="viridis", alpha=0.7)
    fig.colorbar(sc, ax=ax, label="current volatility")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(85, 115)
    ax.set_xlabel("stock price S at day 15")
    ax.set_ylabel("deep hedge − BS delta (current vol)")
    ax.set_title("The network holds fewer shares than the Black-Scholes delta (ρ < 0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "heston_skew_adjustment.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved figures to {FIGURES}/")


if __name__ == "__main__":
    main()
