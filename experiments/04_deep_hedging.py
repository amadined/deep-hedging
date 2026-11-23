"""Experiment 04: deep hedging compared with the classical strategies.

For each transaction cost level, a neural network (DeepHedger) is trained by
minimising the CVaR of the hedging losses of a short one-month at-the-money
call, rebalanced daily (N = 30). It is then evaluated on fresh test paths,
together with Black-Scholes delta hedging, Leland and Whalley-Wilmott.
All strategies receive the same Black-Scholes premium.

Trained models are saved in models/ and loaded on the next run, so the
figures can be regenerated without retraining. Delete models/ to retrain.

Run from the repository root:
    python experiments/04_deep_hedging.py

Saves figures/deep_training.png, figures/deep_vs_delta.png,
figures/deep_comparison.png and figures/deep_no_trade.png,
and prints a Markdown table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_hedging.baselines import (
    bs_delta_strategy,
    leland_strategy,
    whalley_wilmott_strategy,
)
from deep_hedging.black_scholes import call_delta, call_price
from deep_hedging.deep_hedger import DeepHedger, deep_hedge_strategy, train_hedger
from deep_hedging.hedging import hedge_pnl, turnover
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.monte_carlo import call_payoff
from deep_hedging.risk import cvar, summarize

# Option and market (same as experiments 02 and 03)
S0, K, T, R, SIGMA = 100, 100, 30 / 252, 0.0, 0.2
N_STEPS = 30
COSTS = [0.0, 0.001, 0.005]
ALPHA = 0.95

# Training settings
N_ITERS = 1500
BATCH_SIZE = 10_000
TRAIN_SEED = 0

# Evaluation (same test paths as experiment 03)
N_TEST, TEST_SEED = 50_000, 2
N_WW_TRAIN, WW_TRAIN_SEED = 20_000, 1
RISK_AVERSIONS = [0.3, 1, 3, 10, 30, 100]

STRATEGIES = ["Delta", "Leland", "Whalley-Wilmott", "Deep hedge"]
FIGURES = Path("figures")
MODELS = Path("models")


def get_model(cost):
    """Load the trained model for this cost level, or train and save it."""
    model_file = MODELS / f"deep_hedger_cost{cost}.pt"
    loss_file = MODELS / f"deep_hedger_cost{cost}_losses.npy"
    if model_file.exists():
        model = DeepHedger(K, T)
        model.load_state_dict(torch.load(model_file))
        return model, np.load(loss_file)

    print(f"Training deep hedger for cost {cost:.1%} ...")
    model, losses = train_hedger(K, T, SIGMA, cost, N_STEPS, s0=S0, r=R,
                                 n_iters=N_ITERS, batch_size=BATCH_SIZE, seed=TRAIN_SEED)
    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_file)
    np.save(loss_file, np.array(losses))
    return model, np.array(losses)


def best_risk_aversion(cost, p0):
    """Whalley-Wilmott risk aversion with the lowest CVaR on separate paths."""
    if cost == 0.0:
        return 1.0  # without costs the band has width 0, any value works
    paths = simulate_gbm(S0, R, SIGMA, T, N_STEPS, N_WW_TRAIN, seed=WW_TRAIN_SEED)
    payoff = call_payoff(paths[:, -1], K)
    cvars = [cvar(hedge_pnl(paths, whalley_wilmott_strategy(paths, K, T, R, SIGMA, cost, ra),
                            payoff, p0, cost), ALPHA)
             for ra in RISK_AVERSIONS]
    return RISK_AVERSIONS[int(np.argmin(cvars))]


def main():
    p0 = call_price(S0, K, T, R, SIGMA)
    test_paths = simulate_gbm(S0, R, SIGMA, T, N_STEPS, N_TEST, seed=TEST_SEED)
    test_payoff = call_payoff(test_paths[:, -1], K)

    results = {}    # results[cost][strategy] = summary dict
    deltas = {}     # deltas[cost][strategy] = positions on the test paths
    losses = {}     # training curves of the deep hedgers

    for cost in COSTS:
        model, losses[cost] = get_model(cost)
        ra = best_risk_aversion(cost, p0)
        deltas[cost] = {
            "Delta": bs_delta_strategy(test_paths, K, T, R, SIGMA),
            "Leland": leland_strategy(test_paths, K, T, R, SIGMA, cost),
            "Whalley-Wilmott": whalley_wilmott_strategy(test_paths, K, T, R, SIGMA, cost, ra),
            "Deep hedge": deep_hedge_strategy(model, test_paths),
        }
        results[cost] = {}
        for name, d in deltas[cost].items():
            summary = summarize(hedge_pnl(test_paths, d, test_payoff, p0, cost), ALPHA)
            summary["turnover"] = turnover(d).mean()
            results[cost][name] = summary

    # --- Table -----------------------------------------------------------
    print("\n| Cost | Strategy | Mean P&L | Std P&L | CVaR 95% | Turnover |")
    print("|---|---|---|---|---|---|")
    for cost in COSTS:
        for name in STRATEGIES:
            s = results[cost][name]
            print(f"| {cost:.1%} | {name} | {s['mean']:.3f} | {s['std']:.3f} "
                  f"| {s['cvar']:.3f} | {s['turnover']:.2f} |")

    FIGURES.mkdir(exist_ok=True)

    # --- Figure 1: training curves ---------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for cost in COSTS:
        ax.plot(losses[cost], linewidth=0.8, label=f"cost {cost:.1%}")
    ax.set_xlabel("training iteration")
    ax.set_ylabel(f"CVaR {ALPHA:.0%} loss (training batch)")
    ax.set_ylim(0, 6)
    ax.set_title("Training of the deep hedgers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "deep_training.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: learned hedge vs Black-Scholes delta (no costs) ---------
    # At three times, plot the network's position on each test path against
    # the stock price, together with the Black-Scholes delta curve.
    dt = T / N_STEPS
    steps = [0, 15, 29]
    s_grid = np.linspace(85, 115, 200)
    fig, axes = plt.subplots(1, len(steps), figsize=(12, 3.8), sharey=True)
    for ax, j in zip(axes, steps):
        tau = T - j * dt
        s_j = test_paths[:2000, j]
        ax.scatter(s_j, deltas[0.0]["Deep hedge"][:2000, j], s=3, alpha=0.4,
                   label="deep hedge (test paths)")
        ax.plot(s_grid, call_delta(s_grid, K, tau, R, SIGMA), color="black",
                linewidth=1.2, label="Black-Scholes delta")
        ax.set_title(f"day {j}, {tau * 252:.0f} days to maturity")
        ax.set_xlabel("stock price S")
        ax.set_xlim(85, 115)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("position (shares)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Without costs, the network learns the Black-Scholes delta")
    fig.tight_layout()
    fig.savefig(FIGURES / "deep_vs_delta.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: CVaR of all strategies --------------------------------
    width = 0.2
    x = np.arange(len(COSTS))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, name in enumerate(STRATEGIES):
        values = [results[c][name]["cvar"] for c in COSTS]
        ax.bar(x + (i - 1.5) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cost {c:.1%}" for c in COSTS])
    ax.set_ylabel(f"CVaR {ALPHA:.0%} of loss")
    ax.set_title("Hedging risk on test paths (N = 30, Black-Scholes premium)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "deep_comparison.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: learned no-trade behaviour (cost 0.5%) -----------------
    # x: how far the previous position is from the Black-Scholes delta
    # y: how much the strategy trades. A no-trade band shows up as a flat
    # region around x = 0 where the strategy does not trade.
    cost = COSTS[-1]
    j = 15
    s_j = test_paths[:3000, j]
    bs_delta_j = call_delta(s_j, K, T - j * dt, R, SIGMA)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["Whalley-Wilmott", "Deep hedge"]):
        d = deltas[cost][name][:3000]
        gap = d[:, j - 1] - bs_delta_j
        trade = d[:, j] - d[:, j - 1]
        ax.scatter(gap, trade, s=3, alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.plot([-0.3, 0.3], [0.3, -0.3], ":", color="grey", label="trade fully to delta")
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel("previous position − Black-Scholes delta")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("trade size at day 15 (shares)")
    fig.suptitle(f"Trading decisions at day {j}, cost {cost:.1%}")
    fig.tight_layout()
    fig.savefig(FIGURES / "deep_no_trade.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved figures to {FIGURES}/")


if __name__ == "__main__":
    main()