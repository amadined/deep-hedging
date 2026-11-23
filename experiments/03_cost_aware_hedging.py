"""Experiment 03: cost-aware hedging strategies compared with delta hedging.

A short one-month at-the-money call is hedged with three strategies:
  - Black-Scholes delta hedging (rebalance to delta every step),
  - Leland (1985): delta hedging with an adjusted volatility,
  - Whalley-Wilmott (1997): no-trade band around the delta.

The risk aversion of Whalley-Wilmott is chosen on a training set of paths
(lowest CVaR) and all strategies are then evaluated on fresh test paths.
All strategies receive the same Black-Scholes premium, so the comparison
is fair, the coverage of the Leland premium is reported separately.

Run from the repository root:
    python experiments/03_cost_aware_hedging.py

Saves figures/cost_aware_cvar.png, figures/cost_aware_turnover.png and
figures/cost_aware_pnl.png, and prints Markdown tables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deep_hedging.baselines import (
    bs_delta_strategy,
    leland_sigma,
    leland_strategy,
    whalley_wilmott_strategy,
)
from deep_hedging.black_scholes import call_price
from deep_hedging.hedging import hedge_pnl, turnover
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.monte_carlo import call_payoff
from deep_hedging.risk import cvar, summarize

# Option and market (same as experiment 02)
S0, K, T, R, SIGMA = 100, 100, 30 / 252, 0.0, 0.2

N_TRAIN, N_TEST = 20_000, 50_000
TRAIN_SEED, TEST_SEED = 1, 2

N_STEPS_LIST = [10, 30, 100, 400]
COSTS = [0.001, 0.005]
RISK_AVERSIONS = [0.3, 1, 3, 10, 30, 100]   # grid for Whalley-Wilmott
ALPHA = 0.95

STRATEGIES = ["Delta", "Leland", "Whalley-Wilmott"]
FIGURES = Path("figures")


def simulate(n_steps, n_paths, seed):
    """Paths and call payoffs for the experiment."""
    paths = simulate_gbm(S0, R, SIGMA, T, n_steps, n_paths, seed=seed)
    return paths, call_payoff(paths[:, -1], K)


def best_risk_aversion(n_steps, cost, p0):
    """Choose the Whalley-Wilmott risk aversion with the lowest CVaR on training paths."""
    paths, payoff = simulate(n_steps, N_TRAIN, TRAIN_SEED)
    cvars = []
    for ra in RISK_AVERSIONS:
        deltas = whalley_wilmott_strategy(paths, K, T, R, SIGMA, cost, ra)
        cvars.append(cvar(hedge_pnl(paths, deltas, payoff, p0, cost), ALPHA))
    return RISK_AVERSIONS[int(np.argmin(cvars))]


def run(n_steps, cost, p0):
    """Evaluate all strategies on the same test paths.

    Returns {strategy: (pnl, deltas)} and the chosen risk aversion.
    """
    ra = best_risk_aversion(n_steps, cost, p0)
    paths, payoff = simulate(n_steps, N_TEST, TEST_SEED)
    all_deltas = {
        "Delta": bs_delta_strategy(paths, K, T, R, SIGMA),
        "Leland": leland_strategy(paths, K, T, R, SIGMA, cost),
        "Whalley-Wilmott": whalley_wilmott_strategy(paths, K, T, R, SIGMA, cost, ra),
    }
    out = {name: (hedge_pnl(paths, d, payoff, p0, cost), d) for name, d in all_deltas.items()}
    return out, ra


def main():
    p0 = call_price(S0, K, T, R, SIGMA)
    print(f"Black-Scholes premium p0 = {p0:.4f}\n")

    # results[cost][strategy] = list of summaries, one per N in N_STEPS_LIST
    results = {c: {s: [] for s in STRATEGIES} for c in COSTS}
    chosen_ra = {c: [] for c in COSTS}
    histogram_pnl = {}

    for cost in COSTS:
        for n_steps in N_STEPS_LIST:
            out, ra = run(n_steps, cost, p0)
            chosen_ra[cost].append(ra)
            for name, (pnl, deltas) in out.items():
                summary = summarize(pnl, ALPHA)
                summary["turnover"] = turnover(deltas).mean()
                results[cost][name].append(summary)
            if cost == 0.005 and n_steps == 30:
                histogram_pnl = {name: pnl for name, (pnl, _) in out.items()}
            print(f"done: cost {cost:.1%}, N = {n_steps}, risk aversion {ra}")

    # --- Tables ---------------------------------------------------------
    print("\n| Cost | N | Strategy | Mean P&L | Std P&L | CVaR 95% | Turnover |")
    print("|---|---|---|---|---|---|---|")
    for cost in COSTS:
        for i, n_steps in enumerate(N_STEPS_LIST):
            for name in STRATEGIES:
                s = results[cost][name][i]
                label = name
                if name == "Whalley-Wilmott":
                    label += f" (γ = {chosen_ra[cost][i]})"
                print(f"| {cost:.1%} | {n_steps} | {label} | {s['mean']:.3f} "
                      f"| {s['std']:.3f} | {s['cvar']:.3f} | {s['turnover']:.2f} |")

    # How much of the costs does the Leland premium cover?
    print("\n| Cost | N | Leland sigma | Leland premium | Mean P&L with Leland premium |")
    print("|---|---|---|---|---|")
    for cost in COSTS:
        for i, n_steps in enumerate(N_STEPS_LIST):
            sigma_hat = leland_sigma(SIGMA, cost, T / n_steps)
            p_leland = call_price(S0, K, T, R, sigma_hat)
            # Same hedge, only the premium changes: P&L shifts by p_leland - p0
            mean_leland = results[cost]["Leland"][i]["mean"] + (p_leland - p0)
            print(f"| {cost:.1%} | {n_steps} | {sigma_hat:.3f} | {p_leland:.3f} | {mean_leland:.3f} |")

    FIGURES.mkdir(exist_ok=True)
    n = np.array(N_STEPS_LIST)

    # --- Figure 1: CVaR vs rebalancing frequency --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cost in zip(axes, COSTS):
        for name in STRATEGIES:
            ax.semilogx(n, [s["cvar"] for s in results[cost][name]], "o-", label=name)
        ax.set_xlabel("number of rebalancing steps N")
        ax.set_ylabel(f"CVaR {ALPHA:.0%} of loss")
        ax.set_title(f"Transaction cost {cost:.1%}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Risk of hedging strategies vs rebalancing frequency")
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_aware_cvar.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: turnover vs rebalancing frequency ----------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in STRATEGIES:
        ax.semilogx(n, [s["turnover"] for s in results[0.005][name]], "o-", label=name)
    ax.set_xlabel("number of rebalancing steps N")
    ax.set_ylabel("shares traded per path")
    ax.set_title("Trading volume, transaction cost 0.5%")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_aware_turnover.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: P&L distributions -------------------------------------
    bins = np.linspace(-5, 2, 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in STRATEGIES:
        ax.hist(histogram_pnl[name], bins=bins, alpha=0.5, density=True,
                histtype="stepfilled", label=name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("P&L of the hedged short call (Black-Scholes premium)")
    ax.set_ylabel("density")
    ax.set_title("P&L distribution, N = 30, transaction cost 0.5%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_aware_pnl.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved figures to {FIGURES}/")


if __name__ == "__main__":
    main()