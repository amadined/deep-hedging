"""Experiment 02: Black-Scholes delta hedging with discrete rebalancing and
transaction costs.

A short at-the-money call is hedged with the Black-Scholes delta, rebalanced
N times until maturity. Shows:
  1. Without costs, the hedging error shrinks like 1/sqrt(N).
  2. With costs, more rebalancing is not always better: the risk (CVaR)
     first falls and then rises again, so there is an optimal frequency.
  3. How costs shift and widen the P&L distribution.

Run from the repository root:
    python experiments/02_delta_hedging.py

Saves figures/delta_hedging_error.png, figures/delta_hedging_costs.png
and figures/delta_hedging_pnl.png, and prints a Markdown table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deep_hedging.baselines import bs_delta_strategy
from deep_hedging.black_scholes import call_price
from deep_hedging.hedging import hedge_pnl
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.monte_carlo import call_payoff
from deep_hedging.risk import summarize

# Option and market: one-month at-the-money call, zero interest rate,
# and the stock simulated with the risk-neutral drift mu = r = 0
S0, K, T, R, SIGMA = 100, 100, 30 / 252, 0.0, 0.2
N_PATHS = 50_000
SEED = 0

N_STEPS_LIST = [5, 10, 20, 30, 50, 100, 200, 400]   # rebalancing frequencies
COSTS = [0.0, 0.001, 0.005]                         # proportional cost rates
ALPHA = 0.95                                        # CVaR level

FIGURES = Path("figures")


def run_delta_hedge(n_steps, costs, p0):
    """Hedge on one set of simulated paths and return {cost: P&L array}.

    The same paths are used for every cost level, so differences between
    cost levels come from the costs only, not from simulation noise.
    """
    paths = simulate_gbm(S0, R, SIGMA, T, n_steps, N_PATHS, seed=SEED)
    deltas = bs_delta_strategy(paths, K, T, R, SIGMA)
    payoff = call_payoff(paths[:, -1], K)
    return {c: hedge_pnl(paths, deltas, payoff, p0, c) for c in costs}


def main():
    p0 = call_price(S0, K, T, R, SIGMA)
    print(f"Black-Scholes premium p0 = {p0:.4f}\n")

    # --- Run all combinations of rebalancing frequency and cost ---------
    # results[c][i] holds the summary for COSTS c and N_STEPS_LIST[i]
    results = {c: [] for c in COSTS}
    pnl_for_histogram = {}
    for n_steps in N_STEPS_LIST:
        pnls = run_delta_hedge(n_steps, COSTS, p0)
        for c in COSTS:
            results[c].append(summarize(pnls[c], ALPHA))
        if n_steps == 30:  # daily rebalancing, kept for the histogram
            pnl_for_histogram = pnls

    # --- Markdown table for the README ---------------------------------
    print("| Rebalancing steps N | Cost | Mean P&L | Std P&L | CVaR 95% |")
    print("|---|---|---|---|---|")
    for c in COSTS:
        for n_steps, s in zip(N_STEPS_LIST, results[c]):
            print(f"| {n_steps} | {c:.1%} | {s['mean']:.4f} | {s['std']:.4f} | {s['cvar']:.4f} |")

    FIGURES.mkdir(exist_ok=True)
    n = np.array(N_STEPS_LIST)

    # --- Figure 1: hedging error without costs ---------------------------
    std_no_cost = np.array([s["std"] for s in results[0.0]])
    reference = std_no_cost[0] * np.sqrt(n[0] / n)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(n, std_no_cost, "o-", label="std of hedging P&L")
    ax.loglog(n, reference, ":", color="grey", label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel("number of rebalancing steps N")
    ax.set_ylabel("standard deviation of P&L")
    ax.set_title("Delta hedging error without transaction costs")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "delta_hedging_error.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: risk vs rebalancing frequency for each cost level -----
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in COSTS:
        cvars = [s["cvar"] for s in results[c]]
        ax.semilogx(n, cvars, "o-", label=f"cost {c:.1%}")
    ax.set_xlabel("number of rebalancing steps N")
    ax.set_ylabel(f"CVaR {ALPHA:.0%} of loss")
    ax.set_title("Delta hedging: risk vs rebalancing frequency")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "delta_hedging_costs.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: P&L distributions at daily rebalancing ----------------
    bins = np.linspace(-4, 2, 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in COSTS:
        ax.hist(pnl_for_histogram[c], bins=bins, alpha=0.5, density=True,
                histtype="stepfilled", label=f"delta hedge, cost {c:.1%}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("P&L of the hedged short call")
    ax.set_ylabel("density")
    ax.set_title("P&L distribution, daily rebalancing (N = 30)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "delta_hedging_pnl.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved figures to {FIGURES}/")


if __name__ == "__main__":
    main()