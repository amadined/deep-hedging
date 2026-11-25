"""Experiment 01: convergence of the Monte Carlo price to the Black-Scholes price.

Prices a one-month at-the-money call by Monte Carlo with an increasing number
of paths n and compares it with the exact Black-Scholes price. Both the actual
error |MC price - BS price| and the reported standard error should fall like
1/sqrt(n). Each experiment is repeated with different seeds and averaged. The
three panels show how the curves smooth out with 10, 20 and 100 seeds.

Run from the repository root:
    python experiments/01_mc_convergence.py

Saves figures/mc_convergence.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deep_hedging.black_scholes import call_price
from deep_hedging.monte_carlo import mc_price, call_payoff
from deep_hedging.models.gbm import simulate_gbm


S0, K, T, R, SIGMA = 100, 100, 30 / 252, 0.0, 0.2
N_SEEDS_LIST = [10, 20, 100]        # one panel per number of seeds
PRICE_EXACT = call_price(S0, K, T, R, SIGMA)
FIGURES = Path("figures")


def main():
    path_counts = 2 ** np.arange(8, 19)   # 256, 512, ..., 262144 paths
    max_seeds = max(N_SEEDS_LIST)
    print(f"Black-Scholes price: {PRICE_EXACT:.4f}")

    # Run every path count with seeds 0, ..., max_seeds - 1 once and store the
    # results. The panel with 10 seeds then averages the first 10 runs, the
    # panel with 20 seeds the first 20, and so on, so nothing is simulated twice.
    errors = np.empty((len(path_counts), max_seeds))
    ses = np.empty((len(path_counts), max_seeds))
    for i, n in enumerate(path_counts):
        for seed in range(max_seeds):
            paths = simulate_gbm(S0, R, SIGMA, T, n_steps=1, n_paths=n, seed=seed)
            price_mc, se = mc_price(call_payoff(paths[:, -1], K), R, T)
            errors[i, seed] = np.abs(price_mc - PRICE_EXACT)
            ses[i, seed] = se

    fig, axes = plt.subplots(1, len(N_SEEDS_LIST), figsize=(14, 4), sharey=True)
    for ax, n_seeds in zip(axes, N_SEEDS_LIST):
        mean_errors = errors[:, :n_seeds].mean(axis=1)
        mean_ses = ses[:, :n_seeds].mean(axis=1)

        print(f"\n{n_seeds} seeds")
        for n, err, se in zip(path_counts, mean_errors, mean_ses):
            print(f"n = {n:>7d}   mean |error| = {err:.5f}   mean SE = {se:.5f}   ratio = {err / se:.2f}")

        reference = mean_ses[0] * np.sqrt(path_counts[0] / path_counts)
        ax.loglog(path_counts, mean_errors, "o-", label="mean |MC price − BS price|")
        ax.loglog(path_counts, mean_ses, "s--", label="mean standard error")
        ax.loglog(path_counts, reference, ":", color="grey", label=r"$\propto 1/\sqrt{n}$")
        ax.set_xlabel("number of simulated paths n")
        ax.set_title(f"averaged over {n_seeds} seeds")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("error")
    axes[0].legend()
    fig.suptitle(f"Monte Carlo convergence, ATM call (Black–Scholes price {PRICE_EXACT:.3f})")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "mc_convergence.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {FIGURES / 'mc_convergence.png'}")


if __name__ == "__main__":
    main()
