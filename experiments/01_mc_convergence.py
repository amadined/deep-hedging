"""Experiment 01: <what it shows>.

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
N_SEEDS = 20
PRICE_EXACT = call_price(S0, K, T, R, SIGMA)


def main():
    # 1. compute the results (loops over parameters, simulations)
    # 2. print a small table to the terminal
    # 3. make the figure(s) and save them
    
    path_counts = 2 ** np.arange(8, 19)
    print(PRICE_EXACT)
    print(path_counts)

    mean_errors = np.empty((11,))
    mean_ses = np.empty((11,))

    for i, n in enumerate(path_counts):
        errors = np.empty((N_SEEDS,))
        ses = np.empty((N_SEEDS,))
        for seed in range(N_SEEDS):
            paths = simulate_gbm(S0, R, SIGMA, T, n_steps=1, n_paths=n, seed=seed)
            price_mc, se = mc_price(call_payoff(paths[:, -1], K), R, T)
            errors[seed] = np.abs(price_mc - PRICE_EXACT)
            ses[seed] = se
        
        mean_errors[i] = np.mean(errors)
        mean_ses[i] = np.mean(ses)

        print(f"n = {n:>7d}   mean |error| = {mean_errors[i]:.5f}   mean SE = {mean_ses[i]:.5f}")


    reference = mean_ses[0] * np.sqrt(path_counts[0] / path_counts)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(path_counts, mean_errors, "o-", label=f"mean |MC price - BS price|")
    ax.loglog(path_counts, mean_ses, "s--", label=f"mean standard error" )
    ax.loglog(path_counts, reference, ":", color="grey", label=r"$\propto 1/\sqrt{n}$")
    ax.legend()
    ax.set_xlabel(f"number of simulated paths n")
    ax.set_ylabel(f"error")
    ax.set_title(f"Monte Carlo convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/mc_convergence.png", dpi=150)


if __name__ == "__main__":
    main()