"""Experiment 0: validate Monte Carlo pricing against Black-Scholes.

Prices several European options by Monte Carlo and compares them with the
closed-form Black-Scholes prices. Prints a Markdown table for the README.
"""

from deep_hedging.black_scholes import call_price, put_price
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.monte_carlo import call_payoff, mc_price, put_payoff

N_PATHS = 200_000

# (option type, s0, k, T, r, sigma, label for T in the table)
CASES = [
    ("Call", 100, 100, 1.0, 0.05, 0.2, "1"),
    ("Put", 100, 100, 1.0, 0.05, 0.2, "1"),
    ("Call", 42, 40, 0.5, 0.10, 0.2, "0.5"),  # Hull (2009) textbook example
    ("Put", 42, 40, 0.5, 0.10, 0.2, "0.5"),
    ("Call", 100, 105, 30 / 252, 0.0, 0.2, "30/252"),
]


def main():
    print("| Option | S₀ | K | T | r | σ | Black–Scholes | Monte Carlo | Std. error | z |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    for seed, (kind, s0, k, T, r, sigma, T_label) in enumerate(CASES):
        # Risk-neutral simulation (mu = r). One step is enough for a
        # European option because the GBM simulator is exact.
        s_T = simulate_gbm(s0, r, sigma, T, n_steps=1, n_paths=N_PATHS, seed=seed)[:, -1]

        if kind == "Call":
            payoffs = call_payoff(s_T, k)
            exact = call_price(s0, k, T, r, sigma)
        else:
            payoffs = put_payoff(s_T, k)
            exact = put_price(s0, k, T, r, sigma)

        mc, se = mc_price(payoffs, r, T)
        z = (mc - exact) / se  # difference in units of the standard error

        print(
            f"| {kind} | {s0} | {k} | {T_label} | {r:.0%} | {sigma:.0%} "
            f"| {exact:.4f} | {mc:.4f} | {se:.4f} | {z:+.2f} |"
        )


if __name__ == "__main__":
    main()