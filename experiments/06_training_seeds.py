"""Experiment 06: how much do the deep hedging results depend on the training run?

Experiment 04 trained one network per cost level. A different random start or
different training paths could give a somewhat better or worse network. Here
each deep hedger is trained with several seeds (the seed sets both the initial
weights and the training paths) and every network is evaluated on the same
test paths as in experiment 04. The classical strategies have nothing to
train, so they are evaluated once.

The question: is the deep hedge better than the classical strategies for every
training seed, or only for some?

Seed 0 uses the same settings as experiment 04, so its model is reused from
models/ if it exists. All other models are saved in models/seeds/ and reused
on later runs. Training all models takes a while (about 1-2 minutes each).

Run from the repository root:
    python experiments/06_training_seeds.py

Saves figures/training_seeds.png and prints Markdown tables.
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
from deep_hedging.black_scholes import call_price
from deep_hedging.deep_hedger import DeepHedger, deep_hedge_strategy, train_hedger
from deep_hedging.hedging import hedge_pnl
from deep_hedging.models.gbm import simulate_gbm
from deep_hedging.monte_carlo import call_payoff
from deep_hedging.risk import cvar

# Option, market and training settings (same as experiment 04)
S0, K, T, R, SIGMA = 100, 100, 30 / 252, 0.0, 0.2
N_STEPS = 30
COSTS = [0.0, 0.001, 0.005]
ALPHA = 0.95
N_ITERS = 1500
BATCH_SIZE = 10_000

SEEDS = [0, 1, 2, 3, 4]            # one trained network per seed and cost level

# Evaluation (same test paths as experiments 03 and 04)
N_TEST, TEST_SEED = 50_000, 2
N_WW_TRAIN, WW_TRAIN_SEED = 20_000, 1
RISK_AVERSIONS = [0.3, 1, 3, 10, 30, 100]

CLASSICAL = ["Delta", "Leland", "Whalley-Wilmott"]
FIGURES = Path("figures")
MODELS = Path("models")


def get_model(cost, seed):
    """Load the deep hedger for this cost level and seed, or train and save it."""
    if seed == 0:
        model_file = MODELS / f"deep_hedger_cost{cost}.pt"          # from experiment 04
    else:
        model_file = MODELS / "seeds" / f"deep_hedger_cost{cost}_seed{seed}.pt"
    if model_file.exists():
        model = DeepHedger(K, T)
        model.load_state_dict(torch.load(model_file))
        return model

    print(f"Training deep hedger: cost {cost:.1%}, seed {seed} ...")
    model, _ = train_hedger(K, T, SIGMA, cost, N_STEPS, s0=S0, r=R, n_iters=N_ITERS,
                            batch_size=BATCH_SIZE, seed=seed, print_every=0)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_file)
    return model


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
    paths = simulate_gbm(S0, R, SIGMA, T, N_STEPS, N_TEST, seed=TEST_SEED)
    payoff = call_payoff(paths[:, -1], K)

    classical = {}   # classical[cost][strategy] = CVaR
    deep = {}        # deep[cost] = array of CVaRs, one per seed
    for cost in COSTS:
        ra = best_risk_aversion(cost, p0)
        strategies = {
            "Delta": bs_delta_strategy(paths, K, T, R, SIGMA),
            "Leland": leland_strategy(paths, K, T, R, SIGMA, cost),
            "Whalley-Wilmott": whalley_wilmott_strategy(paths, K, T, R, SIGMA, cost, ra),
        }
        classical[cost] = {name: cvar(hedge_pnl(paths, d, payoff, p0, cost), ALPHA)
                           for name, d in strategies.items()}
        deep[cost] = np.array([
            cvar(hedge_pnl(paths, deep_hedge_strategy(get_model(cost, seed), paths),
                           payoff, p0, cost), ALPHA)
            for seed in SEEDS
        ])

    # --- Tables ---------------------------------------------------------
    print(f"\nCVaR {ALPHA:.0%} of the deep hedge for each training seed")
    print("| Cost | " + " | ".join(f"Seed {s}" for s in SEEDS) + " | Mean | Std |")
    print("|---|" + "---|" * (len(SEEDS) + 2))
    for cost in COSTS:
        values = " | ".join(f"{v:.3f}" for v in deep[cost])
        print(f"| {cost:.1%} | {values} | {deep[cost].mean():.3f} | {deep[cost].std(ddof=1):.3f} |")

    print("\nDeep hedge compared with the best classical strategy")
    print("| Cost | Best classical | Its CVaR | Deep hedge (mean ± std) "
          "| Difference per seed | Seeds where deep is better |")
    print("|---|---|---|---|---|---|")
    for cost in COSTS:
        best_name = min(classical[cost], key=classical[cost].get)
        best = classical[cost][best_name]
        diff = deep[cost] - best
        n_better = int(np.sum(diff < 0))
        print(f"| {cost:.1%} | {best_name} | {best:.3f} "
              f"| {deep[cost].mean():.3f} ± {deep[cost].std(ddof=1):.3f} "
              f"| {', '.join(f'{d:+.3f}' for d in diff)} | {n_better} of {len(SEEDS)} |")

    # --- Figure: CVaR of each trained network vs the classical strategies ---
    FIGURES.mkdir(exist_ok=True)
    colors = {"Delta": "tab:blue", "Leland": "tab:orange", "Whalley-Wilmott": "tab:green"}
    fig, axes = plt.subplots(1, len(COSTS), figsize=(14, 4.5))
    for ax, cost in zip(axes, COSTS):
        for name in CLASSICAL:
            ax.axhline(classical[cost][name], color=colors[name], linewidth=1.5, label=name)
        ax.scatter(np.arange(len(SEEDS)), deep[cost], color="tab:red", zorder=3,
                   label="Deep hedge (one per seed)")
        ax.axhline(deep[cost].mean(), color="tab:red", linestyle="--", linewidth=1,
                   label="Deep hedge, mean")
        ax.set_xticks(np.arange(len(SEEDS)))
        ax.set_xticklabels([str(s) for s in SEEDS])
        ax.set_xlabel("training seed")
        title = f"cost {cost:.1%}"
        if cost == 0.0:
            title += " (all classical strategies equal)"
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        # Zoom in on the relevant range so that small differences are visible
        values = list(classical[cost].values()) + list(deep[cost])
        spread = max(values) - min(values)
        ax.set_ylim(min(values) - 0.15 * spread, max(values) + 0.15 * spread)
    axes[0].set_ylabel(f"CVaR {ALPHA:.0%} of loss")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.suptitle(f"Deep hedging with {len(SEEDS)} training seeds vs classical strategies "
                 "(N = 30, same test paths)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))   # leave room for the legend below the plots
    fig.savefig(FIGURES / "training_seeds.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved {FIGURES / 'training_seeds.png'}")


if __name__ == "__main__":
    main()
