# deep-hedging

[![tests](https://github.com/amadined/deep-hedging/actions/workflows/tests.yml/badge.svg)](https://github.com/amadined/deep-hedging/actions/workflows/tests.yml)

Deep hedging of European options under transaction costs and stochastic volatility (Heston), compared against classical benchmarks. Python / PyTorch.

## Question

A bank sells an option and hedges it by trading the underlying at discrete times. With transaction costs or stochastic volatility, the classical Black–Scholes delta hedge is no longer optimal, and in general there is no closed-form answer.

Following Buehler, Gonon, Teichmann and Wood (2019), this project trains a neural network to choose the hedge position directly, by minimising a risk measure (CVaR) of the hedger's terminal profit and loss on simulated price paths. It then asks:

- Does the network recover the Black–Scholes delta when there are no costs?
- With transaction costs, does it learn to trade less, and how does it compare with the Leland and Whalley–Wilmott strategies?
- Under Heston stochastic volatility, how much better is it than the misspecified Black–Scholes delta?

## Roadmap

- [ ] Black–Scholes pricer and delta
- [ ] Geometric Brownian motion simulator
- [ ] Monte Carlo pricing with standard errors, validated against Black–Scholes
- [ ] Heston simulator and semi-analytic Heston pricer
- [ ] Classical hedging baselines: delta, Leland, Whalley–Wilmott
- [ ] Deep hedging network with CVaR loss (PyTorch)
- [ ] Experiments with transaction costs
- [ ] Experiments under Heston

## Results so far

## Repository structure

```
src/deep_hedging/
├── black_scholes.py     closed-form prices and delta
├── monte_carlo.py       payoffs and Monte Carlo pricing
└── models/
    └── gbm.py           geometric Brownian motion simulator
tests/                   unit tests against known theory
experiments/             one script per experiment, writes to figures/
figures/                 generated plots
```

## Setup

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python experiments/01_mc_convergence.py
```

## References

- H. Buehler, L. Gonon, J. Teichmann, B. Wood. *Deep Hedging.* Quantitative Finance 19(8), 2019.
- F. Black, M. Scholes. *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 1973.
- S. Heston. *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies, 1993.
- H. Leland. *Option Pricing and Replication with Transactions Costs.* Journal of Finance, 1985.
- A. E. Whalley, P. Wilmott. *An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs.* Mathematical Finance, 1997.

## License

MIT
