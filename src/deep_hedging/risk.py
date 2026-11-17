"""
Risk measures for hedging P&L.

All measures are computed on losses L = -P&L, so larger values mean more risk.
VaR_alpha is the alpha-quantile of L, CVaR_alpha (expected shortfall) is the
mean of L over the worst (1 - alpha) fraction of paths.
"""

import numpy as np

def var(pnl, alpha=0.95):
    pnl = np.asarray(pnl)
    losses = -pnl
    return np.quantile(losses, alpha)

def cvar(pnl, alpha=0.95):
    pnl = np.asarray(pnl)
    losses = -pnl
    var_value = var(pnl, alpha)
    return np.mean(losses[losses >= var_value])


def summarize(pnl, alpha=0.95):
    """
    returns the mean, standard deviation, VaR and CVaR in one dictionary
    """
    pnl = np.asarray(pnl)
    std = np.std(pnl, ddof=1)
    var_value = var(pnl, alpha)
    cvar_value = cvar(pnl, alpha)
    return dict(mean=np.mean(pnl), std=std, var=var_value, cvar=cvar_value, alpha=alpha)