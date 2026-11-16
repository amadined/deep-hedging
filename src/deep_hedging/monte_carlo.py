"""
Monte Carlo pricing

Estimates an option's price by averaging over simulated paths.
"""
import numpy as np


def call_payoff(s_T, k):
    """Payoff of a European call at maturity: max(S_T − K, 0)"""
    return np.maximum(s_T - k,0)

def put_payoff(s_T, k):
    """Payoff of a European put at maturity: max(K - S_T, 0)"""
    return np.maximum(k - s_T,0)

def mc_price(payoffs, r, T):
    """
    Input:
        payoffs     array of option payoffs, one per path
        r           risk-free rate
        T           maturity in years
    Output:
        tuple (price, standard_error)
        price ± 1.96 * standard_error gives a 95% confidence interval
    """

    payoffs = np.asarray(payoffs)
    disc_payoffs = payoffs * np.exp(-r * T)
    price = np.mean(disc_payoffs)
    standard_error = np.std(disc_payoffs, ddof=1) / np.sqrt(disc_payoffs.size)
    return price, standard_error
