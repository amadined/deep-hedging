"""
Semi-analytic pricing of European options in the Heston model.

The characteristic function uses the numerically stable "little Heston trap"
formulation (Albrecher, Mayer, Schoutens and Tistaert, 2007). Call prices
are computed as C = S0 * P1 - K * exp(-rT) * P2 by numerical integration.
"""

import numpy as np
from scipy.integrate import quad

def heston_char_func(u, s0, v0, kappa, theta, xi, rho, r, T):
    """
    Characteristic function phi(u) = E[exp(i * u * ln S_T)] of the log-price
    at maturity in the Heston model, under the risk-neutral measure.

    Uses the "little Heston trap" formulation (Albrecher et al., 2007),
    which avoids jumps of the complex logarithm for long maturities.

    Input:
        u           argument of the characteristic function (real or complex)
        s0          starting price
        v0          starting variance
        kappa       speed of mean reversion of the variance
        theta       long-run variance
        xi          volatility of the variance
        rho         correlation between price shocks and variance shocks
        r           risk-free interest rate
        T           time to maturity in years
    Output:
        phi(u), a complex number.
    """
        
    x0 = np.log(s0)
    b = kappa - rho * xi * 1j * u 
    d = np.sqrt(b**2 + xi**2 * (1j * u + u**2))
    g = (b - d) / (b + d)

    e = np.exp(-d * T)
    c_u = r * 1j * u * T + (kappa * theta / xi**2) * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    d_u = ((b - d) / xi**2) * (1 - e) / (1 - g * e)
    
    return np.exp(c_u + d_u * v0 + 1j * u * x0)

def heston_call_price(s0, k, T, r, v0, kappa, theta, xi, rho):
    """
    Price of a European call option in the Heston model.

    Computed as C = s0 * P1 - k * exp(-r * T) * P2, where P1 and P2 are
    obtained from the characteristic function by numerical integration.
    P2 is the risk-neutral probability that S_T > k (like N(d2) in
    Black-Scholes), P1 plays the role of N(d1).

    Input:
        s0          starting price
        k           strike
        T           time to maturity in years
        r           risk-free interest rate
        v0          starting variance
        kappa       speed of mean reversion of the variance
        theta       long-run variance
        xi          volatility of the variance (must be > 0)
        rho         correlation between price shocks and variance shocks
    Output:
        call price (float)
    """
    
    def f_1(u):
        phi = heston_char_func(u - 1j, s0, v0, kappa, theta, xi, rho, r, T)
        return np.real(np.exp(-1j * u * np.log(k)) *  phi / (1j * u * s0 * np.exp(r * T)))
    
    def f_2(u):
        phi = heston_char_func(u, s0, v0, kappa, theta, xi, rho, r, T)
        return np.real(np.exp(-1j * u * np.log(k)) *  phi / (1j * u))

    p_1 = 0.5 + (1 / np.pi) * quad(f_1, 0, np.inf, limit=200)[0]
    p_2 = 0.5 + (1 / np.pi) * quad(f_2, 0, np.inf, limit=200)[0]

    C = s0 * p_1 - k * np.exp(-r * T) * p_2
    return C

   
def heston_put_price(s0, k, T, r, v0, kappa, theta, xi, rho):
    """
    Price of a European put option in the Heston model, via put-call
    parity: P = C - s0 + k * exp(-r * T).

    Input:
        same as heston_call_price
    Output:
        put price (float)
    """

    C = heston_call_price(s0, k, T, r, v0, kappa, theta, xi, rho)
    return C - s0 + k * np.exp(-r * T)