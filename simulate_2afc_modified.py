"""simulate_2afc_modified.py

Verification of the modified 2afc model.

"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def decide(psi_0s, psi_1s, alpha, beta):
        """Returns the decisions 0, 1, or 2 for
        arrays of observations.

        """
        for psi_0, psi_1 in zip(psi_0s, psi_1s):
            if psi_0 - alpha > psi_1:
                yield 0
            elif psi_0 - alpha <= psi_1 and psi_0 + beta > psi_1:
                print '!'
                print psi_0, psi_1, alpha, beta
                print psi_0 - alpha, psi_1
                print psi_0 + beta, psi_1
                yield 1
            else:
                yield 2


def sim_modified_2afc(d, zeta, tau, n0, n1=None):
    """Simulate data under the modified 2afc model.

    Parameters
    ----------
    d : float
        Measure of sensitivity.
    zeta : float
        Measure of bias.
    tau : float
        Measure of uncertainty/ambivalence.
    n0 : int
        Number of trials with stimuli from the first
        class.
    n1: int, optional
        Number of trials with stimuli from the second
        class; defaults to n1.

    Returns
    -------
    f : int
        Count of observed false alarms.
    h : int
        Count of observed hits.
    g : int
        Count of observed type 1 ambivalent
        responses.
    j : int
        Count of observed type 2 ambivalent
        responses.
    m : int
        Count of observed misses.
    r : int
        Count of observed correct rejections.

    """
    if n1 is None:
        n1 = n0
    alpha = zeta - tau/2.
    beta = zeta + tau/2.
    psi_0_a = norm.rvs(0, 1, n0)
    psi_1_a = norm.rvs(d, 1, n0)
    rsp_0 = np.array([x for x in decide(psi_0_a, psi_1_a, alpha, beta)])
    m, j, h = [sum(rsp_0 == i) for i in xrange(3)]
    psi_0_b = norm.rvs(d, 1, n0)
    psi_1_b = norm.rvs(0, 1, n0)
    rsp_1 = np.array([x for x in decide(psi_0_b, psi_1_b, alpha, beta)])
    r, g, f = [sum(rsp_1 == i) for i in xrange(3)]
    return f, h, g, j, m, r


def est_modified_2afc(f, h, g, j, m, r):
    """Calculate maximum-likelihood estimates of sens-
    itivity and bias.

    Parameters
    ----------
    f : int
        Count of observed false alarms.
    h : int
        Count of observed hits.
    m : int
        Count of observed misses.
    r : int
        Count of observed correct rejections.

    Returns
    -------
    d : float
        Measure of sensitivity.
    zeta : float
        Measure of bias.
    tau : float
        Measure of uncertainty/ambivalence.
    n0 : int
        Number of trials with stimuli from the first
        class.
    n1: int
        Number of trials with stimuli from the second
        class.

    """
    n0, n1 = float(f + r + g), float(h + m + j)
    if f == 0:
        f += 0.5
    if f == (f + r + g):
        f -= 0.5
    if h == 0:
        h += 0.5
    if h == (h + m + j):
        h -= 0.5
    fhat = f / n0
    hhat = h / n1
    ghat = g / n0
    d = (norm.ppf(hhat) - norm.ppf(fhat))/np.sqrt(2)
    zeta = -(norm.ppf(hhat) + norm.ppf(fhat + ghat))/np.sqrt(2)
    if g == 0:
        tau = 0
    else:
        tau = norm.ppf(fhat + ghat) - norm.ppf(fhat)
    if np.isnan(zeta) or zeta == -np.inf:
        print f, h, g, j, m, r
    return d, zeta, tau, f + r + g, h + m + j


def test():
    """Plots the estimates of d as a function of true
    d from a series of simulated data sets.

    """
    zeta = 1
    tau = 0
    true_d_vals = np.linspace(-5, 5, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_d_vals = []
        for d in true_d_vals:
            dhat, zhat, that, _, _ = est_modified_2afc(
                *sim_modified_2afc(d, zeta, tau, n)
            )
            est_d_vals.append(zhat)
        plt.plot(true_d_vals, est_d_vals, 'o')
        plt.grid()
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.plot(true_d_vals, true_d_vals, 'k')
        plt.title('%i trials' % (n * 2))
        plt.xlabel('$d$')
        plt.ylabel('$\hat{d}$')
    plt.show()


if __name__ == '__main__':
    test()
