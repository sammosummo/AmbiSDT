"""simulate_yn_regular.py"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def decide(psis, k):
        """The decision rule applied to an
        array of observations."""
        for psi in psis:
            if psi < k:
                yield 0
            else:
                yield 1


def sim_regular_yn(d, c, n0, n1=None):
    """Simulate data under the modified YN model.

    Parameters
    ----------
    d : float
        Measure of sensitivity.
    c : float
        Measure of bias.
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
    k = d/2. + c
    psi_0 = norm.rvs(0, 1, n0)
    rsp_0 = np.array([x for x in decide(psi_0, k)])
    r, f = [sum(rsp_0 == i) for i in xrange(2)]
    psi_1 = norm.rvs(d, 1, n1)
    rsp_1 = np.array([x for x in decide(psi_1, k)])
    m, h = [sum(rsp_1 == i) for i in xrange(2)]
    return f, h, m, r


def est_regular_yn(f, h, m, r):
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
    c : float
        Measure of bias.
    n0 : int
        Number of trials with stimuli from the first
        class.
    n1: int
        Number of trials with stimuli from the second
        class.
    """
    n0, n1 = float(f + r), float(h + m)
    if f == 0:
        f += 0.5
    if f == (f + r):
        f -= 0.5
    if h == 0:
        h += 0.5
    if h == (h + m):
        h -= 0.5
    fhat = f / n0
    hhat = h / n1
    d = norm.ppf(hhat) - norm.ppf(fhat)
    c = -0.5 * (norm.ppf(hhat) + norm.ppf(fhat))
    return d, c, f + r, h + m


def test0():
    """Plots the estimates of d as a function of true
    d from a series of simulated data sets. These
    estimates are very close to the corresponding
    true values, indicating that the model is valid."""
    c = 0.1
    true_d_vals = np.linspace(-5, 5, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_d_vals = []
        for d in true_d_vals:
            dhat, chat, _, _ = est_regular_yn(
                *sim_regular_yn(d, c, n)
            )
            est_d_vals.append(dhat)
        plt.plot(true_d_vals, est_d_vals, 'o')
        plt.grid()
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.plot(true_d_vals, true_d_vals, 'k')
        plt.title('%i trials' % (n * 2))
        plt.xlabel('$d$')
        plt.ylabel('$\hat{d}$')
    plt.show()


def test1():
    """Plots the estimates of lambda as a function of true
    lambda from a series of simulated data sets. These
    estimates are very close to the corresponding
    true values, indicating that the model is valid."""
    d = 1.5
    true_c_vals = np.linspace(-2, 2, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_c_vals = []
        for c in true_c_vals:
            dhat, chat, _, _ = est_regular_yn(
                *sim_regular_yn(d, c, n)
            )
            est_c_vals.append(chat)
        plt.plot(true_c_vals, est_c_vals, 'o')
        plt.grid()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.plot(true_c_vals, true_c_vals, 'k')
        plt.title('%i trials' % (n * 2))
        plt.xlabel(r'$lamda$')
        plt.ylabel(r'$\hat{lamda}$')
    plt.show()


if __name__ == '__main__':
    test0()
    test1()
