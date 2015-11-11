"""simulate_2afc_regular.py"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def decide(psi_pairs, l):
        """The decision rule applied to an
        array of observations."""
        for psi_0, psi_1 in psi_pairs:
            if psi_0 + l > psi_1:
                yield 0
            else:
                yield 1


def sim_regular_2afc(d, l, n0, n1=None):
    """Simulate data under the modified YN model.

    Parameters
    ----------
    d : float
        Measure of sensitivity.
    l : float
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
    psi_pairs_0 = zip(norm.rvs(0, 1, n0), norm.rvs(d, 1, n0))
    rsp_0 = np.array([x for x in decide(psi_pairs_0, l)])
    m, h = [sum(rsp_0 == i) for i in xrange(2)]
    psi_pairs_1 = zip(norm.rvs(d, 1, n1), norm.rvs(0, 1, n1))
    rsp_1 = np.array([x for x in decide(psi_pairs_1, l)])
    r, f = [sum(rsp_1 == i) for i in xrange(2)]
    return f, h, m, r


def est_regular_2afc(f, h, m, r):
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
    n1, n0 = float(f + r), float(h + m)
    if f == 0:
        f += 0.5
    if f == (f + r):
        f -= 0.5
    if h == 0:
        h += 0.5
    if h == (h + m):
        h -= 0.5
    fhat = f / n1
    hhat = h / n0
    d = (norm.ppf(hhat) - norm.ppf(fhat))/np.sqrt(2)
    l = -(norm.ppf(hhat) + norm.ppf(fhat))/np.sqrt(2)
    return d, l, f + r, h + m


def test0():
    """Plots the estimates of d as a function of true
    d from a series of simulated data sets. These
    estimates are very close to the corresponding
    true values, indicating that the model is valid."""
    l = 0.1
    true_d_vals = np.linspace(-5, 5, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_d_vals = []
        for d in true_d_vals:
            dhat, lhat, _, _ = est_regular_2afc(
                *sim_regular_2afc(d, l, n)
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
    """Plots the estimates of l as a function of true
    l from a series of simulated data sets. These
    estimates are very close to the corresponding
    true values, indicating that the model is valid."""
    d = 1.5
    true_l_vals = np.linspace(-2, 2, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_l_vals = []
        for l in true_l_vals:
            dhat, lhat, _, _ = est_regular_2afc(
                *sim_regular_2afc(d, l, n)
            )
            est_l_vals.append(lhat)
        plt.plot(true_l_vals, est_l_vals, 'o')
        plt.grid()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.plot(true_l_vals, true_l_vals, 'k')
        plt.title('%i trials' % (n * 2))
        plt.xlabel(r'$l$')
        plt.ylabel(r'$\hat{l}$')
    plt.show()


if __name__ == '__main__':
    test0()
    test1()
