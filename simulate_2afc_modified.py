"""simulate_2afc_regular.py"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def decide(psi_pairs, alpha, beta):
        """The decision rule applied to an
        array of observations."""
        for psi_0, psi_1 in psi_pairs:
            if psi_0 + alpha > psi_1:
                yield 0
            elif psi_0 + alpha <= psi_1 and psi_0 + beta > psi_1:
                yield 1
            else:
                yield 2


def sim_modified_2afc(d, zeta, tau, n0, n1=None):
    """Simulate data under the modified YN model.

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
    psi_pairs_0 = zip(norm.rvs(0, 1, n0), norm.rvs(d, 1, n0))
    rsp_0 = np.array([x for x in decide(psi_pairs_0, alpha, beta)])
    m, j, h = [sum(rsp_0 == i) for i in xrange(3)]
    psi_pairs_1 = zip(norm.rvs(d, 1, n1), norm.rvs(0, 1, n1))
    rsp_1 = np.array([x for x in decide(psi_pairs_1, alpha, beta)])
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

    Returns
    -------
    d : float
        Measure of sensitivity.
    zeta : float
        Measure of bias.
    tau:
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
        tau = (norm.ppf(fhat + ghat) - norm.ppf(fhat))*np.sqrt(2)
    return d, zeta, tau, f + r, h + m


def test0():
    """Plots the estimates of d as a function of true
    d from a series of simulated data sets. These
    estimates are very close to the corresponding
    true values, indicating that the model is valid."""
    zeta = 0.1
    tau = 0.4
    true_d_vals = np.linspace(-5/np.sqrt(2), 5/np.sqrt(2), 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_d_vals = []
        for d in true_d_vals:
            dhat, zhat, that, _, _ = est_modified_2afc(
                *sim_modified_2afc(d, zeta, tau, n)
            )
            est_d_vals.append(dhat)
        plt.plot(true_d_vals, est_d_vals, 'o')
        plt.grid()
        plt.xlim(-5/np.sqrt(2), 5/np.sqrt(2))
        plt.ylim(-5/np.sqrt(2), 5/np.sqrt(2))
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
    d = 0
    tau = 1
    true_vals = np.linspace(-2, 2, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_vals = []
        for zeta in true_vals:
            dhat, zhat, that, _, _ = est_modified_2afc(
                *sim_modified_2afc(d, zeta, tau, n)
            )
            est_vals.append(zhat)
        plt.plot(true_vals, est_vals, 'o')
        plt.grid()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.plot(true_vals, true_vals, 'k')
        plt.title('%i trials' % (n * 2))
        plt.xlabel(r'$zeta$')
        plt.ylabel(r'$\hat{zeta}$')
    plt.show()

def test2():
    """Plots the estimates of u as a function of true
    u from a series of simulated data sets. These
    estimates are very close to the corresponding
    true values, indicating that the model is valid."""
    d = 1.5
    zeta = 0.2
    true_t_vals = np.linspace(0, 3, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_t_vals = []
        for tau in true_t_vals:
            dhat, zhat, that, _, _ = est_modified_2afc(
                *sim_modified_2afc(d, zeta, tau, n)
            )
            est_t_vals.append(that)
        plt.plot(true_t_vals, est_t_vals, 'o')
        plt.grid()
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.plot(true_t_vals, true_t_vals, 'k')
        plt.title('%i trials' % (n * 2))
        plt.xlabel(r'$tau$')
        plt.ylabel(r'$\hat{tau}$')
    plt.show()


if __name__ == '__main__':
    test0()
    test1()
    test2()
