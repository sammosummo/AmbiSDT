"""simulate_yn_modified.py"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def decide(psis, a, b):
        """The decision rule applied to an
        array of observations."""
        for psi in psis:
            if psi < a:
                yield 0
            elif a <= psi < b:
                yield 1
            else:
                yield 2


def sim_modified_yn(d, lam, u, n0, n1=None):
    """Simulate data under the modified YN model.

    Parameters
    ----------
    d : float
        Measure of sensitivity.
    lam : float
        Measure of bias.
    u : float
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
    a = d/2. + lam - u / 2.
    b = d/2. + lam + u / 2.
    psi_0 = norm.rvs(0, 1, n0)
    rsp_0 = np.array([x for x in decide(psi_0, a, b)])
    r, g, f = [sum(rsp_0 == i) for i in xrange(3)]
    psi_1 = norm.rvs(d, 1, n1)
    rsp_1 = np.array([x for x in decide(psi_1, a, b)])
    m, j, h = [sum(rsp_1 == i) for i in xrange(3)]
    return f, h, g, j, m, r


def est_modified_yn(f, h, g, j, m, r):
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
    lam : float
        Measure of bias.
    u : float
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
    d = norm.ppf(hhat) - norm.ppf(fhat)
    lam = -0.5 * (norm.ppf(hhat) + norm.ppf(fhat + ghat))
    if g == 0:
        u = 0
    else:
        u = norm.ppf(fhat + ghat) - norm.ppf(fhat)
    return d, lam, u, f + r + g, h + m + j


def test0():
    """Plots the estimates of d as a function of true
    d from a series of simulated data sets. These
    estimates are very close to the corresponding
    true values, indicating that the model is valid."""
    c = 0.1
    u = 0.4
    true_d_vals = np.linspace(-5, 5, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_d_vals = []
        for d in true_d_vals:
            dhat, chat, uhat, _, _ = est_modified_yn(
                *sim_modified_yn(d, c, u, n)
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
    u = 0.6
    true_lam_vals = np.linspace(-2, 2, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_lam_vals = []
        for lam in true_lam_vals:
            dhat, chat, uhat, _, _ = est_modified_yn(
                *sim_modified_yn(d, lam, u, n)
            )
            est_lam_vals.append(chat)
        plt.plot(true_lam_vals, est_lam_vals, 'o')
        plt.grid()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.plot(true_lam_vals, true_lam_vals, 'k')
        plt.title('%i trials' % (n * 2))
        plt.xlabel(r'$lamda$')
        plt.ylabel(r'$\hat{lamda}$')
    plt.show()


def test2():
    """Plots the estimates of u as a function of true
    u from a series of simulated data sets. These
    estimates are very close to the corresponding
    true values, indicating that the model is valid."""
    d = 1.5
    lam = 0.2
    true_u_vals = np.linspace(0, 3, 51)
    for i, n in enumerate((25, 100, 250, 500), 1):
        plt.subplot(2, 2, i)
        est_u_vals = []
        for u in true_u_vals:
            dhat, chat, uhat, _, _ = est_modified_yn(
                *sim_modified_yn(d, lam, u, n)
            )
            est_u_vals.append(uhat)
        plt.plot(true_u_vals, est_u_vals, 'o')
        plt.grid()
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.plot(true_u_vals, true_u_vals, 'k')
        plt.title('%i trials' % (n * 2))
        plt.xlabel(r'$u$')
        plt.ylabel(r'$\hat{u}$')
    plt.show()


if __name__ == '__main__':
    test0()
    test1()
    test2()
