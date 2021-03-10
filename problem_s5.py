"""Script for solving statistics assignment problem 5"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from tools import find_fc_int
from mathfunctions import pois_dist

"""
Problem 5
"""

"""
b
Just reuse resulting interval from problem 4b and subtract 3.2 from result
"""

"""
c
Just reuse code from problem 4a but with alpha = 0.32 and subtract 3.2 from result
"""

"""
a
Fieldman-Cousins
"""


def find_fc_int(func, params, nu_bg: float, max_n: int, alpha: float = 0.32):
    """Construct interval according to Fieldman-Cousins"""

    n = np.linspace(0., max_n, max_n, dtype=int)
    p_n_nu = func(n, **params)
    nu_hat = copy(n)
    nu_hat[nu_hat < nu_bg] = nu_bg
    p_n_nu_hat = func(n, nu_hat)

    r = p_n_nu/(p_n_nu_hat + 0.00000001)
    ranks = np.argsort(r)[::-1]

    if False:
        print(f"n\tp_n_nu\tnu_hat\tp_n_nu_hat\tr")
        for i in range(n.shape[0]):
            print(f"{n[i]}\t{p_n_nu[i]:0.4f}\t{nu_hat[i]}\t{p_n_nu_hat[i]:0.4f}\t{r[i]:0.4f}")

    sum = 0
    interval = []
    for i in ranks:
        sum += p_n_nu[i]
        interval.append(i)
        if sum > 1-alpha:
            print(sum)
            break
    interval = np.asarray(interval)

    return np.asarray([interval.min(), interval.max()])


def calc_fieldman_cousins(alpha: float, n_nu_points: int = 100):
    """"""

    max_val = 11
    plot_lh = max_val/(2 * n_nu_points)
    nu_bg = 3.2

    for nu in np.linspace(0., max_val, n_nu_points):
        print(f"nu = {nu}")
        fc_int = np.asarray(
            find_fc_int(
                pois_dist,
                {"nu": nu + nu_bg},
                alpha=alpha,
                nu_bg=nu_bg,
                max_n=int((nu + nu_bg)*2.)
            )
        )

        print(nu, round(fc_int[0]), round(fc_int[1]))

        if round(fc_int[0]) == round(fc_int[1]):
            plt.vlines(round(fc_int[0]), nu - plot_lh, nu + plot_lh, colors="m")
        else:
            plt.vlines(round(fc_int[0]), nu - plot_lh, nu + plot_lh, colors="r")
            plt.vlines(
                round(fc_int[1]), nu - plot_lh, nu + plot_lh, colors="b"
            )

    n_plotmax = max_val

    n = np.linspace(0, n_plotmax, dtype=int)
    plt.plot(n, n - nu_bg, "x", color="k")

    plt.xlim([0, n_plotmax])
    plt.ylim([0, n_plotmax])
    plt.xlabel("Events n [ ]")
    plt.ylabel("nu [ ]")
    plt.title(f"{1-alpha:0.2f}% Conf Interval, lambda={nu_bg}")
    plt.show()

    # plt.savefig(f"output/problem_s5.png", dpi=300)


alpha = 0.32
calc_fieldman_cousins(alpha, n_nu_points=1000)

"""
FC-Interval for n = 9: [3.13, 8.65]
"""