"""Script for solving statistics assignment problem 5"""

import numpy as np
import matplotlib.pyplot as plt

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

alpha = 0.32
n = 9
nu_star = n


def calc_fieldman_cousins(n: int, alpha: float, n_nu_points: int = 100):
    """"""

    max_val = 20
    plot_lh = max_val/(2 * n_nu_points)
    nu_bg = 3.2

    for nu in np.linspace(0., max_val, n_nu_points):

        fc_int = np.asarray(
            find_fc_int(
                pois_dist,
                {"nu": nu},
                alpha=alpha,
                nu_bg=nu_bg
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
        # plt.plot(round(n), nu, marker="x", markersize=5, color="k")

    plt.xlabel("Events n [ ]")
    plt.ylabel("nu [ ]")
    plt.title(f"{1-alpha:0.2f}% Conf Interval, N={n}")
    plt.show()

    # plt.savefig(f"output/problem_s5_n{n}.png", dpi=300)


calc_fieldman_cousins(n, alpha, n_nu_points=4)