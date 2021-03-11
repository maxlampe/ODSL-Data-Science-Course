"""Script for solving statistics assignment problem 4"""

import numpy as np
import matplotlib.pyplot as plt

from tools import find_smallest_interval
from mathfunctions import pois_bay, pois_dist

"""
Problem 4
"""

"""
a
"""
if True:
    alpha = 0.05
    n = 9
    nu_star = n
    plot_lh = 0.1

    small_interval = np.asarray(
        find_smallest_interval(
            pois_bay,
            {"n": n},
            alpha=alpha,
            mode=nu_star,
            step_size=max(nu_star * 0.2, 0.0001),
            int_range_limit=[0.0, 20.0],
        )
    )
    print(small_interval)

    plt.vlines(small_interval[0], n - plot_lh, n + plot_lh, colors="r")
    plt.vlines(small_interval[1], n - plot_lh, n + plot_lh, colors="b")
    plt.plot(n, nu_star, "x", color="k", label="nu*")

    plt.xlabel("nu [ ]")
    plt.ylabel("Events n [ ]")
    plt.title(f"Nu estimates with {1-alpha}% uncertainty")
    plt.legend()
    plt.show()


"""
b
"""

if True:

    alpha = 0.32
    n = 9
    nu_star = n

    def calc_centints(n: int, alpha: float, n_nu_points: int = 100):
        """"""

        max_val = 17
        plot_lh = max_val / (2 * n_nu_points)
        for nu in np.linspace(0.0, max_val, n_nu_points):
            if nu == 0.0:
                small_interval = [0, 0]
            else:
                small_interval = np.asarray(
                    find_smallest_interval(
                        pois_dist,
                        {"nu": nu},
                        alpha=alpha,
                        mode=nu,
                        step_size=max(nu_star * 0.2, 0.0001),
                        int_range_limit=[0.0, max_val * 2.5],
                    )
                )
            print(nu, round(small_interval[0]), round(small_interval[1]))

            if round(small_interval[0]) == round(small_interval[1]):
                plt.vlines(
                    round(small_interval[0]), nu - plot_lh, nu + plot_lh, colors="m"
                )
            else:
                plt.vlines(
                    round(small_interval[0]), nu - plot_lh, nu + plot_lh, colors="r"
                )
                plt.vlines(
                    round(small_interval[1]), nu - plot_lh, nu + plot_lh, colors="b"
                )
            # plt.plot(round(n), nu, marker="x", markersize=5, color="k")

        plt.xlabel("Events n [ ]")
        plt.ylabel("nu [ ]")
        plt.title(f"{1-alpha:0.2f}% Conf Interval, N={n}")
        # plt.show()

        plt.savefig(f"output/problem_s4_n{n}.png", dpi=300)

    calc_centints(n, alpha, n_nu_points=100)
