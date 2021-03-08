"""Script for solving statistics assignment problem 3"""

import numpy as np
import matplotlib.pyplot as plt

from tools import find_central_interval
from mathfunctions import bino_dist

energy = np.linspace(0.5, 4.0, 8)
trials = np.array([100, 100, 100, 100, 100, 1000, 1000, 1000])
success = np.array([0, 4, 20, 58, 92, 987, 995, 998])


def calc_centints(n: int, alpha: float, n_p_points: int = 10):
    """"""

    plot_lh = 1./(2 * n_p_points)
    for p in np.linspace(0., 1., n_p_points):
        if p == 0.:
            cent_interval = [0, 0]
        elif p == 1.:
            cent_interval = [n, n]
        else:
            cent_interval = np.asarray(
                find_central_interval(
                    bino_dist,
                    {"n": n, "p": p},
                    alpha=alpha,
                    step_size=max(n*p*alpha*0.2, 0.0001),
                    int_range_limit=n)
            )
        print(p, round(cent_interval[0]), round(cent_interval[1]))

        if round(cent_interval[0]) == round(cent_interval[1]):
            plt.vlines(round(cent_interval[0]), p - plot_lh, p + plot_lh, colors="m")
        else:
            plt.vlines(round(cent_interval[0]), p - plot_lh, p + plot_lh, colors="r")
            plt.vlines(
                round(cent_interval[1]), p - plot_lh, p + plot_lh, colors="b"
            )
        plt.plot(round(n*p), p, marker="x", markersize=5, color="k")

    plt.xlabel("successes r [ ]")
    plt.ylabel("prob p [ ]")
    plt.title(f"{1-alpha}% Conf Interval, N={n}")
    # plt.show()

    plt.savefig(f"output/problem_s3_n{n}.png", dpi=300)


calc_centints(100, 0.1, n_p_points=10)
calc_centints(1000, 0.1, n_p_points=10)

"""
used 100 instead of 10 points for p during conf interval construction
only 10 for plots

E   |  success | trials | p range
--------------
0.5 0   100 [0, 0]
1.0 4   100 [0.01, 0.09]
1.5 20  100 [0.13, 0.28]
2.0 58  100 [0.49, 0.66]
2.5 92  100 [0.86, 0.96]
3.0 987 1000 [0.975, 0.996]
3.5 995 1000 [0.985, 1.000]
4.0 998 1000 [0.992, 1.000]

"""