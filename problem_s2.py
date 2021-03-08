"""Script for solving statistics assignment problem 2"""

import numpy as np
import matplotlib.pyplot as plt

from tools import find_smallest_interval
from mathfunctions import bino_bay

energy = np.linspace(0.5, 4.0, 8)
trials = np.array([100, 100, 100, 100, 100, 1000, 1000, 1000])
success = np.array([0, 4, 20, 58, 92, 987, 995, 998])

p_star = success / trials
plot_lh = 0.1

for i in range(energy.shape[0]):
    small_interval = np.asarray(
        find_smallest_interval(
            bino_bay,
            {"n": trials[i], "r": success[i]},
            mode=p_star[i],
            step_size=0.00001,
            int_range_limit=[0., 1.]
        )
    )
    print(small_interval)

    plt.vlines(small_interval[0], energy[i] - plot_lh, energy[i] + plot_lh, colors="r")
    plt.vlines(small_interval[1], energy[i] - plot_lh, energy[i] + plot_lh, colors="b")
    plt.plot(p_star[i], energy[i], "x", color="k")

plt.xlabel("Efficiency p [ ]")
plt.ylabel("Energy [ ]")
plt.title("Efficiency estimates with 68% uncertainty")
# plt.show()

plt.savefig("output/problem_s2.png", dpi=300)
