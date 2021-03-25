"""Script for solving monte carlo assignment problem 5"""

import numpy as np
import matplotlib.pyplot as plt


def target_dist(x: float, par_r: float):
    """Target distribution for rejection sampling"""

    fac = 2.0 / (np.pi * par_r ** 2)
    return fac * np.sqrt(par_r ** 2 - x ** 2)


def cover_func1(x: float, par_h: float):
    """Box function as simple covering function"""
    try:
        res = np.asarray([par_h] * len(x))
    except TypeError:
        res = par_h
    return res


def acc_rej_method(
    t_func, t_par: dict, c_func, c_par: dict, x_range: list, n_trials: int = 100
):
    """Rejection sampling algorithm"""

    sample = []
    for i in range(n_trials):
        u_1 = np.random.uniform(x_range[0], x_range[1])
        u_2 = np.random.uniform(0.0, c_func(x=u_1, **c_par))
        if u_2 <= t_func(x=u_1, **t_par):
            sample.append(u_1)
    sample = np.asarray(sample)

    efficiency = sample.shape[0] / n_trials

    return sample, efficiency


n_trial = 1000000
par_r = 1.0
par_h = 2.0 / (np.pi * par_r)

run_1, eff1 = acc_rej_method(
    target_dist,
    {"par_r": par_r},
    cover_func1,
    {"par_h": par_h},
    x_range=[-par_r, par_r],
    n_trials=n_trial,
)

x_vals = np.linspace(-par_r, par_r, 1000)
plt.hist(run_1, bins=100, density=True)
plt.plot(x_vals, target_dist(x=x_vals, par_r=par_r), label="target dist.")
plt.plot(x_vals, cover_func1(x=x_vals, par_h=par_h), label="covering func.")
plt.title("Rejection Sampling")
plt.xlabel("x [ ]")
plt.ylabel("a.u. [ ]")
plt.legend()
plt.annotate(
    f"Eff. = {eff1:0.3f}",
    xy=(0.05, 0.25),
    xycoords="axes fraction",
    ha="left",
    va="top",
    bbox=dict(boxstyle="round", fc="1"),
)
# plt.savefig("output/problem_m5.png", dpi=300)
plt.show()
