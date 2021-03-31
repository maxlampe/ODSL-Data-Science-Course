"""Script for solving monte carlo assignment problem 6"""

import numpy as np
import matplotlib.pyplot as plt
from tools import M6Hist


def target_dist(x: float, par_r: float):
    """Target distribution for rejection sampling"""

    fac = 2.0 / (np.pi * par_r ** 2)
    return fac * np.sqrt(par_r ** 2 - x ** 2)


def gen_sample(n_sample_size: int = 1000, par_r: float = 1.0):
    """Generate one sample of the target distribution for a given sample size."""

    samples = []
    weights = []
    for i in range(n_sample_size):
        u = np.random.uniform(-par_r, par_r)
        weight = 2 * par_r * target_dist(x=u, par_r=par_r)
        samples.append(u)
        weights.append(weight)
    weights = np.asarray(weights)
    samples = np.asarray(samples)

    return samples, weights


def sim_variance(
    n_trials: int = 20, n_sample_size: int = 100, par_r: float = 1.0, bplot_last=False
):
    """Numerically obtain variance of generated sample from target density"""

    var = np.asarray([])
    for i in range(n_trials):
        samp, weig = gen_sample(
            n_sample_size=n_sample_size,
            par_r=par_r,
        )

        hist_class = M6Hist(
            data=samp,
            weights=weig,
            bincount=int(np.sqrt(n_sample_size) * 0.5),
            range_lowlim=-par_r,
            range_uplim=par_r,
        )

        deviations = hist_class.calc_deviations(target_dist, {"par_r": par_r})
        var = np.append(var, deviations.mean())

        if bplot_last and i == n_trials - 1:
            hist_class.plot_hist()
            x_vals = np.linspace(-par_r, par_r, 1000)
            plt.plot(
                x_vals,
                target_dist(x=x_vals, par_r=par_r),
                label="target dist.",
                color="r",
            )
            plt.legend()
            plt.title("Generated sample")
            plt.xlabel("x [ ]")
            plt.ylabel("a.u. [ ]")
            plt.savefig(f"output/problem_m6_gen{n_sample_size}.png", dpi=300)
            plt.show()

    return var.mean()


sim_variance(n_trials=1, n_sample_size=100, bplot_last=True)
sim_variance(n_trials=1, n_sample_size=1000, bplot_last=True)
sim_variance(n_trials=1, n_sample_size=10000, bplot_last=True)


n_vals = np.concatenate(
    [np.linspace(10, 1000, 30, dtype=int), np.linspace(1000, 100000, 10, dtype=int)]
)
n_vals = np.linspace(10, 100000, 30, dtype=int)
variances = np.asarray([sim_variance(n_sample_size=n) for n in n_vals])
n_plot = np.linspace(10, 100000, 1000)

plt.plot(n_vals, variances, label="sim. deviation")
plt.plot(n_plot, 0.14 / np.sqrt(n_plot), label="calc. deviation")
plt.yscale("log")
plt.legend()
plt.title("Numerical and analytical deviation")
plt.xlabel("x [ ]")
plt.ylabel("a.u. [ ]")
plt.savefig("output/problem_m6_var.png", dpi=300)
plt.show()
