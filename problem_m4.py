"""Script for solving monte carlo assignment problem 4"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def gen_exp_sample(n_sim: int = 100, mu: float = 1.0):
    """Generate sample according to f(x) = mu * np.exp(-mu * x)"""

    x_vals = np.random.rand(n_sim)
    vals = -np.log(x_vals) / mu
    return vals.sum()


def wait_anal(t: float, n: int):
    """Analytic expression for waiting time"""
    return t ** (n - 1) * np.exp(-t) / gamma(n)


def compare_results(n_target: int, bsave_fig: bool = False):
    """Compare generated and analytic results"""

    sample = []
    for i in range(500000):
        sample.append(gen_exp_sample(n_sim=n_target))
    sample = np.asarray(sample)

    t_plot = np.linspace(0.0, max(10, 2 * n_target), 1000)
    plt.hist(sample, bins=100, density=True, label="gen. samples")
    plt.plot(t_plot, wait_anal(t_plot, n_target), label="anal. dist.")
    plt.title("Generated and analytical results for waiting time")
    plt.xlabel("Waiting time t [ ]")
    plt.ylabel("P_n (t)[ ]")
    plt.legend()
    if bsave_fig:
        plt.savefig(f"output/problem_m4_{n_target}.png", dpi=300)
    plt.show()


compare_results(1, bsave_fig=True)
compare_results(10, bsave_fig=True)
compare_results(100, bsave_fig=True)
