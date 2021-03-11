"""Script for solving monte carlo assignment problem 4"""

import numpy as np
import matplotlib.pyplot as plt


def gen_exp_sample(n_sim: int = 100, mu: float = 1.0):
    """Generate sample according to f(x) = mu * np.exp(-mu * x)"""

    x_vals = np.random.rand(n_sim)
    return -np.log(x_vals) / mu


def wait_nth_event(t: float, n: int):
    """"""
    pass


sample = gen_exp_sample(n_sim=1000)
wait_time = sample[1:] - sample[:-1]

plt.hist(wait_time, density=True)
plt.show()
