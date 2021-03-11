"""Daily assignment 6_1"""

import numpy as np
import matplotlib.pyplot as plt
from mathfunctions import gaussian


def metropolis(func, params, n_trials: int = 1000, prop_rng=None, x0: float = 0.0):
    """Metropolis Algorithm"""

    if prop_rng is None:
        prop_rng = [-1.0, 1.0]

    x_curr = x0
    accept = []
    for i in range(n_trials):
        u_1 = np.random.uniform(prop_rng[0], prop_rng[1])
        u_2 = np.random.uniform()
        y_curr = x_curr + u_1

        rho = min(func(y_curr, **params) / func(x_curr, **params), 1.0)

        if u_2 <= rho:
            accept.append(y_curr)
            x_curr = y_curr
        else:
            accept.append(x_curr)

    accept = np.asarray(accept)
    efficiency = accept.shape[0] / n_trials
    print(f"# Accepted points: {100. * efficiency:0.2f}%")

    return np.asarray([accept, efficiency])


n_sim = 100000
tests = [
    [[-1.0, 1.0], 0.0],
    [[-0.5, 0.5], 0.0],
    [[-0.1, 0.1], 0.0],
    [[-3.0, 3.0], 0.0],
    [[-1.0, 1.0], 1.0],
    [[-1.0, 0.5], 0.5],
]

for test in tests:
    res = metropolis(
        gaussian, {"mu": 0.0, "sig": 1.0}, n_sim, prop_rng=test[0], x0=test[1]
    )
    test.append(res[0])
    test.append(res[1])

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
fig.suptitle(f"Metropolis Gaussian Tests - {n_sim} Iterations")
x_vals = np.linspace(-4.5, 4.5, 1000)
bins = int(n_sim * 0.001)

for test_i, test in enumerate(tests):
    axs.flat[test_i].hist(
        test[2], bins=bins, range=[-4.5, 4.5], label="gen. RN", density=True
    )
    axs.flat[test_i].plot(x_vals, gaussian(x_vals), label="Unit. Gaussian")
    axs.flat[test_i].legend()
    axs.flat[test_i].set_xlabel("x [ ]")
    axs.flat[test_i].set_ylabel("a.u. [ ]")
    axs.flat[test_i].set_title(f"Test {test_i + 1}")
    axs.flat[test_i].annotate(
        f"prop. rng = {test[0]} \n" f"x0 = {test[1]}\n" f"eff = {test[3]:0.2}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="1"),
    )

plt.show()
