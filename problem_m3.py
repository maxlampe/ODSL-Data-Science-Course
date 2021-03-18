"""Script for solving monte carlo assignment problem 3"""

import numpy as np
import matplotlib.pyplot as plt


def theo_dist(x: float):
    """Distribution calculated in MC assignment 2"""
    return 0.5 * np.log(x) ** 2


def theo_cdf(x: float):
    """CDF calculated in MC assignment 2"""
    return 0.5 * x * np.log(x) ** 2 - x * (np.log(x) - 1.0)


def sample_cdf(x: float, data: np.array):
    """CDF of sample for Kolmogorov-Smirnov test"""

    n = data.shape[0]
    result = []
    for elem in x:
        result.append((data <= elem).sum() / n)
    result = np.asarray(result)

    return result


def kolmogorov_smirnov(ftheo, fsample, param1, param2, xrange=None):
    """Calculate max. deviation for Kolmogorov-Smirnov test"""
    if xrange is None:
        xrange = [0.0, 1.0]
    x_eval = np.linspace(xrange[0], xrange[1], 1000)

    return np.abs(ftheo(x_eval, **param1) - fsample(x_eval, **param2)).max()


n_points = 1000000
x = np.random.rand(n_points, 3)
data = x.T[0] * x.T[1] * x.T[2]

ks_test = kolmogorov_smirnov(
    theo_cdf,
    sample_cdf,
    param1={},
    param2={"data": data},
    xrange=[0.005, 1.0],
)


fig, axs = plt.subplots(1, 2, sharex=True, figsize=(14, 7))
fig.suptitle("Kologorov-Smirnov Test")
x_plot = np.linspace(0.005, 1.0, 1000)
axs[0].hist(data, bins=100, density=True, label="gen. sample")
axs[0].plot(x_plot, theo_dist(x_plot), label="0.5 * log(x)**2")
axs[0].legend()
axs[0].set_title("Comparing generated sample with theoretical distribution")
axs[0].set_xlabel("x [ ]")
axs[0].set_ylabel("a.u. [ ]")

axs[1].plot(
    x_plot,
    np.abs(sample_cdf(x_plot, data) - theo_cdf(x_plot)),
    label=f"KS result: {ks_test:0.8f}",
)
axs[1].legend()
axs[1].set_title("Abs. difference of CDFs of sample and theory")
axs[1].set_xlabel("x [ ]")
axs[1].set_ylabel("abs. diff [ ]")

plt.tight_layout()
plt.show()
# plt.savefig("output/problem_m3.png", dpi=300)
