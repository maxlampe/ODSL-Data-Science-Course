"""Script for solving monte carlo assignment problem 1"""

import numpy as np
import matplotlib.pyplot as plt


def lcg(n_samples: int, a: float, c: float, m: int, x0: float):
    """Generate sample with Linear Congruential Generator"""

    sample = []
    x_curr = x0
    for i in range(n_samples):
        x_curr = (a * x_curr + c) % m
        sample.append(x_curr)

    sample = np.asarray(sample)

    return sample


x_0 = 3.0
n_samples = 1000000
data = lcg(n_samples=n_samples, a=1664525, c=1013904223, m=2 ** 32, x0=x_0) / 2 ** 32

print(f"Sample mean = {data.mean():0.5f}, Sample variance = {data.var():0.5f}")
print(f"Theoretical expected values: mean = 0.5, variance = {1./12.:0.5f}")

plt.hist(data, bins=int(0.1 * np.sqrt(n_samples)))
plt.title("Linear Congruential Generator")
plt.xlabel("x [ ]")
plt.ylabel("a.u. [ ]")
plt.annotate(
    f"mean = {data.mean():0.5f}\n" f"variance = {data.var():0.5f}",
    xy=(0.05, 0.25),
    xycoords="axes fraction",
    ha="left",
    va="top",
    bbox=dict(boxstyle="round", fc="1"),
)
# plt.show()
plt.savefig(f"output/problem_m1.png", dpi=300)
