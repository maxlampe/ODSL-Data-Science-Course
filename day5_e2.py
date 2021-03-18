"""Daily assignment 5_2"""

import numpy as np
import matplotlib.pyplot as plt


def int_func(x: float) -> float:
    """Function to integrate"""
    return (np.cos(50.0 * x) + np.sin(20.0 * x)) ** 2


def mc_int(func, int_range: list, n_sim: int = 100, y_max: float = None) -> float:
    """Monte Carlo hit and miss integration for a given function"""

    if y_max is None:
        x = np.linspace(int_range[0], int_range[1], 1000)
        y = func(x)

        y_max = y.max() * 1.1

    u_1 = np.random.rand(n_sim) * int_range[1] - int_range[0]
    u_2 = np.random.rand(n_sim) * y_max

    mask = int_func(u_1) > u_2
    max_area = y_max * (int_range[1] - int_range[0])
    valid_area = (u_1[mask].shape[0] / u_1.shape[0]) * max_area

    return valid_area


true_integral = 0.9652
n_sim = int(2e4)
n_step = int(n_sim / 1000.0)


diff = []
n_iter = range(100, n_sim, n_step)
for n in n_iter:
    res = mc_int(int_func, [0, 1], n_sim=n, y_max=4.0)
    print(f"{n}\t{res:0.5f}\t{res-true_integral:0.5f}")
    diff.append((res - true_integral) / true_integral)
diff = np.asarray(diff)


plt.plot(n_iter, diff, ".")
plt.plot(n_iter, 1.0 / np.sqrt(np.asarray(n_iter)), color="r")
plt.plot(n_iter, -1.0 / np.sqrt(np.asarray(n_iter)), color="r")
plt.title("MC Integration Deviation")
plt.xlabel("n_sim points [ ]")
plt.ylabel("rel deviation [ ]")
plt.savefig("output/day5_e2.png", dpi=300)
plt.show()
