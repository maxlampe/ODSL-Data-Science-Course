"""Script for solving monte carlo assignment problem 8"""

import numpy as np
import matplotlib.pyplot as plt


def polar_to_cart(r: np.array, phi: np.array):
    """"""

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return [x, y]


def cart_to_polar(x: np.array, y: np.array):
    """"""

    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan(y / x)

    return [r, phi]


def rayleigh_dist(x: float, n: int):
    """Rayleigh distribution as seen in the lecture"""
    return (2 * x / n) * np.exp(-(x ** 2) / n)


def do_rand_walk(n_steps: int = 100000):
    """Do random walk for uniform angle and exp.-dist. step size"""

    u = np.random.rand(2, n_steps)
    step = -np.log(u[0])
    angle = u[1] * 2 * np.pi

    x_steps, y_steps = polar_to_cart(step, angle)

    final_pos = np.asarray([x_steps.sum(), y_steps.sum()])
    final_dist, _ = cart_to_polar(*final_pos)

    return final_dist


n_steps = 100000

res = []
for i in range(10000):
    res.append(do_rand_walk(n_steps=n_steps))
res = np.asarray(res)


x_vals = np.linspace(0.0, 1200)
plt.hist(res, bins=50, density=True)
plt.plot(x_vals, rayleigh_dist(x=x_vals, n=n_steps), label="const. step size")
plt.plot(
    x_vals, rayleigh_dist(x=x_vals * np.log(2), n=n_steps) * np.log(2), label="modified"
)
plt.legend()
plt.xlabel("Distance from origin [ ]")
plt.ylabel("a.u. [ ]")
plt.title("Random walk with exp.-dist. step size")
plt.savefig("output/problem_m8.png", dpi=300)
plt.show()
