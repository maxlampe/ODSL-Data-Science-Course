"""Daily assignment 5_1"""

import numpy as np
import matplotlib.pyplot as plt


def cart_to_polar(vec: np.array):
    """Transform an array of [x, y, z] cartesian coordinates to polar coordinates."""

    r = np.sqrt(vec.T[0] ** 2 + vec.T[1] ** 2 + vec.T[2] ** 2)
    phi = np.arctan(vec.T[1] / vec.T[0])
    theta = np.arccos(vec.T[2] / r)

    return np.array([r, phi, theta]).T


def polar_to_cart(vec: np.array):
    """Transform an array of [r, phi, theta] polar coordinates to cartesian coord."""

    x = vec.T[0] * np.cos(vec.T[1]) * np.sin(vec.T[2])
    y = vec.T[0] * np.sin(vec.T[1]) * np.sin(vec.T[2])
    z = vec.T[0] * np.cos(vec.T[2])

    return np.array([x, y, z]).T


n_sim = 3000000

u = np.random.rand(n_sim, 3) * 2.0 - 1.0
r, p, t = cart_to_polar(u).T

filt_mask = r <= 1.0
p_valid = p[filt_mask]
t_valid = t[filt_mask]
r_1s = np.ones(filt_mask.sum())

sphere_points = np.asarray([r_1s, p_valid, t_valid]).T
x, y, z = polar_to_cart(sphere_points).T

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

axs.flat[0].hist2d(x=y, y=z, bins=100, density=True)
axs.flat[0].set_title("yz 2D-Hist")
axs.flat[0].set(xlabel="y [ ] ", ylabel="z [ ]")
axs.flat[1].hist(x=x, bins=100, density=True)
axs.flat[1].set_title("x Distribution")
axs.flat[1].set(xlabel="x [ ] ", ylabel="a.u. [ ]")
axs.flat[2].hist(x=y, bins=100, density=True)
axs.flat[2].set_title("y Distribution")
axs.flat[2].set(xlabel="y [ ] ", ylabel="a.u. [ ]")
axs.flat[3].hist(x=z, bins=100, density=True)
axs.flat[3].set_title("z Distribution")
axs.flat[3].set(xlabel="z [ ] ", ylabel="a.u. [ ]")
plt.tight_layout()
plt.savefig("output/day5_e1.png", dpi=300)
plt.show()
