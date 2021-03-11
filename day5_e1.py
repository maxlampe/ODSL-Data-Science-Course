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


n_sim = 300000

u = np.random.rand(n_sim, 3) * 2.0 - 1.0
r, p, t = cart_to_polar(u).T

filt_mask = r <= 1.0
p_valid = p[filt_mask]
t_valid = t[filt_mask]
r_1s = np.ones(filt_mask.sum())

sphere_points = np.asarray([r_1s, p_valid, t_valid]).T
x, y, z = polar_to_cart(sphere_points).T

plt.hist2d(x=y, y=z)
plt.show()
plt.hist(x=x)
plt.show()
plt.hist(x=y)
plt.show()
plt.hist(x=z)
plt.show()
