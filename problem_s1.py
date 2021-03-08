"""Script for solving statistics assignment problem 1"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from mathfunctions import exp_dist
from tools import find_median, find_central_interval, find_mode, find_smallest_interval

"""
a
"""


mean = integrate.quad(lambda x: exp_dist(x=x, n=2), 0, np.inf)[0]
var = integrate.quad(lambda x: exp_dist(x=x, n=3), 0, np.inf)[0] - mean ** 2
sig = np.sqrt(var)
print(f"Mean = {mean}, Var = {var}, Sig = {sig:0.5f}")

prob_content = integrate.quad(lambda x: exp_dist(x=x, n=1), mean - sig, mean + sig)[0]
print(f"Probability content in the range mu +- sig: {prob_content:0.5f}")

"""
b
"""

median = find_median(exp_dist, {"n": 1})
print(f"Median = {median:0.5f}")
cent_interval = np.asarray(find_central_interval(exp_dist, {"n": 1}))
print(f"Central interval: {cent_interval}")

"""
c
"""

mode = find_mode(exp_dist, {"n": 1}, [0, 10])
print(f"Mode = {mode:0.5f}")
small_interval = np.asarray(find_smallest_interval(exp_dist, {"n": 1}, mode=mode))
print(f"Smallest interval: {small_interval}")

"""
Plotting results
"""

x = np.linspace(0.0, 10.0, 100)
plt.plot(x, exp_dist(x))
plt.plot(median, exp_dist(median), "x", label="median")
plt.plot(mode, exp_dist(mode), "x", label="mode")
plt.plot(mean, exp_dist(mean), "x", label="mean")
ymin = 0.15
ymax = 0.18
plt.vlines(small_interval[0], ymin, ymax, colors="r")
plt.vlines(small_interval[1], ymin, ymax, colors="r")
plt.vlines(cent_interval[0], ymin, ymax, colors="g")
plt.vlines(cent_interval[1], ymin, ymax, colors="g")
plt.errorbar(
    cent_interval.mean(),
    ymin,
    xerr=0.5*(cent_interval[1]-cent_interval[0]),
    label="cent_interval",
    color="g"
)
plt.errorbar(
    small_interval.mean(),
    ymax,
    xerr=0.5*(small_interval[1]-small_interval[0]),
    label="small_interval",
    color="r")
plt.legend()
plt.xlabel("x [a.u.]")
plt.ylabel("probability [ ]")
# plt.show()

plt.savefig("output/problem_s1.png", dpi=300)
