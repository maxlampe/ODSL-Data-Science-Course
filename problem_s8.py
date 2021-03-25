"""Script for solving statistics assignment problem 8"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mathfunctions import bino_dist


def sigmoid(x: float, a: float = 3.0, e0: float = 2.0):
    """Sigmoid function to fit"""
    return 1.0 / (1 + np.exp(-a * (x - e0)))


def epsilon(x: float, a: float = 3.0, e0: float = 2.0):
    """Epsilon function to compare"""
    return np.sin(a * (x - e0))


def test_statistic(meas_data: pd.DataFrame, t_func, t_par: dict):
    """Product of data points of binomial distribution as test statistic"""

    return bino_dist(
        x=meas_data["successes"],
        n=meas_data["trials"],
        p=t_func(meas_data["energy"], **t_par),
    ).prod()



"""
https://tum.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=741b33b6-89fa-4199-bf47-ace0011edf59
around 45 min

This is frequentist

1) create grid point of a and e0
2) for each E get success prob p
3) generate 8 numbers acc. to succ. prob.
4) calc test stat with that
5) get dist of test stat P (T | a, e0)


e.g. a = 3., e0 = 1.6
sigmoid(0.5, 3., 1.6) * 100 = 3.56 (to get expected value)
do 10k experiments to get dist for r
generate samples and calculate dist of test stat

find if meas values in 68% or whatever of dist of test stat
one entry on global a, e0 grid map


https://tum.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=741b33b6-89fa-4199-bf47-ace0011edf59
around 59 min

Bayesian

get P( a, e0 | data ) with const priors
"""


data = pd.DataFrame()
data["energy"] = np.linspace(0.5, 4.0, num=8)
data["trials"] = [100] * 8
data["successes"] = [0.0, 4.0, 22.0, 55.0, 80.0, 97.0, 99.0, 99.0]

print(test_statistic(data, sigmoid, {"a": 3.0, "e0": 2.0}))

x_vals = np.linspace(0.0, 4.0, 1000)
plt.plot(x_vals, sigmoid(x_vals))
plt.plot(data["energy"], data["successes"] / 100.0, ".")
plt.show()
print(data)
