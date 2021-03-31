"""Script for solving statistics assignment problem 7"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mathfunctions import pois_dist
from scipy import integrate
from scipy.special import factorial


data1 = np.asarray([0.5, 0.5, 0.5, 0.5])
data2 = np.asarray([0.2, 0.5, 0.75, 0.9])


def f_problem(x: float, fac: float):
    """Function given in problem"""
    return fac*x


def big_lambda(func, f_par: dict, a: float = 0., b: float = 1.):
    """Expectation in bin from a to b"""
    return integrate.quad(lambda x: func(x=x, **f_par), a, b)[0]


def norm_unb_likelihood(data: np.array, func, f_par, a: float = 0., b: float = 1.):
    """Normalized unbinned likelihood"""

    return (func(data, **f_par) / big_lambda(func, f_par, a, b)).prod()


def ext_likelihood(data: np.array, func, f_par: dict):
    """Extended likelihood"""

    nul = norm_unb_likelihood(data, func, f_par)
    lam = big_lambda(func, f_par)
    n = data.shape[0]
    corr = lam ** n * np.exp(-lam) / factorial(n)

    return corr * nul


def binned_poisson(data: np.array, func, f_par: dict):
    """Binned poisson probability"""

    lam = big_lambda(func, f_par)
    n = data.shape[0]
    corr = lam ** n * np.exp(-lam) / factorial(n)

    return corr


print("For dataset 1")
print(norm_unb_likelihood(data1, f_problem, {"fac": 10}))
print(ext_likelihood(data1, f_problem, {"fac": 10}))
print(binned_poisson(data1, f_problem, {"fac": 10}))


print("For dataset 2")
print(norm_unb_likelihood(data2, f_problem, {"fac": 10}))
print(ext_likelihood(data2, f_problem, {"fac": 10}))
print(binned_poisson(data2, f_problem, {"fac": 10}))
