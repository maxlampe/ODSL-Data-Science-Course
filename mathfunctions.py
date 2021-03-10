"""Define math functions here"""

import numpy as np
# from math import factorial
from scipy.special import comb, factorial


def exp_dist(x: float, n: float = 1):
    """"""
    return x ** n * np.exp(-x)


def bino_dist(x: int, n: int, p: float):
    """"""
    return comb(n, x) * p**x * (1-p)**(n-x)


def bino_bay(x: float, n: int, r: int):
    """Flat prior"""
    return bino_dist(x=r, n=n, p=x) * (n + 1)


def pois_dist(x: int, nu: float):
    """"""
    x = np.asarray(x, dtype=int)
    return np.exp(-nu) * nu**x / factorial(x)


def pois_bay(x: float, n: int):
    """Flat prior"""
    return np.exp(-x) * x**n / factorial(n)


