""""""

import numpy as np
import scipy.integrate as integrate

DEBUG = True


def int_from_point(
        func,
        params: dict,
        target_p: float = 0.5,
        step_size: float = 0.01,
        int_start: float = 1.,
        int_range_limit: list = [0., np.inf]
):
    """x needs to be named x"""

    target_precision = 0.001
    step_precision = min(step_size * 10., target_p * 2)
    curr_lolim = int_start - step_size
    curr_uplim = int_start + step_size
    curr_int = None

    for i in range(50000):
        if int_range_limit[0] > curr_lolim:
            curr_lolim = int_range_limit[0]
        if int_range_limit[1] < curr_uplim:
            curr_uplim = int_range_limit[1]

        curr_int = integrate.quad(
            lambda x: func(x=x, **params), curr_lolim, curr_uplim
        )[0]

        if np.abs(curr_int - target_p) <= target_precision:
            if True:
                print(f"Reached final precision: {curr_int:0.5f}\t{np.abs(curr_int - target_p):0.5f}")
            break

        if np.abs(curr_int - target_p) <= step_precision:
            if DEBUG:
                print(f"Reached step precision: {curr_int:0.5f}\t{np.abs(curr_int - target_p):0.5f}")
            step_size = step_size * 0.1
            step_precision = step_precision * 0.1

        lower = func(x=(curr_lolim - step_size), **params)
        upper = func(x=(curr_uplim + step_size), **params)

        if upper >= lower:
            curr_uplim = curr_uplim + step_size
        else:
            curr_lolim = curr_lolim - step_size

    return np.array([curr_lolim, curr_uplim, curr_int, np.abs(curr_int - target_p)])


def int_from_zero(
        func,
        params: dict,
        target_p: float = 0.5,
        step_size: float = 0.01,
        int_start: float = 1.,
        int_range_limit: float = np.inf
):
    """x needs to be named x"""

    target_precision = 0.001
    step_precision = min(step_size * 10., target_p*2)
    curr_lolim = 0.
    curr_uplim = int_start + 0.000001
    curr_int = None

    for i in range(50000):
        if 0. > curr_lolim:
            curr_lolim = 0.
        if int_range_limit < curr_uplim:
            curr_uplim = int_range_limit

        curr_int = integrate.quad(
            lambda x: func(x=x, **params), curr_lolim, curr_uplim
        )[0]

        if np.abs(curr_int - target_p) <= target_precision:
            if True:
                print(f"Reached final precision: {curr_int:0.5f}\t{np.abs(curr_int - target_p):0.5f}")
            break

        step_deviation = curr_int - target_p
        if np.abs(step_deviation) <= step_precision:
            if DEBUG:
                print(f"Reached step precision: {curr_int:0.5f}\t{np.abs(step_deviation):0.5f}")
            step_size = step_size * 0.1
            step_precision = step_precision * 0.1

        if step_deviation > 0:
            curr_uplim = curr_uplim - step_size
        else:
            curr_uplim = curr_uplim + step_size

    return np.array([curr_lolim, curr_uplim, curr_int, np.abs(curr_int - target_p)])


def find_median(func, params: dict):
    """"""

    target_p = 0.5
    median_result = int_from_zero(
        func=func, params=params, target_p=target_p, int_start=0
    )

    return median_result[1]


def find_mode(func, params: dict, search_range: list):
    """"""

    x_vals = np.linspace(search_range[0], search_range[1], int(10e4))
    y_vals = func(x=x_vals, **params)

    mode = x_vals[y_vals.argmax()]

    return mode


def find_central_interval(
        func,
        params: dict,
        alpha: float = 0.32,
        step_size=0.01,
        int_range_limit=np.inf
):
    """"""

    target_p = alpha * 0.5
    lower_bound = int_from_zero(
        func=func,
        params=params,
        target_p=target_p,
        int_start=0,
        step_size=step_size,
        int_range_limit=int_range_limit
    )[1]
    target_p = 1. - alpha * 0.5
    upper_bound = int_from_zero(
        func=func,
        params=params,
        target_p=target_p,
        int_start=lower_bound,
        step_size=step_size,
        int_range_limit=int_range_limit
    )[1]

    return [lower_bound, upper_bound]


def find_smallest_interval(
        func,
        params: dict,
        mode: float,
        alpha: float = 0.32,
        step_size: float = 0.01,
        int_range_limit: list = [0., np.inf]
):
    """"""

    target_p = 1. - alpha
    bounds = int_from_point(
        func=func,
        params=params,
        target_p=target_p,
        int_start=mode,
        step_size=step_size,
        int_range_limit=int_range_limit
    )

    return [bounds[0], bounds[1]]


def find_fc_int(func, params: dict, nu_bg: float, alpha: float = 0.32):
    """"""

    pass
