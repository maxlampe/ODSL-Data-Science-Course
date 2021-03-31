"""Script for solving statistics assignment problem 8"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mathfunctions import bino_dist
from scipy import integrate


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


def post_prob(meas_data: pd.DataFrame, t_func, t_par: dict):
    """Product of data points of binomial distribution as test statistic"""

    prob_given_data = test_statistic(meas_data=meas_data, t_func=t_func, t_par=t_par)
    f = lambda x, y: test_statistic(meas_data=meas_data, t_func=t_func, t_par={"a": x, "e0": y})
    # int = integrate.dblquad(f, a=0, b=np.inf, gfun=0, hfun=np.inf)[0]
    int = integrate.dblquad(f, a=0, b=(2*np.pi), gfun=0, hfun=10)[0]

    return prob_given_data / int


data = pd.DataFrame()
data["energy"] = np.linspace(0.5, 4.0, num=8)
data["trials"] = [100] * 8
data["successes"] = [0.0, 4.0, 22.0, 55.0, 80.0, 97.0, 99.0, 99.0]


# Part a)
if False:
    n_sim = 20

    a_vals, e0_vals = np.meshgrid(np.linspace(2.6, 3.4, num=n_sim), np.linspace(1.85, 2.05, num=n_sim))
    z = np.ones([a_vals.shape[0], e0_vals.shape[0]])
    for i in range(a_vals.shape[0]):
        for j in range(e0_vals.shape[0]):
            z[i][j] = post_prob(data, sigmoid, {"a": a_vals[i][j], "e0": e0_vals[i][j]})

    fig, axs = plt.subplots(figsize=(8, 8), constrained_layout=True)
    fig.suptitle("Posterior probability P(A, E0 | Data)")

    cs = axs.contourf(a_vals, e0_vals, z)
    # cs2 = axs.contour(cs, levels=cs.levels[::2], colors='r')
    axs.set_xlabel("A [ ]")
    axs.set_ylabel("E_0 [ ]")
    # cbar = fig.colorbar(cs)
    # cbar.ax.set_ylabel('Probability')
    # cbar.add_lines(cs2)

    plt.savefig("output/problem_s8_postprob.png", dpi=300)
    plt.show()

# Part c.1)
if False:
    n_sim = 3

    a_vals, e0_vals = np.meshgrid(np.linspace(0., 1., num=n_sim), np.linspace(0., 1., num=n_sim))
    z = np.ones([a_vals.shape[0], e0_vals.shape[0]])
    for i in range(a_vals.shape[0]):
        for j in range(e0_vals.shape[0]):
            z[i][j] = post_prob(data, epsilon, {"a": a_vals[i][j], "e0": e0_vals[i][j]})
            print(i, j,  a_vals[i][j], e0_vals[i][j], z[i][j])

    fig, axs = plt.subplots(figsize=(8, 8), constrained_layout=True)
    fig.suptitle("Posterior probability P(A, E0 | Data)")

    cs = axs.contourf(a_vals, e0_vals, z)
    # cs2 = axs.contour(cs, levels=cs.levels[::2], colors='r')
    axs.set_xlabel("A [ ]")
    axs.set_ylabel("E_0 [ ]")
    # cbar = fig.colorbar(cs)
    # cbar.ax.set_ylabel('Probability')
    # cbar.add_lines(cs2)

    plt.savefig("output/problem_s8_postprob_sin.png", dpi=300)
    plt.show()

# Part b)

if True:
    n_sim = 2000
    n_grid = 20

    a_vals, e0_vals = np.meshgrid(np.linspace(2.5, 3.4, num=n_grid), np.linspace(1.85, 2.05, num=n_grid))
    z = np.zeros([a_vals.shape[0], e0_vals.shape[0]])
    for i in range(a_vals.shape[0]):
        for j in range(e0_vals.shape[0]):

            a = a_vals[i][j]
            e0 = e0_vals[i][j]

            p_dist = []
            for n in range(n_sim):
                data_tmp = pd.DataFrame()
                data_tmp["energy"] = np.linspace(0.5, 4.0, num=8)
                data_tmp["trials"] = [100] * 8

                samples = []
                for e in data_tmp["energy"]:
                    samples.append(np.random.binomial(n=100, p=sigmoid(e, a, e0)))
                data_tmp["successes"] = samples

                p_dist.append(test_statistic(data_tmp, sigmoid, {"a": a, "e0": e0}))

            p_dist = np.asarray(p_dist)
            p_dist.sort()
            p_sort = p_dist[::-1]
            p_sim = p_sort[int(n_sim * 0.68)]

            p_meas = test_statistic(data, sigmoid, {"a": a, "e0": e0})
            print(f"{a:0.2f} \t {e0:0.2f} \t {p_meas} \t {p_sim}")
            if p_meas >= p_sim:
                print("Got one")
                z[i][j] = 1.

    fig, axs = plt.subplots(figsize=(8, 8), constrained_layout=True)
    fig.suptitle("Confidence region for A and E0 with sigmoid")

    cs = axs.contourf(a_vals, e0_vals, z)
    axs.set_xlabel("A [ ]")
    axs.set_ylabel("E_0 [ ]")

    plt.savefig("output/problem_s8_conflvl.png", dpi=300)
    plt.show()


    """
    https://tum.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=741b33b6-89fa-4199-bf47-ace0011edf59
    around 45 min

    This is frequentist

    1) create grid point of a and e0
    2) for each E get success prob p given a-e0-point
    3) generate 8 numbers acc. to succ. prob. 
    4) calc test statistic with that
    5) get distribution of test stat P (T | a, e0)


    e.g. a = 3., e0 = 1.6
    sigmoid(0.5, 3., 1.6) * 100 = 3.56 (to get expected value)
    do 10k experiments to get dist for r
    generate samples and calculate dist of test stat

    find if meas values in 68% or whatever of dist of test stat
    one entry on global a, e0 grid map
    """

# Part c.1)

if False:
    n_sim = 2000
    n_grid = 5

    a_vals, e0_vals = np.meshgrid(np.linspace(0.4, 0.7, num=n_grid), np.linspace(0.8, 1.0, num=n_grid))
    z = np.zeros([a_vals.shape[0], e0_vals.shape[0]])
    for i in range(a_vals.shape[0]):
        for j in range(e0_vals.shape[0]):

            a = a_vals[i][j]
            e0 = e0_vals[i][j]

            p_dist = []
            for n in range(n_sim):
                data_tmp = pd.DataFrame()
                data_tmp["energy"] = np.linspace(0.5, 4.0, num=8)
                data_tmp["trials"] = [100] * 8

                samples = []
                for e in data_tmp["energy"]:
                    samples.append(np.random.binomial(n=100, p=sigmoid(e, a, e0)))
                data_tmp["successes"] = samples

                p_dist.append(test_statistic(data_tmp, epsilon, {"a": a, "e0": e0}))

            p_dist = np.asarray(p_dist)
            p_dist.sort()
            p_sort = p_dist[::-1]
            p_sim = p_sort[int(n_sim * 0.68)]

            p_meas = test_statistic(data, sigmoid, {"a": a, "e0": e0})
            print(f"{a:0.2f} \t {e0:0.2f} \t {p_meas} \t {p_sim}")
            if p_meas >= p_sim:
                print("Got one")
                z[i][j] = 1.

    fig, axs = plt.subplots(figsize=(8, 8), constrained_layout=True)
    fig.suptitle("Confidence region for A and E0 with sigmoid")

    cs = axs.contourf(a_vals, e0_vals, z)
    axs.set_xlabel("A [ ]")
    axs.set_ylabel("E_0 [ ]")

    plt.savefig("output/problem_s8_conflvl.png", dpi=300)
    plt.show()


if False:
    x_vals = np.linspace(0.0, 4.0, 1000)
    plt.plot(x_vals, sigmoid(x_vals, a=3.0, e0=2.0), label="sigmoid")
    plt.plot(x_vals, epsilon(x_vals, a=0.6, e0=0.9), label="sigmoid")
    plt.plot(data["energy"], data["successes"] / 100.0, ".", label="data")
    plt.legend()
    plt.title("Best fits")
    plt.show()
