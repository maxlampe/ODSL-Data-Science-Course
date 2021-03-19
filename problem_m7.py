"""Script for solving monte carlo assignment problem 7"""

import numpy as np
import matplotlib.pyplot as plt
from tools import M6Hist


def do_ehrenfest(
    n_atoms: int = 1000,
    n_iter: int = 100000,
    r_barrier: float = 1.0,
    bplot: bool = True,
    bsave_plot: bool = False,
):
    """Do simulation with Ehrenfest model"""

    u_all = np.random.rand(n_iter)
    i = 0
    a = 0.5 * n_atoms

    prog = [i]
    for u in u_all:
        if u <= r_barrier * (a - i) / n_atoms:
            i += 1
        elif u <= r_barrier * (a - i) / n_atoms + (a + i) / n_atoms:
            i -= 1
        else:
            i = i
        prog.append(i)
    prog = np.asarray(prog)

    # re-using histogram class from last problem
    hist_class = M6Hist(data=prog, bincount=100, range_uplim=a, range_lowlim=-a)
    # find mode
    i_star = hist_class.hist["x"][hist_class.hist["y"].argmax()]
    left_star = i_star + a
    right_star = a - i_star

    if bsave_plot or bplot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Ehrenfest (n_atoms = {n_atoms}, n_iter = {n_iter}, r = {r_barrier})")
        axs.flat[0].plot(prog)
        axs.flat[1].hist(prog, bins=100, range=(-2* a, 2* a))
        axs.flat[0].set_title("Equilibrium i progression")
        axs.flat[0].set_xlabel("iter [ ]")
        axs.flat[0].set_ylabel("i [ ]")
        axs.flat[1].set_title("Histogram of equilibrium i")
        axs.flat[1].set_xlabel("i [ ]")
        axs.flat[1].set_ylabel("Counts [ ]")
        axs.flat[1].annotate(
            f"Most likely final dist.\n left: {left_star}\n" f"right: {right_star}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", fc="1"),
        )
        if bsave_plot:
            plt.savefig(f"output/problem_m7_r{r_barrier}.png", dpi=300)
        if bplot:
            plt.show()

    return [i_star, left_star, right_star]


do_ehrenfest(r_barrier=1, bsave_plot=True)
do_ehrenfest(r_barrier=1.3333, bsave_plot=True)

r_samples = np.logspace(-2, 2, 100)
i_sample = np.asarray([do_ehrenfest(r_barrier=r, bplot=False)[0] for r in r_samples])

plt.plot(r_samples, i_sample)
plt.xscale("log")
plt.xlabel("r_barrier [ ]")
plt.ylabel("i_final [ ]")
plt.title("Final equilibrium for different r_barrier (n_atoms = 1000)")
plt.savefig("output/problem_m7_rdep.png", dpi=300)
plt.show()
