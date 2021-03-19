"""Script for solving monte carlo assignment problem 6"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate


class M6Hist:
    """Histogram class with enhanced functions for this problem"""

    def __init__(
        self,
        data: np.array(float),
        bincount: int = 1024,
        range_lowlim: int = 0,
        range_uplim: int = 52000,
        weights: np.array(float) = None,
    ):
        self._data = data
        self._histpar = {
            "binc": bincount,
            "lowlim": range_lowlim,
            "uplim": range_uplim,
            "weights": weights,
            "binw": (range_uplim - range_lowlim) / bincount,
        }
        self._binedge = None
        self.hist = self._ret_hist()
        self.exp_val = None

    def _ret_hist(self):
        """Create a histogram as pd.DataFrame from an input array."""

        raw_bins = np.linspace(
            self._histpar["lowlim"], self._histpar["uplim"], self._histpar["binc"] + 1
        )
        use_bins = [np.array([-np.inf]), raw_bins, np.array([np.inf])]
        use_bins = np.concatenate(use_bins)

        hist, binedge = np.histogram(
            self._data, bins=use_bins, weights=self._histpar["weights"], density=True
        )

        bincent = []
        for j in range(binedge.size - 1):
            bincent.append(0.5 * (binedge[j] + binedge[j + 1]))

        hist = hist[1:-1]
        bincent = bincent[1:-1]
        self._binedge = binedge[1:-1]

        return pd.DataFrame({"x": bincent, "y": hist})

    def plot_hist(
        self,
        rng: list = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Plot histogram."""

        plt.figure(figsize=(8, 6))
        plt.bar(x=self.hist["x"], height=self.hist["y"], width=self._histpar["binw"])
        if rng is not None:
            plt.axis([rng[0], rng[1], rng[2], rng[3]])

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        return 0

    def _calc_expected(self, t_func, t_par: dict):
        """Calc expected values for target function and bin edges"""

        # FIXME: Wrong values? Why?
        exp_val = np.array([])
        for i in range(self._binedge.shape[0] - 1):
            curr_int = integrate.quad(
                lambda x: t_func(x=x, **t_par), self._binedge[i], self._binedge[i + 1]
            )[0]
            exp_val = np.append(exp_val, curr_int)

        self.exp_val = exp_val

        return 0

    def calc_deviations(self, t_func, t_par: dict):
        """Calculate variance of current sample"""

        # FIXME: Why not integrate over bin?
        # if self.exp_val is None:
        #    self._calc_expected(t_func=t_func, t_par=t_par)

        return (self.hist["y"] - t_func(self.hist["x"], **t_par)) ** 2


def target_dist(x: float, par_r: float):
    """Target distribution for rejection sampling"""

    fac = 2.0 / (np.pi * par_r ** 2)
    return fac * np.sqrt(par_r ** 2 - x ** 2)


def gen_sample(n_sample_size: int = 1000, par_r: float = 1.0):
    """Generate one sample of the target distribution for a given sample size."""

    samples = []
    weights = []
    for i in range(n_sample_size):
        u = np.random.uniform(-par_r, par_r)
        weight = 2 * par_r * target_dist(x=u, par_r=par_r)
        samples.append(u)
        weights.append(weight)
    weights = np.asarray(weights)
    samples = np.asarray(samples)

    return samples, weights


def sim_variance(
    n_trials: int = 20, n_sample_size: int = 100, par_r: float = 1.0, bplot_last=False
):
    """Numerically obtain variance of generated sample from target density"""

    var = np.asarray([])
    for i in range(n_trials):
        samp, weig = gen_sample(
            n_sample_size=n_sample_size,
            par_r=par_r,
        )

        hist_class = M6Hist(
            data=samp,
            weights=weig,
            bincount=int(np.sqrt(n_sample_size) * 0.5),
            range_lowlim=-par_r,
            range_uplim=par_r,
        )

        deviations = hist_class.calc_deviations(target_dist, {"par_r": par_r})
        var = np.append(var, deviations.mean())

        if bplot_last and i == n_trials - 1:
            hist_class.plot_hist()
            x_vals = np.linspace(-par_r, par_r, 1000)
            plt.plot(
                x_vals,
                target_dist(x=x_vals, par_r=par_r),
                label="target dist.",
                color="r",
            )
            plt.legend()
            plt.title("Generated sample")
            plt.xlabel("x [ ]")
            plt.ylabel("a.u. [ ]")
            plt.savefig(f"output/problem_m6_gen{n_sample_size}.png", dpi=300)
            plt.show()

    return var.mean()


sim_variance(n_trials=1, n_sample_size=100, bplot_last=True)
sim_variance(n_trials=1, n_sample_size=1000, bplot_last=True)
sim_variance(n_trials=1, n_sample_size=10000, bplot_last=True)


n_vals = np.concatenate([np.linspace(10, 1000, 30, dtype=int), np.linspace(1000, 100000, 10, dtype=int)])
n_vals = np.linspace(10, 100000, 30, dtype=int)
variances = np.asarray([sim_variance(n_sample_size=n) for n in n_vals])
n_plot = np.linspace(10, 100000, 1000)

plt.plot(n_vals, variances, label="sim. var")
plt.plot(n_plot, 0.14/np.sqrt(n_plot), label="calc. var")
plt.yscale("log")
plt.legend()
plt.title("Numerical and analytical variance")
plt.xlabel("x [ ]")
plt.ylabel("a.u. [ ]")
plt.savefig("output/problem_m6_var.png", dpi=300)
plt.show()

