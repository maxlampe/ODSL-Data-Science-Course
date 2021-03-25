"""Script for solving monte carlo assignment problem 9"""

import numpy as np
import matplotlib.pyplot as plt


class M9Map:
    """Problem 9 travelling salesman map with enhanced features."""

    def __init__(self, n_towns: int, sim_range: list = [0.0, 1.0]):
        self.n_towns = n_towns
        self.sim_range = sim_range

        self.map = self._gen_map()
        self.order = np.linspace(0, self.n_towns - 1, self.n_towns, dtype=int)
        self.dist_mat = np.ones([self.n_towns, self.n_towns])
        self._calc_dist_mat()
        self.curr_dist = self._calc_total_dist(self.order)

        self.cache = []

    def _gen_map(self):
        """Generate a number of cities with coordinates"""
        return (
            np.random.rand(self.n_towns, 2) * (self.sim_range[1] - self.sim_range[0])
            + self.sim_range[0]
        )

    def plot_map(self, bsave_fig: bool = False):
        """Plot map of towns"""

        plot_map = self.map[self.order].T

        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        fig.suptitle("Traveling salesman problem")

        axs.plot(plot_map[0], plot_map[1], label="Path")
        axs.plot(
            [plot_map[0][-1], plot_map[0][0]],
            [plot_map[1][-1], plot_map[1][0]],
            label="Final step",
        )
        axs.plot(plot_map[0], plot_map[1], ".", label="Towns")
        axs.set_xlabel("x-coord. [ ]")
        axs.set_ylabel("y-coord. [ ]")

        delta_range = self.sim_range[1] - self.sim_range[0]
        axs.set_xlim(
            0.9 * self.sim_range[0] - 0.1 * delta_range,
            1.1 * self.sim_range[1] + 0.1 * delta_range,
        )
        axs.set_ylim(
            0.9 * self.sim_range[0] - 0.1 * delta_range,
            1.1 * self.sim_range[1] + 0.1 * delta_range,
        )

        i = 0
        for xy in zip(plot_map[0], plot_map[1]):
            axs.annotate(f"T{i}", xy=xy, textcoords="data")
            i += 1

        plt.legend()
        if bsave_fig:
            plt.savefig("output/problem_9_map.png", dpi=300)
        plt.show()

    def _get_distances(self, town_i: int, town_j: int):
        """Calculate distance between two towns"""

        t1 = self.map[town_i]
        t2 = self.map[town_j]

        return np.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2)

    def _calc_dist_mat(self):
        """Calculate distance matrix between all towns as LUT"""
        for i in range(self.n_towns):
            for j in range(self.n_towns):
                self.dist_mat[i, j] = self._get_distances(i, j)

    def _calc_total_dist(self, order: np.array):
        """Calculate distance for current order"""

        total_dist = 0
        n = order.shape[0]
        for i in range(order.shape[0]):
            total_dist += self.dist_mat[order[i % n], order[(i + 1) % n]]

        return total_dist

    def _do_step(self, temp: float):
        """Do one step of the MCMC algorithm"""

        tmp_order = self.order
        u = np.random.randint(0, self.n_towns)
        v1 = u % self.n_towns
        v2 = (u + 1) % self.n_towns

        tmp_order[v1], tmp_order[v2] = tmp_order[v2], tmp_order[v1]
        tmp_dist = self._calc_total_dist(tmp_order)

        p = np.random.uniform()
        delta_d = tmp_dist - self.curr_dist
        if p < min(1.0, np.exp(-delta_d / temp) / temp):
            self.order = tmp_order
            self.curr_dist = tmp_dist

        self.cache.append(np.asarray([self.curr_dist, temp]))

    def do_mcmc(self, n_sim: int = 1000000, temp: float = 10000.0, bplot_sim: bool = True):
        """Run the Markov-Chain-Monte-Carlo"""

        # Reset cache
        self.cache = []
        t = temp

        for n in range(n_sim):
            self._do_step(t)
            if n % 50000 == 0 and n > 0:
                t *= 0.1

        self.cache = np.asarray(self.cache)

        if bplot_sim:
            fig, axs = plt.subplots(2, 1, figsize=(9, 14), sharex=True)
            fig.suptitle("Simulation run")
            axs[0].plot(self.cache.T[0])
            axs[1].plot(self.cache.T[1])
            axs[1].set_yscale("log")
            axs[1].set_xlabel("Sim steps [ ]")
            axs[0].set_ylabel("Total distance [ ]")
            axs[1].set_ylabel("Temp [ ]")
            fig.subplots_adjust(hspace=0.01)
            plt.tight_layout()
            plt.show()


map_class = M9Map(n_towns=19)

print(map_class.curr_dist)
map_class.do_mcmc()
print(map_class.order, "\n", map_class.curr_dist)
map_class.plot_map()
