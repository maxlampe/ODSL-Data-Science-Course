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
        self.curr_dist = self._calc_total_dist(self.order)

        self.cache = []
        self.lazy = 0

    def _gen_map(self):
        """Generate a number of cities with coordinates"""
        return (
            np.random.rand(self.n_towns, 2) * (self.sim_range[1] - self.sim_range[0])
            + self.sim_range[0]
        )

    def plot_map(self, bsave_fig: bool = False):
        """Plot map of towns"""

        plot_map = self.map[self.order].T
        # plot_map = np.append(plot_map, self.map[self.order[0]]).T
        plt.plot(plot_map[0], plot_map[1], label="Path")
        plt.plot(plot_map[0], plot_map[1], ".", label="T")
        i = 0
        for xy in zip(plot_map[0], plot_map[1]):
            plt.annotate(f"T{i}", xy=xy, textcoords="data")
            i += 1

        plt.xlabel("x-coord. [ ]")
        plt.ylabel("y-coord. [ ]")
        plt.title("Traveling salesman problem")
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.legend()
        if bsave_fig:
            plt.savefig("output/problem_9_map.png", dpi=300)
        plt.show()

    def _get_distances(self, town_i: int, town_j: int):
        """Calculate distance between two towns"""

        t1 = self.map[town_i]
        t2 = self.map[town_j]

        return np.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2)

    def _calc_total_dist(self, order: np.array):
        """Calculate distance for current order"""

        total_dist = 0
        for i in range(order.shape[0] - 1):
            total_dist += self._get_distances(order[i], order[i + 1])
        total_dist += self._get_distances(order[-1], order[0])

        return total_dist

    def _do_iteration(self, temp: float):
        """Do one iteration of the MCMC algorithm"""

        inter_order = self.order
        u = round(np.random.uniform(0, self.n_towns - 1))

        if u < self.n_towns - 1:
            inter_val = inter_order[u]
            inter_order[u] = inter_order[u + 1]
            inter_order[u + 1] = inter_val
        else:
            inter_val = inter_order[u]
            inter_order[u] = inter_order[0]
            inter_order[0] = inter_val

        inter_dist = self._calc_total_dist(inter_order)
        if inter_dist < self.curr_dist:
            self.order = inter_order
            self.curr_dist = inter_dist
        else:
            p = np.random.uniform()
            if p < (np.exp(-inter_dist / temp)):
                self.order = inter_order
                self.curr_dist = inter_dist
            else:
                self.lazy += 1

        self.cache.append(np.asarray([self.curr_dist, temp]))

    def do_MCMC(self, n_sim: int = 100000, temp: float = 10.0):
        """Run the Markov-Chain-Monte-Carlo"""

        # Reset cache
        self.cache = []
        self.lazy = 0
        t = temp
        count_same = 0
        count_n = 0
        last_dist = self.curr_dist
        for n in range(n_sim):
            # if (n % 10000) == 0 and n > 0:
            #     t *= 0.5
            self._do_iteration(t)

            if np.abs(last_dist - self.curr_dist) < 0.0001:
                count_same += 1
            if count_same > 10000:
                t *= 0.5
                count_n = 0
                count_same = 0
            else:
                count_n += 1
            if count_n > 15000:
                t *= 0.5
                count_n = 0
                count_same = 0

            last_dist = self.curr_dist

        self.cache = np.asarray(self.cache)
        print(self.lazy, n_sim)

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))

        axs[0].plot(self.cache.T[0])
        axs[1].plot(self.cache.T[1])
        axs[1].set_yscale("log")
        plt.show()


map_class = M9Map(n_towns=25)

print(map_class.curr_dist)
map_class.do_MCMC()
print(map_class.order, map_class.curr_dist)
map_class.plot_map()
