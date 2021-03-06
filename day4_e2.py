"""Daily assignment 4_2"""

import numpy as np
import matplotlib.pyplot as plt

n_sim = 10000
x_generated = []

for i in range(n_sim):
    u = np.random.uniform()
    x = 1.0 / (100.0 * (1.0 - u))
    x_generated.append(x)

x_generated = np.asarray(x_generated)

plt.hist(x_generated, bins=1000, range=[0.0, 4.0], label="gen. RN", density=True)
x_vals = np.linspace(0.01, 4, 1000)
y_vals = 0.01 / (x_vals) ** 2
plt.plot(x_vals, y_vals, label="P(y) = 0.01/y**2")
plt.yscale("log")
plt.title("Inverse transform result")
plt.ylabel("prob p")
plt.xlabel("y [ ]")
plt.legend()
# plt.show()

plt.savefig(f"output/day4_e2_{n_sim}.png", dpi=300)
