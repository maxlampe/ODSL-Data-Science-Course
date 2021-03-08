""""""

import numpy as np
import matplotlib.pyplot as plt

n_sim = 30000
x_generated = []

for i in range(n_sim):
    u = np.random.uniform()
    x = 1./(100. * (1.-u))
    x_generated.append(x)

x_generated = np.asarray(x_generated)

plt.hist(x_generated, bins=1000, range=[0., 4.], label="gen. RN")
x_vals = np.linspace(0.01, 4, 1000)
y_vals = 1./(x_vals)**2
plt.plot(x_vals, y_vals, label="P(y)")
plt.yscale("log")
plt.title("Box-Mueller result")
plt.ylabel("a.u.")
plt.xlabel("y [ ]")
plt.legend()
# plt.show()

plt.savefig(f"output/day4_e2.png", dpi=300)
