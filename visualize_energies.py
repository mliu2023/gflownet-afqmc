import numpy as np
import matplotlib.pyplot as plt
from utils.visualize import visualize_terminal_states

fields = np.load("data/af_7x7_24up_24down_fields.npy")
fields = 2 * fields - 1
energies = np.load("data/af_7x7_24up_24down_energies.npy")
visualize_terminal_states(fields.reshape(-1, 7, 7)[:256], "data/af_7x7_24up_24down_fields.png", 16)
plt.hist(energies, bins=20)
plt.savefig("data/af_7x7_24up_24down_energies.png")
plt.close()