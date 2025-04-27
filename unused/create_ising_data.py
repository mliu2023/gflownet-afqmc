import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from environments.ising_env import IsingEnvironment
from utils.visualize import visualize_terminal_states

if __name__ == "__main__":
    width = 7
    height = 7
    batch_size = 1
    initial_lattice = 2 * torch.bernoulli(torch.full((batch_size, height, width), 0.5)) - 1
    temp = .1
    env = IsingEnvironment(initial_lattice, width*height, temp=temp, batch_size=batch_size)
    env.J = -3

    configurations = []
    state = env.reset()
    log_reward = None
    done = False

    # mcmc steps
    burn = 1000000
    steps = 1000000
    start_time = time.time()
    for i in range(burn + steps):
        if done:
            state = env.reset()
            log_reward = None
            done = False
        state, done = env.mcmc_step()
        if i >= burn and i % 10 == 0:
            log_reward = env.reward_fn(env.state[:, :-1].reshape(env.batch_size, env.h, env.w))
            configurations.append(torch.concat([state[:, :-1], -log_reward.unsqueeze(1)], dim=1))

    # save data
    os.makedirs(f"data/temp_{temp}", exist_ok=True)
    configurations = torch.stack(configurations).reshape(-1, width*height+1).numpy()
    np.save(f"data/temp_{temp}/ising_{env.J}.npy", configurations)
    fields = configurations[:, :-1]
    energies = configurations[:, -1]
    plt.hist(energies, bins=18)
    plt.savefig(f"data/temp_{temp}/energies_{env.J}.png")
    plt.close()
    visualize_terminal_states(fields.reshape(-1, 7, 7)[np.arange(0, len(fields), 2000)], filename=f"data/temp_{temp}/fields_{env.J}.png", cols=16)

    end_time = time.time()
    print(end_time-start_time)