import torch
import numpy as np

class IsingEnvironment():
    def __init__(self, initial_lattice: torch.Tensor, max_steps: int):
        self.initial_lattice = initial_lattice
        self.max_steps = max_steps

        self.h = initial_lattice.shape[0]
        self.w = initial_lattice.shape[1]
        self.kb = 1
        self.temp = 1 # high is 3
        self.beta = 1 / (self.kb * self.temp)
        self.J = 1

        self.reset()

    def reset(self):
        self.state = torch.cat((self.initial_lattice.flatten(), torch.tensor([0.0])))
        return self.state

    def step(self, action):
        new_state = self.state.clone()
        new_state[action] = 1 - new_state[action]  # flip the spin
        new_state[-1] += 1  # increment the time step
        self.state = new_state
        done = self.state[-1] == self.max_steps
        reward = self.reward_fn(self.state[:-1].reshape(self.initial_lattice.shape)) if done else torch.tensor(0.0)
        return self.state, reward, done
    
    def reverse_step(self, action):
        new_state = self.state.clone()
        new_state[action] = 1 - new_state[action]  # flip the spin
        new_state[-1] -= 1  # decrement the time step
        self.state = new_state
        done = self.state[-1] == 0
        reward = self.reward_fn(self.state[:-1].reshape(self.initial_lattice.shape)) if done else torch.tensor(0.0)
        return self.state, reward, done

    def reward_fn(self, lattice):
        """
        Computes the potential energy of a given spin state in the Ising model.
        The reward is proportional to exp(-energy).

        Parameters:
        - state: A 2D torch tensor of shape (h, w), where each element is +1 or -1.

        Returns:
        - reward: The potential reward associated with the given spin configuration.
        """
        lattice = 2 * lattice - 1 # convert 1 and 0 to 1 and -1
        # Right neighbor interaction (with periodic boundary)
        right_neighbors = torch.roll(lattice, shifts=-1, dims=1)
        
        # Bottom neighbor interaction (with periodic boundary)
        bottom_neighbors = torch.roll(lattice, shifts=-1, dims=0)
        
        # Calculate total interaction energy
        interaction_energy = -self.J * lattice * (right_neighbors + bottom_neighbors)
        
        # Sum the total energy over all sites
        total_energy = torch.sum(interaction_energy)
        
        # Reward is proportional to exp(-total_energy)
        reward = torch.exp(-self.beta * total_energy)
        return reward