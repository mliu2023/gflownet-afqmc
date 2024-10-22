import torch

class IsingEnvironmentVoid():
    def __init__(self, initial_lattice: torch.Tensor, max_steps: int, temp: float, batch_size: int):
        self.initial_lattice = initial_lattice.float()  # Ensure float type
        self.max_steps = max_steps
        self.batch_size = batch_size

        self.h = initial_lattice.shape[0]
        self.w = initial_lattice.shape[1]
        self.kb = 1
        self.temp = temp
        self.beta = 1 / (self.kb * self.temp)
        self.J = 1

        self.reset()

    def reset(self):
        self.state = torch.cat((
            self.initial_lattice.flatten().repeat(self.batch_size, 1),
            torch.zeros(self.batch_size, 1)
        ), dim=1).float()  # Ensure float type
        return self.state

    def step(self, action):
        new_state = self.state.clone()
        batch_indices = torch.arange(self.batch_size)
        site_indices = action % (self.h * self.w)
        spin_values = (2 * (action // (self.h * self.w)) - 1).float()  # Convert to float
        new_state[batch_indices, site_indices] = spin_values
        new_state[:, -1] += 1  # increment the time step
        self.state = new_state
        done = (self.state[:, -1] == self.max_steps)
        log_reward = torch.where(
            done,
            self.reward_fn(self.state[:, :-1].reshape(self.batch_size, self.h, self.w)),
            torch.zeros(self.batch_size)
        )
        return self.state, log_reward, done
    
    def reverse_step(self, action):
        new_state = self.state.clone()
        batch_indices = torch.arange(self.batch_size)
        new_state[batch_indices, action] = 0.0  # set the spin to 0 (as float)
        new_state[:, -1] -= 1  # decrement the time step
        self.state = new_state
        done = (self.state[:, -1] == 0)
        reward = torch.where(
            done,
            self.reward_fn(self.state[:, :-1].reshape(self.batch_size, self.h, self.w)),
            torch.zeros(self.batch_size)
        )
        return self.state, reward, done

    def reward_fn(self, lattice):
        right_neighbors = torch.roll(lattice, shifts=-1, dims=2)
        bottom_neighbors = torch.roll(lattice, shifts=-1, dims=1)
        interaction_energy = -self.J * lattice * (right_neighbors + bottom_neighbors)
        total_energy = torch.sum(interaction_energy, dim=(1, 2))
        log_reward = -self.beta * total_energy
        return log_reward