import torch

class AFEnvironment():
    def __init__(self, 
                 n_fields: int,
                 max_steps: int, 
                 model: torch.nn.Module, 
                 batch_size: int):
        self.n_fields = n_fields
        self.initial_lattice = torch.zeros(n_fields)
        self.max_steps = max_steps
        self.model = model
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.state = torch.cat((
            self.initial_lattice.repeat(self.batch_size, 1),
            torch.zeros(self.batch_size, 1)
        ), dim=1).float()
        return self.state

    def step(self, action):
        new_state = self.state.clone()
        batch_indices = torch.arange(self.batch_size)
        site_indices = action % self.n_fields
        spin_values = (2 * (action // self.n_fields) - 1).float()
        new_state[batch_indices, site_indices] = spin_values
        new_state[:, -1] += 1  # increment the time step
        self.state = new_state
        done = (self.state[0, -1] == self.max_steps)
        if done:
            log_reward = self.reward_fn(self.state[:, :-1].reshape(self.batch_size, self.n_fields))
        else:
            log_reward = torch.zeros(self.batch_size)
        return self.state, log_reward, done
    
    def reverse_step(self, action):
        new_state = self.state.clone()
        batch_indices = torch.arange(self.batch_size)
        new_state[batch_indices, action] = 0.0  # set the spin to 0 (as float)
        new_state[:, -1] -= 1  # decrement the time step
        self.state = new_state
        done = (self.state[0, -1] == 0)
        if done:
            log_reward = self.reward_fn(self.state[:, :-1].reshape(self.batch_size, self.n_fields))
        else:
            log_reward = torch.zeros(self.batch_size)
        return self.state, log_reward, done

    def energy_fn(self, lattice):
        return self.model(lattice)
    
    def reward_fn(self, lattice):
        return -self.model(lattice)