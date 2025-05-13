import torch
import torch.nn as nn
import numpy as np
import os

from gflownet_afqmc import GFNAgentAF
from environments.afqmc_env import AFEnvironment
from utils.visualize import (visualize_terminal_states, 
                             visualize_distribution,
                             visualize_parity_plot)

class EnergyModel(nn.Module):
    def __init__(self, n_fields, hidden_size):
        super().__init__()
        self.n_fields = n_fields
        input_size = n_fields
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.network(x)

class GFlowNetAFVoid(nn.Module):
    def __init__(self, n_fields, hidden_size):
        super().__init__()
        input_size = n_fields
        forward_output_size = 2 * n_fields
        backward_output_size = n_fields
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.fwp = nn.Linear(hidden_size, forward_output_size)
        self.bwp = nn.Linear(hidden_size, backward_output_size)
        self.log_Z = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        state = state[:, :-1]
        state = self.network(state)
        return self.fwp(state), torch.ones(state.shape[0], 1) # uniform backward policy
        # return self.fwp(state), self.bwp(state)

class GFNAgentAFVoid(GFNAgentAF):
    def get_environment(self, n_fields, trajectory_len, model, batch_size):
        return AFEnvironment(n_fields, trajectory_len, model, batch_size)

    def get_gflownet(self, n_fields, hidden_size):
        return GFlowNetAFVoid(n_fields, hidden_size)
    
    def get_energy_model(self, n_fields, hidden_size):
        return EnergyModel(n_fields, hidden_size)

    def create_forward_actions_mask(self, state: torch.Tensor):
        mask = (state[:, :-1] == 0).float()
        return torch.cat((mask, mask), 1)

    def create_backwards_actions_mask(self, state: torch.Tensor):
        mask = (state[:, :-1] != 0).float()
        return mask

if __name__ == "__main__":
    torch.manual_seed(1)

    height = 5
    width = 5
    n_fields = 25
    fields = torch.from_numpy(np.load("data/af_5x5_12up_12down_fields.npy"))
    fields = 2 * fields - 1
    fields = fields.to(torch.float32)
    energies = torch.from_numpy(np.load("data/af_5x5_12up_12down_energies.npy"))
    energies = energies.to(torch.float32)

    device = torch.device("cpu")

    agent = GFNAgentAFVoid(n_fields=n_fields,
                           trajectory_len=n_fields,
                           nx=5,
                           ny=5,
                           hidden_size=256, 
                           batch_size=256, 
                           env_folder="afqmc_void",
                           device=device,
                           fields=fields,
                           energies=energies)
    os.makedirs(agent.output_folder, exist_ok=True)
    agent.train_gflownet(iterations=1000, warmup_k=1000)

    log_rewards = []
    for i in range(20):
        trajectory, forward_probs, backward_probs, actions, log_reward = agent.forward_sample_trajectory(eps=0, K=agent.trajectory_len)
        log_rewards.append(log_reward)
    log_rewards = torch.stack(log_rewards).detach().flatten().numpy()
    visualize_distribution(-log_rewards, os.path.join(agent.output_folder, "energies.png"))
    
    trajectory, forward_probs, backward_probs, actions, log_reward = agent.forward_sample_trajectory(eps=0, K=agent.trajectory_len)
    sampled_fields = [state[:, :-1].reshape((agent.batch_size, height, width)) for state in trajectory]
    visualize_terminal_states(
        lattices=sampled_fields[-1],
        filename=os.path.join(agent.output_folder, f"fields.png"),
        cols=16
    )
    energies_pred = []
    for field in fields:
        energies_pred.append(agent.energy_model(field))
    energies_pred = torch.stack(energies_pred).flatten().detach().numpy()
    visualize_parity_plot(energies, energies_pred, os.path.join(agent.output_folder, f"parity_plot.png"))
    print("Done")