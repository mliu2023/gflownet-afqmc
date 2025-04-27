import torch
import torch.nn as nn
import numpy as np
import os

from af.gflownet_af import GFNAgentAF
from environments.af_env import AFEnvironment
from utils.visualize import (visualize_terminal_states, 
                             visualize_reward_distribution,
                             visualize_parity_plot)
class EnergyModel(nn.Module):
    def __init__(self, n_fields, hidden_size):
        super().__init__()
        self.n_fields = n_fields
        # self.J = nn.Parameter(torch.normal(0, 1, (n_fields, n_fields)))
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
        # return -torch.einsum('bi,ij,bj->b', x, self.J, x)
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
        return self.fwp(state), torch.ones(state.shape[0], 1)
        # return self.fwp(state), self.bwp(state)

class GFNAgentAFVoid(GFNAgentAF):
    def get_environment(self, n_fields, trajectory_len, model, batch_size):
        return AFEnvironment(n_fields, trajectory_len, model, batch_size)

    def get_gflownet(self, n_fields, hidden_size):
        return GFlowNetAFVoid(n_fields, hidden_size)
    
    def get_energy_model(self, n_fields, hidden_size):
        # return nn.Sequential(
        #     nn.Linear(n_fields, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1),
        #     # nn.ELU(),
        # )
        return EnergyModel(n_fields, hidden_size)

    def create_forward_actions_mask(self, state: torch.Tensor):
        mask = (state[:, :-1] == 0).float()
        return torch.cat((mask, mask), 1)

    def create_backwards_actions_mask(self, state: torch.Tensor):
        mask = (state[:, :-1] != 0).float()
        return mask

if __name__ == "__main__":
    torch.manual_seed(1)

    height = 7
    width = 7
    n_fields = 49
    fields = torch.from_numpy(np.load("data/af_7x7_24up_24down_fields.npy"))
    fields = 2 * fields - 1
    fields = fields.to(torch.float32)
    energies = torch.from_numpy(np.load("data/af_7x7_24up_24down_energies.npy"))
    energies = energies.to(torch.float32)

    device = torch.device("cpu")

    agent = GFNAgentAFVoid(n_fields=n_fields,
                           trajectory_len=n_fields,
                           nx=7,
                           ny=7,
                           hidden_size=256, 
                           batch_size=256, 
                           env_folder="af_void",
                           device=device,
                           fields=fields,
                           energies=energies)
    os.makedirs(agent.output_folder, exist_ok=True)
    agent.train_gflownet(iterations=2000) # 500 iterations is good

    log_rewards = []
    for i in range(20):
        trajectory, forward_probs, backward_probs, actions, log_reward = agent.forward_sample_trajectory(eps=0, K=agent.trajectory_len)
        # sampled_fields = [state[:, :-1].reshape((agent.batch_size, height, width)) for state in trajectory]
        # visualize_trajectory(
        #     trajectory=[lattice[0] for lattice in lattices],  # Visualize only the first batch item
        #     filename=os.path.join(agent.output_folder, f"trajectory_{i}.gif"),
        #     reward=reward[0].item()
        # )
        # visualize_terminal_state(
        #     lattice=fields[-1][0],  # Visualize the terminal state of the first batch item
        #     filename=os.path.join(agent.output_folder, f"trajectory_{i}.png")
        # )
        log_rewards.append(log_reward)
    log_rewards = torch.stack(log_rewards).detach().flatten().numpy()
    visualize_reward_distribution(-log_rewards, os.path.join(agent.output_folder, "energies.png"))
    
    trajectory, forward_probs, backward_probs, actions, log_reward = agent.forward_sample_trajectory(eps=0, K=agent.trajectory_len)
    sampled_fields = [state[:, :-1].reshape((agent.batch_size, height, width)) for state in trajectory]
    visualize_terminal_states(
        lattices=sampled_fields[-1],  # Visualize the terminal states
        filename=os.path.join(agent.output_folder, f"fields.png"),
        cols=16
    )
    energies_pred = []
    for field in fields:
        energies_pred.append(agent.energy_model(field))
    energies_pred = torch.stack(energies_pred).detach().numpy()
    visualize_parity_plot(energies, energies_pred, os.path.join(agent.output_folder, f"parity_plot.png"))
    print("Done")