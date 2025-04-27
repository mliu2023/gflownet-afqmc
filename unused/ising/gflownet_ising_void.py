import torch
import torch.nn as nn
import os
from tqdm import tqdm

from ising.gflownet_ising import GFNAgentIsing
from environments.ising_void_env import IsingEnvironmentVoid
from utils.visualize import visualize_terminal_state, visualize_terminal_states, visualize_reward_distribution

class GFlowNetIsingVoid(nn.Module):
    def __init__(self, initial_lattice, hidden_size):
        super().__init__()
        input_size = initial_lattice.shape[0] * initial_lattice.shape[1]
        forward_output_size = 2 * initial_lattice.shape[0] * initial_lattice.shape[1]
        backward_output_size = initial_lattice.shape[0] * initial_lattice.shape[1]
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU()
        )
        self.fwp = nn.Linear(hidden_size, forward_output_size)
        self.bwp = nn.Linear(hidden_size, backward_output_size)
        self.log_Z = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        state = state[:, :-1]
        state = self.network(state)
        return self.fwp(state), torch.ones(state.shape[0], 1)

class GFNAgentIsingVoid(GFNAgentIsing):
    def get_environment(self, initial_lattice, trajectory_len, temp, batch_size):
        return IsingEnvironmentVoid(initial_lattice, trajectory_len, temp, batch_size)

    def get_gflownet(self, initial_lattice, hidden_size):
        return GFlowNetIsingVoid(initial_lattice, hidden_size)

    def create_forward_actions_mask(self, state: torch.Tensor):
        mask = (state[:, :-1] == 0).float()
        return torch.cat((mask, mask), 1)

    def create_backwards_actions_mask(self, state: torch.Tensor):
        mask = (state[:, :-1] != 0).float()
        return mask

if __name__ == "__main__":
    height = 7
    width = 7
    initial_lattice = torch.zeros((height, width))

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")

    for temp in [0.1]:
        agent = GFNAgentIsingVoid(initial_lattice=initial_lattice, 
                                  trajectory_len=height*width, 
                                  hidden_size=256, 
                                  temp=temp, 
                                  batch_size=256, 
                                  replay_batch_size=256, 
                                  buffer_size=100_000,
                                  alpha=0.5,
                                  beta=0.1,
                                  env_folder="ising_void_2",
                                  device=device)
        os.makedirs(agent.output_folder, exist_ok=True)
        agent.train_gflownet(iterations=400, steps_per_iteration=8, resamples_per_iteration=1)

        rewards = []
        for i in range(20):
            trajectory, forward_probs, backward_probs, actions, reward = agent.sample_trajectory(eps=0)
            lattices = [state[:, :-1].reshape((agent.batch_size, agent.height, agent.width)) for state in trajectory]
            # visualize_trajectory(
            #     trajectory=[lattice[0] for lattice in lattices],  # Visualize only the first batch item
            #     filename=os.path.join(agent.output_folder, f"trajectory_{i}.gif"),
            #     reward=reward[0].item()
            # )
            # visualize_terminal_state(
            #     lattice=lattices[-1][0],  # Visualize the terminal state of the first batch item
            #     filename=os.path.join(agent.output_folder, f"trajectory_{i}.png")
            # )
            rewards.append(reward)
        rewards = torch.stack(rewards).flatten().numpy()
        visualize_reward_distribution(rewards, os.path.join(agent.output_folder, "rewards.png"))

        trajectory, forward_probs, backward_probs, actions, log_reward = agent.sample_trajectory(eps=0)
        lattices = [state[:, :-1].reshape((agent.batch_size, 7, 7)) for state in trajectory]
        visualize_terminal_states(
            lattices=lattices[-1],  # Visualize the terminal states
            filename=os.path.join(agent.output_folder, f"lattices.png"),
            cols=16
        )
        print("Done")