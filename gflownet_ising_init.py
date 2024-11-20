import torch
import torch.nn as nn
from gflownet_utils import GFNAgent
from env import IsingEnvironment

from matplotlib import pyplot as plt
import os

from visualize import visualize_trajectory, visualize_terminal_state

class GFlowNet(nn.Module):
    def __init__(self, initial_lattice, hidden_size):
        super(GFlowNet, self).__init__()
        input_size = initial_lattice.shape[0] * initial_lattice.shape[1]
        output_size = initial_lattice.shape[0] * initial_lattice.shape[1]
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
        )
        self.fwp = nn.Linear(hidden_size, output_size)
        self.bwp = nn.Linear(hidden_size, output_size)
        self.log_Z = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        state = state[:, :-1]
        state = self.network(state)
        # return self.fwp(state), self.bwp(state)
        return self.fwp(state), 1
    
class GFNAgentInit(GFNAgent):
    def get_env_name(self):
        return "init"

    def get_environment(self, initial_lattice, trajectory_len, temp, batch_size):
        return IsingEnvironment(initial_lattice, trajectory_len, temp, batch_size)
    
    def get_gflownet(self, initial_lattice, hidden_size):
        return GFlowNet(initial_lattice, hidden_size)
    
    def create_forward_actions_mask(self, state: torch.Tensor):
        batch_size = state.shape[0]
        num_actions = state.shape[1] - 1
        return torch.ones(batch_size, num_actions)
    
    # def create_backwards_actions_mask(self, state: torch.Tensor):
    #     diff = self.initial_lattice.reshape(self.initial_lattice.shape[0], -1) - state[:, :-1]
    #     batch_size = state.shape[0]
    #     num_actions = state.shape[1] - 1
    #     mask = torch.zeros(diff.shape)
    #     for i in range(batch_size):
    #         if diff[i].count_nonzero().item() <= state[i, -1].item() - 2:
    #             mask[i] = torch.ones(num_actions)
    #         else:
    #             mask[i] = (diff[i] != 0).float()
    #     return mask

    def create_backwards_actions_mask(self, state: torch.Tensor):
        diff = self.initial_lattice.flatten().expand(self.batch_size, self.initial_lattice.shape[0] * self.initial_lattice.shape[1]) - state[:, :-1]
        diff_counts = diff.count_nonzero(dim=1).unsqueeze(1).expand(self.batch_size, self.initial_lattice.shape[0] * self.initial_lattice.shape[1])
        trajectory_lengths = state[:, -1].unsqueeze(1).expand(self.batch_size, self.initial_lattice.shape[0] * self.initial_lattice.shape[1])
        mask = torch.where(
            diff_counts <= trajectory_lengths - 2, 
            torch.ones_like(diff), 
            (diff != 0).float()
        )   
        return mask
    
if __name__ == "__main__":
    height = 7
    width = 7
    initial_lattice = random_tensor = 2 * torch.bernoulli(torch.full((height, width), 0.5)) - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = GFNAgentInit(initial_lattice=initial_lattice, 
                         trajectory_len=height*width, 
                         hidden_size=256, 
                         temp=.1, 
                         batch_size=256, 
                         replay_batch_size=256, 
                         buffer_size=100_000,
                         alpha=0.5,
                         beta=0.1,
                         device=device)
    os.makedirs(agent.output_folder, exist_ok=True)
    agent.train_gflownet(iterations=300, p=0.005)

    for i in range(20):
        trajectory, forward_probs, backward_probs, actions, reward = agent.sample_trajectory(eps=0)
        lattices = [state[:, :-1].reshape((agent.batch_size, agent.height, agent.width)) for state in trajectory]
        # visualize_trajectory(
        #     trajectory=[lattice[0] for lattice in lattices],  # Visualize only the first batch item
        #     filename=os.path.join(agent.output_folder, f"trajectory_{i}.gif"),
        #     reward=reward[0].item()
        # )
        visualize_terminal_state(
            lattice=lattices[-1][0],  # Visualize the terminal state of the first batch item
            filename=os.path.join(agent.output_folder, f"trajectory_{i}.png")
        )
    plt.show()
    print("Done")