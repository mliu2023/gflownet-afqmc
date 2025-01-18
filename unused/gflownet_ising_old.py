import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt
from tqdm import tqdm
import os

from environments.ising_env import IsingEnvironment
from visualize import visualize_trajectory, visualize_terminal_state

INITIAL_LATTICE = random_tensor = 2 * torch.bernoulli(torch.full((7, 7), 0.5)) - 1

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
        state = self.network(state)
        # return self.fwp(state), self.bwp(state)
        return self.fwp(state), 1


"""
How to make the paths go on a DAG?
- include timestep in the state
    - but can potentially ignore timestep when computing the flow?
- force energy to strictly decrease
How to calculate backwards mask?
- calculate number of different spins from starting state and 
check that it is less than or equal to the time step
"""

"""
Differences with Alan's implementation:

1. The paths generated are forced to form a DAG. This is done by adding the timestep to the state.
2. Some of the backwards actions are masked based on whether it was possible to get there from s_0.
3. The trajectory length is fixed at 10, so "stopping" is not an action.
4. This implementation is for the Ising model instead of the Hubbard model.
"""
class GFNAgent():
    """
    Example Generative Flow Network as described in:
    https://arxiv.org/abs/2106.04399 and
    https://arxiv.org/abs/2201.13259
    """

    def __init__(self, initial_lattice, trajectory_len, hidden_size, temp):
        """
        Initialize the GFlowNet agent.

        Args:
            max_trajectory_len (int): Max length for each sampled path
            n_hidden (int): Number of nodes in hidden layer of neural network
            epochs (int): Number of epochs to complete during training cycle
            lr (float): Learning rate
        """
        super().__init__()
        self.initial_lattice = initial_lattice  # all trajectories will start from this lattice
        self.height = initial_lattice.shape[0]
        self.width = initial_lattice.shape[1]
        self.trajectory_len = trajectory_len

        self.env = IsingEnvironment(self.initial_lattice, trajectory_len, temp)

        self.model = GFlowNet(self.initial_lattice, hidden_size)

    def create_forward_actions_mask(self, state: torch.Tensor):
        """
        Build a list of boolean masks with zeros for invalid actions and ones for valid actions

        Args:
            state (torch.Tensor): state (lattice + timestep)

        Returns:
            (torch.Tensor): Mask corresponding to the lattice in input
        """
        num_actions = state.shape[0] - 1
        return torch.ones(num_actions)

    def mask_and_norm_forward_actions(self, state: torch.Tensor, forward_flow: torch.Tensor):
        """
        Remove invalid actions and normalize probabilities so that they sum to one.

        Args:
            state (torch.Tensor): state (lattice + timestep)
            forward_probs (torch.Tensor): probabilities over actions

        Returns:
            (torch.Tensor): Masked and normalized probabilities
        """
        mask = self.create_forward_actions_mask(state)
        masked_actions = mask * F.softmax(forward_flow, dim=0)
        normalized_actions = masked_actions / (masked_actions.sum(axis=0, keepdims=True) + 1e-10)
        return normalized_actions

    def sample_trajectory(self):
        """
        Sample a trajectory using the current policy.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): (trajectory, forward probs, actions, rewards)
        """
        state = self.env.reset()
        trajectory = [state]
        forward_probs = []
        backward_probs = []
        actions = []
        for t in range(self.trajectory_len):
            log_forward_flow_values, log_backward_flow_values = self.model(state[:-1])
            probs = self.mask_and_norm_forward_actions(state, F.softmax(log_forward_flow_values, dim=0))

            # Debug information
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"Invalid probabilities at step {t}")
                print(f"Forward flow values: {log_forward_flow_values}")
                print(f"Backward flow values: {log_backward_flow_values}")
                print(f"Probs: {probs}")

            action = torch.multinomial(probs, 1).item()
            next_state, log_reward, done = self.env.step(action)
            state = next_state

            forward_probs.append(probs[action])
            actions.append(action)
            trajectory.append(next_state)

            if t >= 1:
                prev_state = trajectory[t-1]
                curr_state = trajectory[t]
                diff = (curr_state - prev_state)[:-1]
                back_action = diff.abs().argmax()
                prev_probs = self.mask_and_norm_backward_actions(state, F.softmax(log_backward_flow_values, dim=0))
                backward_probs.append(prev_probs[back_action])

            if done:
                break

        return trajectory, forward_probs, backward_probs, actions, log_reward

    def create_backwards_actions_mask(self, state: torch.Tensor):
        """
        Build a list of boolean masks with zeros for invalid previous actions 
        and ones for valid previous actions

        Args:
            state (torch.Tensor): state (lattice + timestep)

        Returns:
            (torch.Tensor): Mask corresponding to the lattice in input
        """
        diff = self.initial_lattice.flatten() - state[:-1]
        num_actions = state.shape[0] - 1
        if diff.count_nonzero().item() <= state[-1].item() - 2:
            return torch.ones(num_actions)
        else:
            return (diff != 0).float()

    def mask_and_norm_backward_actions(self, state: torch.Tensor, backward_flow: torch.Tensor):
        """
        Remove invalid actions and normalize probabilities so that they sum to one.

        Args:
            state (torch.Tensor): state (lattice + timestep)
            backward_probs (torch.Tensor): probabilities over actions

        Returns:
            (torch.Tensor): Masked and normalized probabilities
        """
        mask = self.create_backwards_actions_mask(state)
        masked_actions = mask * F.softmax(backward_flow, dim=0)
        normalized_actions = masked_actions / (masked_actions.sum(axis=0, keepdims=True) + 1e-10)
        return normalized_actions

    def back_sample_trajectory(self):
        """
        Follow current backward policy from a position back to the origin.
        Returns them in "forward order" such that origin is first.

        Args:
            lattice (torch.Tensor): starting lattice for the back trajectory
        
        Returns:
            (torch.Tensor, torch.Tensor): (positions, actions)
        """
        # Attempt to trace a path back to the origin from a given position
        state = self.env.state
        log_reward = self.env.reward_fn(self.env.state[:-1].reshape(self.env.initial_lattice.shape))
        trajectory = [state]
        forward_probs = []
        backward_probs = []
        actions = []
        for t in range(self.trajectory_len - 1):
            forward_flow_values, back_flow_values = self.model(state[:-1])
            probs = self.mask_and_norm_backward_actions(state, F.softmax(back_flow_values))
            action = torch.multinomial(probs, 1)
            next_state, _, done = self.env.reverse_step(action)
            state = next_state

            backward_probs.append(probs[action])
            actions.append(action)
            trajectory.append(next_state)

            if t >= 1:
                prev_state = trajectory[t-1]
                curr_state = trajectory[t]
                diff = (curr_state - prev_state)[:-1]
                prev_action = diff.abs().argmax()
                prev_probs = self.mask_and_norm_forward_actions(state, F.softmax(forward_flow_values))
                forward_probs.append(prev_probs[prev_action])

            if done:
                break

        return trajectory, forward_probs, backward_probs, actions, log_reward

    def trajectory_balance_loss(self, forward_probs: torch.Tensor, backward_probs: torch.Tensor, log_reward: torch.Tensor):
        """
        Calculate Trajectory Balance Loss function as described in https://arxiv.org/abs/2201.13259.
        """
        forward_log_prob = torch.sum(torch.log(torch.stack(forward_probs)))
        backward_log_prob = torch.sum(torch.log(torch.stack(backward_probs))) if backward_probs else torch.tensor(0.0)
        log_ratio = self.model.log_Z + forward_log_prob - log_reward - backward_log_prob
        loss = log_ratio ** 2
        return loss

    def trajectory_balance_loss(self, forward_probs: torch.Tensor, backward_probs: torch.Tensor, log_reward: torch.Tensor):
        """
        Calculate Trajectory Balance Loss function as described in https://arxiv.org/abs/2201.13259.
        """
        forward_log_prob = torch.sum(torch.log(torch.stack(forward_probs)))
        backward_log_prob = torch.sum(torch.log(torch.stack(backward_probs))) if backward_probs else torch.tensor(0.0)
        log_ratio = self.model.log_Z + forward_log_prob - log_reward - backward_log_prob
        loss = log_ratio ** 2
        return loss

    def train_gflownet(self, num_episodes=20000):
        optimizer = optim.Adam([
            {'params': self.model.log_Z, 'lr': 1e-1},
            {'params': self.model.network.parameters(), 'lr': 1e-3},
            {'params': self.model.fwp.parameters(), 'lr': 1e-3},
            {'params': self.model.bwp.parameters(), 'lr': 1e-3},], weight_decay=5e-5)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes)
        # scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

        # Warmup scheduler (adjust the number of epochs/steps as needed)
        warmup_episodes = 1000
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_episodes)

        # # Cosine Annealing scheduler (adjust T_max to control the duration of cosine annealing)
        # T_max = num_episodes - warmup_episodes
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

        # # Combine warmup and cosine annealing schedulers
        # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_episodes])

        total_log_reward = 0
        total_loss = 0
        for episode in tqdm(range(num_episodes)):
            trajectory_1, forward_probs_1, backward_probs_1, actions_1, log_reward_1 = self.sample_trajectory()
            # trajectory_2, forward_probs_2, backward_probs_2, actions_2, log_reward_2 = self.back_sample_trajectory()
            
            loss_forward = self.trajectory_balance_loss(forward_probs_1, backward_probs_1, log_reward_1)
            # loss_backward = self.trajectory_balance_loss(forward_probs_2, backward_probs_2, log_reward_2)
            # loss = (loss_forward + loss_backward) / 2
            loss = loss_forward
            log_reward = log_reward_1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_log_reward += log_reward.item()
            total_loss += loss.item()

            if (episode + 1) % (1000) == 0:
                print(self.model.log_Z)
                print(f"Episode {episode + 1}, Avg loss: {total_loss / (1000)}, Avg forward log(reward): {total_log_reward / (1000)}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                total_log_reward = 0
                total_loss = 0

if __name__ == "__main__":
    """
    Notes:
    9/25:

    - Works well for shorter trajectory lengths (5, 9)
    - Struggles with longer trajectory lengths (13)
        - maybe add a stopping action to allow for shorter paths?
    - Struggles with larger grids
    - Z_0, the sum of rewards (partition function), is generally too small
    - Maybe add backwards sampling into the training procedure to explore more states
        - https://proceedings.mlr.press/v162/zhang22v/zhang22v.pdf
    - "void" entries
        - everything is set to null, and the network learns to choose a void and turn it into ±1
    - learn the energy function (as a neural network)?
        - in this case, learn the J matrix
    - Add support for larger batch sizes


    8/9:
    - Fixed issue with backwards mask
    - Added implementation that starts with void states
    - Lower temperature makes it more ordered, but works less well for longer sequences
        - some spins don't ever change––not enough states being explored
            - e.g., 7x7, length 30, temp 0.1
        - maybe this many steps are unnecessary, or stopping should be allowed
    - Both methods converge to a mode instead of multiple modes
    
    """
    agent = GFNAgent(initial_lattice=INITIAL_LATTICE, trajectory_len=20, hidden_size = 64, temp=0.01)
    agent.train_gflownet()
    os.makedirs(f"eval_trajectories/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}", exist_ok=True)
    for i in range(5):
        trajectory, forward_probs, backward_probs, actions, reward = agent.sample_trajectory()
        lattices = [state[:-1].reshape((agent.height, agent.width)) for state in trajectory]
        visualize_trajectory(
            trajectory=lattices,
            filename=f"eval_trajectories/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}/trajectory_{i}.gif",
            reward=reward.item(),
        )
        visualize_terminal_state(
            lattice=lattices[-1],
            filename=f"eval_trajectories/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}/trajectory_{i}.png"
        )
    plt.show()
    print("Done")