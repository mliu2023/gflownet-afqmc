import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from matplotlib import pyplot as plt
from tqdm import tqdm

from env import IsingEnvironment
from visualize import visualize_trajectory

# INITIAL_LATTICE = torch.Tensor(
#     [[1, 0, 1, 1, 0, 0, 1], 
#      [1, 0, 0, 0, 1, 0, 1], 
#      [0, 0, 0, 1, 1, 0, 1],
#      [0, 1, 1, 0, 0, 1, 0],
#      [1, 1, 0, 1, 0, 1, 0],
#      [0, 0, 1, 1, 0, 1, 1],
#      [1, 0, 1, 1, 1, 0, 1]],
# )
INITIAL_LATTICE = torch.Tensor(
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
)

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
        # return F.softplus(self.fwp(state)), F.softplus(self.bwp(state))
        # return self.fwp(state).exp(), self.bwp(state).exp()
        return self.fwp(state).exp(), 1


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

    def __init__(self, initial_lattice, trajectory_len, hidden_size):
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

        self.env = IsingEnvironment(initial_lattice=self.initial_lattice, max_steps=trajectory_len)

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
        masked_actions = mask * forward_flow
        normalized_actions = masked_actions / (masked_actions.sum(axis=0, keepdims=True) + 1e-10)
        return normalized_actions

    def sample_trajectory(self):
        """
        Sample a trajectory using the current policy.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): (trajectory, forward probs, actions, rewards)
        """
        state = self.env.reset()
        # print(state)
        trajectory = [state]
        forward_probs = []
        backward_probs = []
        actions = []
        for t in range(self.trajectory_len):
            forward_flow_values, backward_flow_values = self.model(state[:-1])
            probs = self.mask_and_norm_forward_actions(state, forward_flow_values)
            action = torch.multinomial(probs, 1).item()
            # print(flow_values)
            # action = torch.distributions.categorical.Categorical(logits=flow_values).sample()
            # print(action)
            next_state, reward, done = self.env.step(action)
            state = next_state

            forward_probs.append(probs[action])
            actions.append(action)
            trajectory.append(next_state)

            if t >= 1:
                prev_state = trajectory[t-1]
                curr_state = trajectory[t]
                diff = (curr_state - prev_state)[:-1]
                back_action = diff.abs().argmax()
                prev_probs = self.mask_and_norm_backward_actions(state, backward_flow_values)
                backward_probs.append(prev_probs[back_action])

            if done:
                break

        return trajectory, forward_probs, backward_probs, actions, reward

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
        mask = self.create_forward_actions_mask(state)
        masked_actions = mask * backward_flow
        normalized_actions = masked_actions / (masked_actions.sum(axis=0, keepdims=True) + 1e-10)
        return normalized_actions

    def back_sample_trajectory(self, state: torch.Tensor):
        """
        Follow current backward policy from a position back to the origin.
        Returns them in "forward order" such that origin is first.

        Args:
            lattice (torch.Tensor): starting lattice for the back trajectory
        
        Returns:
            (torch.Tensor, torch.Tensor): (positions, actions)
        """
        # Attempt to trace a path back to the origin from a given position
        trajectory = [state]
        backward_probs = []
        actions = []
        for _ in range(self.max_trajectory_len - 1):
            _, back_flow_values = self.model(state)
            probs = self.mask_and_norm_backward_actions(state, back_flow_values)
            action = torch.multinomial(probs, 1)
            next_state, reward, done = self.env.reverse_step(action)
            state = next_state

            backward_probs.append(probs[action])
            actions.append(action)
            trajectory.append(next_state)

            if done:
                break

        return trajectory, backward_probs, actions, reward

    def trajectory_balance_loss(self, forward_probs: torch.Tensor, backward_probs: torch.Tensor, reward: torch.Tensor):
        """
        Calculate Trajectory Balance Loss function as described in https://arxiv.org/abs/2201.13259.
        """
        forward_log_prob = torch.sum(torch.log(torch.stack(forward_probs)))
        backward_log_prob = torch.sum(torch.log(torch.stack(backward_probs))) if backward_probs else torch.tensor(0.0)
        log_ratio = self.model.log_Z + forward_log_prob - torch.log(reward + 1e-10) - backward_log_prob
        loss = log_ratio ** 2
        return loss

    def train_gflownet(self, num_episodes=40000):
        optimizer = optim.Adam([
            {'params': self.model.log_Z, 'lr': 1e-2},
            {'params': self.model.network.parameters(), 'lr': 1e-3},
            {'params': self.model.fwp.parameters(), 'lr': 1e-3},
            {'params': self.model.bwp.parameters(), 'lr': 1e-3},], weight_decay=5e-5)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes)
        # scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

        total_reward = 0
        total_loss = 0
        for episode in tqdm(range(num_episodes)):
            trajectory, forward_probs, backward_probs, actions, reward = self.sample_trajectory()
            # total_reward += reward.item()
            loss = self.trajectory_balance_loss(forward_probs, backward_probs, reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            total_reward = reward
            total_loss += loss.item()

            if (episode + 1) % 1000 == 0:
                print(self.model.log_Z)
                print(f"Episode {episode + 1}, Loss: {total_loss:.4f}, Reward: {total_reward}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                total_reward = 0
                total_loss = 0
        # print(total_reward / num_episodes)


if __name__ == "__main__":
    """
    Notes:

    - Works well for shorter trajectory lengths (5)
    - Struggles with longer trajectory lengths (10)
        - maybe add a stopping action to allow for shorter paths?
    """
    agent = GFNAgent(initial_lattice=INITIAL_LATTICE, trajectory_len=9, hidden_size = 32)
    agent.train_gflownet()
    for i in range(5):
        trajectory, forward_probs, backward_probs, actions, reward = agent.sample_trajectory()
        lattices = [state[:-1].reshape((agent.height, agent.width)) for state in trajectory]
        visualize_trajectory(
            trajectory=lattices,
            filename=f"eval_trajectories/trajectory_{i}_{agent.height}_x_{agent.width}.gif",
            reward=reward.item(),
        )
    plt.show()
    print("Done")