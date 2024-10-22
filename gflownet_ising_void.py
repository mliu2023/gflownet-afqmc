import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import random
import math

from env import IsingEnvironmentVoid
from visualize import visualize_trajectory, visualize_terminal_state

class GFlowNetVoid(nn.Module):
    def __init__(self, initial_lattice, hidden_size):
        super(GFlowNetVoid, self).__init__()
        input_size = initial_lattice.shape[0] * initial_lattice.shape[1]
        forward_output_size = 2 * initial_lattice.shape[0] * initial_lattice.shape[1]
        backward_output_size = initial_lattice.shape[0] * initial_lattice.shape[1]
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
        state = self.network(state)
        # return F.softplus(self.fwp(state)), F.softplus(self.bwp(state))
        # return self.fwp(state).exp(), self.bwp(state).exp()
        return self.fwp(state), 1

class GFNAgent():
    def __init__(self, initial_lattice, trajectory_len, hidden_size, temp):
        super().__init__()
        self.initial_lattice = initial_lattice  # all trajectories will start from this lattice
        self.height = initial_lattice.shape[0]
        self.width = initial_lattice.shape[1]
        self.trajectory_len = trajectory_len

        self.env = IsingEnvironmentVoid(self.initial_lattice, trajectory_len, temp)

        self.model = GFlowNetVoid(self.initial_lattice, hidden_size)

        self.buffer = {'forward_probs': [], 'backward_probs': [], 'log_reward': []}

    def create_forward_actions_mask(self, state: torch.Tensor):
        """
        Build a list of boolean masks with zeros for invalid actions and ones for valid actions

        Args:
            state (torch.Tensor): state (lattice + timestep)

        Returns:
            (torch.Tensor): Mask corresponding to the lattice in input
        """
        mask = (state[:-1] == 0).float()
        return torch.cat((mask, mask), 0)

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
        masked_flow = torch.where(mask.bool(), forward_flow, torch.tensor(-float('inf')))
        log_probs = F.log_softmax(masked_flow, dim=0)
        probs = log_probs.exp()
        return probs

    def sample_trajectory(self, greedy=False):
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
        for t in range(self.trajectory_len + 1):
            if t < self.trajectory_len:
                log_forward_flow_values, log_backward_flow_values = self.model(state[:-1])
                probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)

                # Debug information
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print(f"Invalid probabilities at step {t}")
                    print(f"Forward flow values: {log_forward_flow_values}")
                    print(f"Backward flow values: {log_backward_flow_values}")
                    print(f"Probs: {probs}")
                    print(f"Weights: {self.model.fwp.weight}")
                    print(f"Backward weights: {self.model.bwp.weight}")

                eps = 0.4
                if not greedy and random.random() < eps:
                    nonzero_indices = torch.nonzero(probs, as_tuple=False)
                    if nonzero_indices.numel() > 0:
                        action = nonzero_indices[torch.randint(0, nonzero_indices.numel(), (1,))].item()
                    else:
                        raise Exception("this is bad")
                else:
                    action = torch.multinomial(probs, 1).item()

                # action = torch.multinomial(probs, 1).item()
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
                prev_probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)
                backward_probs.append(prev_probs[back_action])

            if done and t == self.trajectory_len:
                break

        if not greedy:
            self.buffer['forward_probs'].append(forward_probs)
            self.buffer['backward_probs'].append(backward_probs)
            self.buffer['log_reward'].append(log_reward)
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
        mask = (state[:-1] != 0).float()
        return mask

    def mask_and_norm_backward_actions(self, state: torch.Tensor, backward_flow: torch.Tensor):
        """
        Improved method to calculate backward probabilities.
        """
        mask = self.create_backwards_actions_mask(state)
        masked_flow = torch.where(mask.bool(), backward_flow, torch.tensor(-float('inf')))
        log_probs = F.log_softmax(masked_flow, dim=0)
        probs = log_probs.exp()
        return probs

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
        for t in range(self.trajectory_len):
            if t < self.trajectory_len - 1:
                log_forward_flow_values, log_backward_flow_values = self.model(state[:-1])
                probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)
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
                prev_probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)
                forward_probs.append(prev_probs[prev_action])

            if done and t == self.trajectory_len - 1:
                break

        return trajectory, forward_probs, backward_probs, actions, log_reward

    def trajectory_balance_loss(self, forward_probs: torch.Tensor, backward_probs: torch.Tensor, log_reward: torch.Tensor, step):
        """
        Calculate Trajectory Balance Loss function with improved numerical stability.
        """
        forward_log_prob = torch.sum(torch.log(torch.stack(forward_probs)))
        backward_log_prob = torch.sum(torch.log(torch.stack(backward_probs))) if backward_probs else torch.tensor(0.0)
        log_ratio = self.env.beta * self.model.log_Z + forward_log_prob - log_reward - backward_log_prob
        if step % 500 == 0:
            print(self.model.log_Z)
            print(forward_log_prob)
            print(log_reward)
            print(backward_log_prob)
        loss = log_ratio ** 2
        return self.env.temp * loss

    def train_gflownet(self, num_episodes=20000):
        optimizer = optim.Adam([
            {'params': self.model.log_Z, 'lr': 1e-1},
            {'params': self.model.network.parameters(), 'lr': 1e-4},
            {'params': self.model.fwp.parameters(), 'lr': 1e-4},
            {'params': self.model.bwp.parameters(), 'lr': 1e-4},], weight_decay=5e-5)
        
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes)

        # Warmup scheduler (adjust the number of epochs/steps as needed)
        warmup_episodes = 1000
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_episodes)

        # Cosine Annealing scheduler (adjust T_max to control the duration of cosine annealing)
        T_max = num_episodes - warmup_episodes
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

        # Combine warmup and cosine annealing schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_episodes])

        total_log_reward = 0
        total_loss = 0
        for episode in tqdm(range(num_episodes)):
            trajectory, forward_probs, backward_probs, actions, log_reward = self.sample_trajectory()
            
            loss = self.trajectory_balance_loss(forward_probs, backward_probs, log_reward, episode)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            total_log_reward += log_reward.item()
            total_loss += loss.item()

            if (episode + 1) % 1000 == 0:
                print(self.model.log_Z)
                print(f"Episode {episode + 1}, Avg loss: {total_loss / 1000}, Avg log reward: {total_log_reward / 1000}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                total_log_reward = 0
                total_loss = 0

if __name__ == "__main__":
    height = 9
    width = 9
    initial_lattice = torch.zeros((height, width))
    agent = GFNAgent(initial_lattice=initial_lattice, trajectory_len=height*width, hidden_size = 128, temp=0.1)
    agent.train_gflownet()
    os.makedirs(f"eval_trajectories_void/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}", exist_ok=True)
    for i in range(5):
        trajectory, forward_probs, backward_probs, actions, reward = agent.sample_trajectory(greedy=True)
        lattices = [state[:-1].reshape((agent.height, agent.width)) for state in trajectory]
        visualize_trajectory(
            trajectory=lattices,
            filename=f"eval_trajectories_void/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}/trajectory_{i}.gif",
            reward=reward.item(),
        )
        visualize_terminal_state(
            lattice=lattices[-1],
            filename=f"eval_trajectories_void/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}/trajectory_{i}.png"
        )
    plt.show()
    print("Done")