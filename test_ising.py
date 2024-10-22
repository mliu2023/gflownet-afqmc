import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import random
from collections import deque
import numpy as np

from test_env import IsingEnvironmentVoid
from visualize import visualize_trajectory, visualize_terminal_state

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, trajectory, forward_probs, backward_probs, actions, log_reward):
        for i in range(len(trajectory[0])):
            self.buffer.append((trajectory[:, i], forward_probs[:, i], backward_probs[:, i], actions[:, i], log_reward[i]))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

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
        return self.fwp(state), 1

class GFNAgent():
    def __init__(self, initial_lattice, trajectory_len, hidden_size, temp, batch_size, replay_batch_size = 32, buffer_size=100):
        super().__init__()
        self.initial_lattice = initial_lattice
        self.height = initial_lattice.shape[0]
        self.width = initial_lattice.shape[1]
        self.trajectory_len = trajectory_len
        self.batch_size = batch_size
        self.replay_batch_size = replay_batch_size

        self.env = IsingEnvironmentVoid(self.initial_lattice, trajectory_len, temp, batch_size)
        self.train_env = IsingEnvironmentVoid(self.initial_lattice, trajectory_len, temp, replay_batch_size)

        self.model = GFlowNetVoid(self.initial_lattice, hidden_size)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def create_forward_actions_mask(self, state: torch.Tensor):
        mask = (state[:, :-1] == 0).float()
        return torch.cat((mask, mask), 1)

    def mask_and_norm_forward_actions(self, state: torch.Tensor, forward_flow: torch.Tensor):
        mask = self.create_forward_actions_mask(state)
        masked_flow = torch.where(mask.bool(), forward_flow, torch.tensor(-float('inf')))
        log_probs = F.log_softmax(masked_flow, dim=1)
        probs = log_probs.exp()
        return probs

    def sample_trajectory(self, greedy=False):
        state = self.env.reset()
        trajectory = [state]
        forward_probs = []
        backward_probs = []
        actions = []
        for t in range(self.trajectory_len + 1):
            if t < self.trajectory_len:
                log_forward_flow_values, log_backward_flow_values = self.model(state[:, :-1])
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
                action = torch.zeros(self.batch_size, dtype=torch.long)
                for i in range(self.batch_size):
                    if not greedy and random.random() < eps:
                        nonzero_indices = torch.nonzero(probs[i], as_tuple=False)
                        if nonzero_indices.numel() > 0:
                            action[i] = nonzero_indices[torch.randint(0, nonzero_indices.numel(), (1,))].item()
                        else:
                            raise Exception(f"No valid actions for batch item {i}")
                    else:
                        action[i] = torch.multinomial(probs[i], 1).item()

                next_state, log_reward, done = self.env.step(action)
                state = next_state

                forward_probs.append(probs[torch.arange(self.batch_size), action])
                actions.append(action)
                trajectory.append(next_state)

            if t >= 1:
                prev_state = trajectory[t-1]
                curr_state = trajectory[t]
                diff = (curr_state - prev_state)[:, :-1]
                back_action = diff.abs().argmax(dim=1)
                prev_probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)
                backward_probs.append(prev_probs[torch.arange(self.batch_size), back_action])

            if done.all():
                break

        return trajectory, forward_probs, backward_probs, actions, log_reward

    def create_backwards_actions_mask(self, state: torch.Tensor):
        mask = (state[:, :-1] != 0).float()
        return mask

    def mask_and_norm_backward_actions(self, state: torch.Tensor, backward_flow: torch.Tensor):
        mask = self.create_backwards_actions_mask(state)
        masked_flow = torch.where(mask.bool(), backward_flow, torch.tensor(-float('inf')))
        log_probs = F.log_softmax(masked_flow, dim=1)
        probs = log_probs.exp()
        return probs

    def back_sample_trajectory(self):
        state = self.env.state
        log_reward = self.env.reward_fn(self.env.state[:, :-1].reshape(self.batch_size, self.env.initial_lattice.shape[0], -1))
        trajectory = [state]
        forward_probs = []
        backward_probs = []
        actions = []
        for t in range(self.trajectory_len - 1):
            log_forward_flow_values, log_backward_flow_values = self.model(state[:, :-1])
            probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)
            action = torch.multinomial(probs, 1).squeeze(-1)
            next_state, _, done = self.env.reverse_step(action)
            state = next_state

            backward_probs.append(probs[torch.arange(self.batch_size), action])
            actions.append(action)
            trajectory.append(next_state)

            if t >= 1:
                prev_state = trajectory[t-1]
                curr_state = trajectory[t]
                diff = (curr_state - prev_state)[:, :-1]
                prev_action = diff.abs().argmax(dim=1)
                prev_probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)
                forward_probs.append(prev_probs[torch.arange(self.batch_size), prev_action])

            if done.all():
                break

        return trajectory, forward_probs, backward_probs, actions, log_reward

    def trajectory_balance_loss(self, forward_probs: torch.Tensor, backward_probs: torch.Tensor, log_reward: torch.Tensor, step):
        forward_log_prob = torch.sum(torch.log(torch.stack(forward_probs)), dim=0)
        backward_log_prob = torch.sum(torch.log(torch.stack(backward_probs)), dim=0) if backward_probs else torch.zeros(self.batch_size)
        log_ratio = self.env.beta * self.model.log_Z + forward_log_prob - log_reward - backward_log_prob
        if step % 500 == 0:
            print(f"log_Z: {self.model.log_Z.item()}")
            print(f"forward_log_prob: {forward_log_prob.mean().item()}")
            print(f"log_reward: {log_reward.mean().item()}")
            print(f"backward_log_prob: {backward_log_prob.mean().item()}")
        loss = log_ratio ** 2
        return self.env.temp * loss.mean()

    def recompute_probabilities(self, batch):
        trajectory = []
        actions = []
        log_reward = []
        for (traj, _, _, act, log_rew) in batch:
            trajectory.append(traj)
            actions.append(act)
            log_reward.append(log_rew)
        trajectory = torch.stack(trajectory)
        actions = torch.stack(actions)
        log_reward = torch.stack(log_reward)
            
        state = self.train_env.reset()
        forward_probs = []
        backward_probs = []
        for t in range(self.trajectory_len + 1):
            if t < self.trajectory_len:
                log_forward_flow_values, log_backward_flow_values = self.model(state[:, :-1])
                probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)

                # Debug information
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print(f"Invalid probabilities at step {t}")
                    print(f"Forward flow values: {log_forward_flow_values}")
                    print(f"Backward flow values: {log_backward_flow_values}")
                    print(f"Probs: {probs}")
                    print(f"Weights: {self.model.fwp.weight}")
                    print(f"Backward weights: {self.model.bwp.weight}")

                next_state, log_reward, done = self.train_env.step(actions[:, t])
                state = next_state

                forward_probs.append(probs[torch.arange(self.replay_batch_size), actions[:, t]])

            if t >= 1:
                prev_state = trajectory[:, t-1]
                curr_state = trajectory[:, t]
                diff = (curr_state - prev_state)[:, :-1]
                back_action = diff.abs().argmax(dim=1)
                prev_probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)
                backward_probs.append(prev_probs[torch.arange(self.replay_batch_size), back_action])

            if done.all():
                break
        
        return trajectory, forward_probs, backward_probs, actions, log_reward

    def train_gflownet(self, num_episodes=5000):
        optimizer = optim.Adam([
            {'params': self.model.log_Z, 'lr': 1e-1},
            {'params': self.model.network.parameters(), 'lr': 1e-4},
            {'params': self.model.fwp.parameters(), 'lr': 1e-4},
            {'params': self.model.bwp.parameters(), 'lr': 1e-4},], weight_decay=5e-5)

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
            total_log_reward += log_reward.mean().item()
            
            # Add experience to replay buffer
            self.replay_buffer.push(torch.stack(trajectory), torch.stack(forward_probs), torch.stack(backward_probs), torch.stack(actions), log_reward)

            # Train on a batch from the replay buffer
            if len(self.replay_buffer) >= self.replay_batch_size:
                batch = self.replay_buffer.sample(self.replay_batch_size)
                
                # Recompute probabilities for the batch
                traj, fwd_probs, bwd_probs, acts, log_rew = self.recompute_probabilities(batch)
                
                optimizer.zero_grad()
                loss = self.trajectory_balance_loss(fwd_probs, bwd_probs, log_rew, episode)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}, Avg loss: {total_loss / 1000}, Avg log reward: {total_log_reward / 1000}")
                total_log_reward = 0
                total_loss = 0

if __name__ == "__main__":
    height = 9
    width = 9
    batch_size = 8 # Adjust this value as needed
    initial_lattice = torch.zeros((height, width))
    agent = GFNAgent(initial_lattice=initial_lattice, trajectory_len=height*width, hidden_size=128, temp=0.01, batch_size=batch_size)
    agent.train_gflownet()
    os.makedirs(f"eval_trajectories_void/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}", exist_ok=True)
    for i in range(5):
        trajectory, forward_probs, backward_probs, actions, reward = agent.sample_trajectory(greedy=True)
        lattices = [state[:, :-1].reshape((agent.batch_size, agent.height, agent.width)) for state in trajectory]
        visualize_trajectory(
            trajectory=[lattice[0] for lattice in lattices],  # Visualize only the first batch item
            filename=f"eval_trajectories_void/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}/trajectory_{i}.gif",
            reward=reward[0].item(),
        )
        visualize_terminal_state(
            lattice=lattices[-1][0],  # Visualize only the first batch item
            filename=f"eval_trajectories_void/{agent.height}x{agent.width}/length_{agent.trajectory_len}/temp_{agent.env.temp}/trajectory_{i}.png"
        )
    plt.show()
    print("Done")