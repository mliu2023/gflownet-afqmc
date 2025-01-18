import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import os

from abc import ABC, abstractmethod

from utils.visualize import visualize_terminal_state
from utils.prioritized_replay_buffer import PrioritizedReplayBuffer

class GFNAgentIsing(ABC):

    def __init__(self, 
                 initial_lattice, 
                 trajectory_len, 
                 hidden_size, 
                 temp, 
                 batch_size, 
                 replay_batch_size, 
                 buffer_size,
                 alpha,
                 beta,
                 env_folder,
                 device):
        super().__init__()
        self.initial_lattice = initial_lattice
        self.height = initial_lattice.shape[0]
        self.width = initial_lattice.shape[1]
        self.trajectory_len = trajectory_len
        self.sample_env = self.get_environment(self.initial_lattice, 
                                               trajectory_len, 
                                               temp, 
                                               batch_size)
        self.train_env = self.get_environment(self.initial_lattice, 
                                              trajectory_len, 
                                              temp, 
                                              replay_batch_size)

        self.batch_size = batch_size
        self.replay_batch_size = replay_batch_size
        self.buffer_size = buffer_size
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta)

        self.model = self.get_gflownet(self.initial_lattice, hidden_size)
        self.model.to(device)
        self.device = device
        self.output_folder = "trajectories/" \
                             f"{env_folder}/" \
                             f"{self.height}x{self.width}/" \
                             f"length_{self.trajectory_len}/" \
                             f"temp_{self.sample_env.temp}"
    
    @abstractmethod
    def get_environment(self, initial_lattice: torch.Tensor, 
                        trajectory_len: int, 
                        temp: float, 
                        batch_size: int):
        pass

    @abstractmethod
    def get_gflownet(self, initial_lattice: torch.Tensor, hidden_size: int):
        pass

    @abstractmethod
    def create_forward_actions_mask(self, state: torch.Tensor):
        pass

    @abstractmethod
    def create_backwards_actions_mask(self, state: torch.Tensor):
        pass
    
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
        log_probs = F.log_softmax(masked_flow, dim=1)
        probs = log_probs.exp()
        return probs

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
        masked_flow = torch.where(mask.bool(), backward_flow, torch.tensor(-float('inf')))
        log_probs = F.log_softmax(masked_flow, dim=1)
        probs = log_probs.exp()
        return probs

    # def get_action(self, probs: torch.Tensor, eps: float):
    #     action = torch.zeros(self.batch_size, dtype=torch.long)
    #     nonzero_indices = torch.nonzero(probs >= 1e-10, as_tuple=False)
    #     for i in range(self.batch_size):
    #         if random.random() < eps:
    #             action[i] = nonzero_indices[i][torch.randint(0, nonzero_indices[i].numel(), (1,))].item()
    #         else:
    #             action[i] = torch.multinomial(probs[i], 1).item()
    #     return action
    def get_action(self, probs: torch.Tensor, eps: float):
        batch_size = probs.shape[0]

        rand_vals = torch.rand(batch_size, 1)
        random_mask = rand_vals < eps

        uniform_actions = torch.multinomial((probs > 1e-10).float(), 1)
        probabilistic_actions = torch.multinomial(probs, 1)

        actions = torch.where(random_mask, uniform_actions, probabilistic_actions)
        return actions.squeeze(1)

    def sample_trajectory(self, eps: float):
        """
        Samples trajectories using the training policy. 
        The training policy uses the learned policy 1-epsilon 
        of the time and samples uniformly epsilon of the time.
        """
        state = self.sample_env.reset()
        trajectory = [state]
        forward_probs = []
        backward_probs = []
        actions = []
        for t in range(self.trajectory_len + 1):
            state.to(self.device)
            if t < self.trajectory_len:
                log_forward_flow_values, log_backward_flow_values = self.model(state)
                probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)

                # Debug information
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print(f"Invalid probabilities at step {t}")
                    print(f"Forward flow values: {log_forward_flow_values}")
                    print(f"Backward flow values: {log_backward_flow_values}")
                    print(f"Probs: {probs}")
                    print(f"Weights: {self.model.fwp.weight}")
                    print(f"Backward weights: {self.model.bwp.weight}")

                action = self.get_action(probs, eps)
                next_state, log_reward, done = self.sample_env.step(action)
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

    def back_sample_trajectory(self, eps: float):
        """
        Samples backwards trajectories using the training policy. 
        The training policy uses the network 1-epsilon of 
        the time and samples uniformly epsilon of the time.
        """
        state = self.sample_env.state
        log_reward = self.sample_env.reward_fn(self.sample_env.state[:, :-1].reshape(self.batch_size, 
                                                                                     self.sample_env.initial_lattice.shape[0], 
                                                                                     -1))
        trajectory = [state]
        forward_probs = []
        backward_probs = []
        actions = []
        for t in range(self.trajectory_len):
            state.to(self.device)
            if t < self.trajectory_len - 1:
                log_forward_flow_values, log_backward_flow_values = self.model(state)
                probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)

                # Debug information
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print(f"Invalid probabilities at step {t}")
                    print(f"Forward flow values: {log_forward_flow_values}")
                    print(f"Backward flow values: {log_backward_flow_values}")
                    print(f"Probs: {probs}")
                    print(f"Weights: {self.model.fwp.weight}")
                    print(f"Backward weights: {self.model.bwp.weight}")

                action = self.get_action(probs, eps)
                next_state, _, done = self.sample_env.reverse_step(action)
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

    def compute_probabilities(self, batch: list[tuple]):
        trajectory = []
        actions = []
        log_reward = []
        for (traj, act, log_rew) in batch:
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
            state.to(self.device)
            if t < self.trajectory_len:
                log_forward_flow_values, log_backward_flow_values = self.model(state)
                probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)

                # Debug information
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print(f"Invalid probabilities at step {t}")
                    print(f"Forward flow values: {log_forward_flow_values}")
                    print(f"Backward flow values: {log_backward_flow_values}")
                    print(f"Probs: {probs}")
                    print(f"Weights: {self.model.fwp.weight}")
                    print(f"Backward weights: {self.model.bwp.weight}")
                    # nans occur when a low-probability action is sampled using epsilon greedy

                next_state, log_reward, done = self.train_env.step(actions[:, t])
                state = next_state
                assert torch.equal(state, trajectory[:, t+1])

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
    
    def trajectory_balance_loss(self, 
                                forward_probs: torch.Tensor, 
                                backward_probs: torch.Tensor, 
                                log_reward: torch.Tensor):
        """
        Calculate Trajectory Balance Loss function as described in https://arxiv.org/abs/2201.13259.
        """
        forward_log_prob = torch.sum(torch.log(torch.clamp(torch.stack(forward_probs), min=1e-10)), dim=0)
        backward_log_prob = torch.sum(torch.log(torch.clamp(torch.stack(backward_probs), min=1e-10)), dim=0)
        log_ratio = self.model.log_Z + forward_log_prob - log_reward - backward_log_prob
        loss = log_ratio ** 2
        return loss.mean()

    def train_gflownet(self, iterations: int, p: float):
        optimizer = optim.Adam([
            {'params': self.model.log_Z, 'lr': 1e-1},
            {'params': self.model.network.parameters(), 'lr': 1e-3},
            {'params': self.model.fwp.parameters(), 'lr': 1e-3},
            {'params': self.model.bwp.parameters(), 'lr': 1e-3},], weight_decay=1e-5)

        warmup_episodes = 1000
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                      start_factor=0.1, 
                                                      total_iters=warmup_episodes)

        print(f"Filling replay buffer...")
        for _ in tqdm(range(self.buffer_size // self.batch_size)):
            trajectory, _, _, actions, log_reward = self.sample_trajectory(eps=1)
            self.replay_buffer.push(torch.stack(trajectory), torch.stack(actions), log_reward)

        for i in tqdm(range(iterations)):
            # Sample new trajectories
            for _ in range(int(p * self.buffer_size // self.batch_size)):
                trajectory, _, _, actions, log_reward = self.sample_trajectory(eps=0.01)
                self.replay_buffer.push(torch.stack(trajectory), torch.stack(actions), log_reward)

            # Train on batches from the replay buffer
            total_log_reward = 0
            total_loss = 0
            total_episodes = 0
            
            for _ in range(8):
                batch = self.replay_buffer.sample(self.replay_batch_size)
                
                # Compute probabilities for the batch
                _, fwd_probs, bwd_probs, _, log_rew = self.compute_probabilities(batch)
                
                optimizer.zero_grad()
                loss = self.trajectory_balance_loss(fwd_probs, bwd_probs, log_rew)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_log_reward += log_rew.mean().item()
                total_loss += loss.item()
                total_episodes += 1

            if i % 40 == 0:
                trajectory, _, _, actions, _ = self.sample_trajectory(eps=0)
                lattices = [state[:, :-1].reshape((self.batch_size, self.height, self.width)) for state in trajectory]
                visualize_terminal_state(
                    lattice=lattices[-1][0],
                    filename=os.path.join(self.output_folder, f"training_trajectory_{i}.png")
                )
            # print(f"Avg loss: {total_loss / total_episodes}, Avg log reward: {total_log_reward / total_episodes}")