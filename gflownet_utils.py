import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from abc import ABC, abstractmethod

class GFNAgent(ABC):

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

        self.env = self.get_environment(self.initial_lattice, trajectory_len, temp)

        self.model = self.get_gflownet(self.initial_lattice, hidden_size)
    
    @abstractmethod
    def get_environment(self, initial_lattice, trajectory_len, temp):
        pass

    @abstractmethod
    def get_gflownet(self, initial_lattice, hidden_size):
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
        masked_actions = mask * forward_flow
        normalized_actions = masked_actions / (masked_actions.sum(axis=0, keepdims=True) + 1e-10)
        return normalized_actions

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
        masked_actions = mask * backward_flow
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
            log_forward_flow_values, log_backward_flow_values = self.model(state[:-1])
            probs = self.mask_and_norm_backward_actions(state, F.softmax(log_backward_flow_values, dim=0))
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
                prev_probs = self.mask_and_norm_forward_actions(state, F.softmax(log_forward_flow_values, dim=0))
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

    def train_gflownet(self, num_episodes=10000, batch_size = 1):
        optimizer = optim.Adam([
            {'params': self.model.log_Z, 'lr': 3e-2},
            {'params': self.model.network.parameters(), 'lr': 1e-3},
            {'params': self.model.fwp.parameters(), 'lr': 1e-3},
            {'params': self.model.bwp.parameters(), 'lr': 1e-3},], weight_decay=5e-5)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes)
        # scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

        total_log_reward = 0
        total_loss = 0
        for episode in tqdm(range(num_episodes)):
            losses = []
            log_rewards = []
            for _ in range(batch_size):
                trajectory_1, forward_probs_1, backward_probs_1, actions_1, log_reward_1 = self.sample_trajectory()
                # trajectory_2, forward_probs_2, backward_probs_2, actions_2, log_reward_2 = self.back_sample_trajectory()
                
                loss_forward = self.trajectory_balance_loss(forward_probs_1, backward_probs_1, log_reward_1)
                # loss_backward = self.trajectory_balance_loss(forward_probs_2, backward_probs_2, log_reward_2)
                # loss = (loss_forward + loss_backward) / 2
                losses.append(loss_forward)
                log_rewards.append(log_reward_1)
            loss = torch.stack(losses).mean()
            log_reward = torch.stack(log_rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            total_log_reward += log_reward.item()
            total_loss += loss.item()

            if (episode + 1) % (num_episodes // 10) == 0:
                print(self.model.log_Z)
                print(f"Episode {episode + 1}, Avg loss: {total_loss / (num_episodes // 10)}, Avg forward log(reward): {total_log_reward / (num_episodes // 10)}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                total_log_reward = 0
                total_loss = 0