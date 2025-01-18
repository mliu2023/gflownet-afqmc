import torch
import torch.nn as nn
import os
from tqdm import tqdm

from gflownet_ising import GFNAgentIsing
from environments.ising_void_env import IsingEnvironmentVoid
from visualize import visualize_trajectory, visualize_terminal_state, visualize_reward_distribution

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
        return self.fwp(state), 1

class GFNAgentIsingVoid(GFNAgentIsing):
    def get_env_name(self):
        return "void_2"
    
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

    def train_gflownet(self, iterations, steps_per_iteration, resamples_per_iteration):
        optimizer = torch.optim.Adam([
            {'params': self.model.log_Z, 'lr': 1e-1},
            {'params': self.model.network.parameters(), 'lr': 1e-3},
            {'params': self.model.fwp.parameters(), 'lr': 1e-3},
            {'params': self.model.bwp.parameters(), 'lr': 1e-3},], weight_decay=1e-5)

        # warmup_episodes = int(0.5 * iterations * steps_per_iteration)
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
            for _ in range(resamples_per_iteration):
                trajectory, _, _, actions, log_reward = self.sample_trajectory(eps=0.01)
                self.replay_buffer.push(torch.stack(trajectory), torch.stack(actions), log_reward)

            # Train on batches from the replay buffer
            total_log_reward = 0
            total_loss = 0
            total_episodes = 0
            
            for _ in range(steps_per_iteration):
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

if __name__ == "__main__":
    height = 7
    width = 7
    initial_lattice = torch.zeros((height, width))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for temp in [0.1, 0.3, 1, 3, 10]:
    # for temp in [0.1]:
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
            visualize_terminal_state(
                lattice=lattices[-1][0],  # Visualize the terminal state of the first batch item
                filename=os.path.join(agent.output_folder, f"trajectory_{i}.png")
            )
            rewards.append(reward)
        rewards = torch.stack(rewards).flatten().numpy()
        visualize_reward_distribution(rewards, os.path.join(agent.output_folder, "rewards.png"))
        print("Done")