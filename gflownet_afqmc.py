import torch
import torch.nn.functional as F
from tqdm import tqdm
from abc import ABC, abstractmethod

class GFNAgentAF(ABC):

    def __init__(self, 
                 n_fields,
                 nx,
                 ny, 
                 trajectory_len, 
                 hidden_size, 
                 batch_size, 
                 env_folder,
                 device,
                 fields,
                 energies):
        super().__init__()
        self.n_fields = n_fields
        self.nx = nx
        self.ny = ny
        self.trajectory_len = trajectory_len

        self.device = device
        self.policy_model = self.get_gflownet(n_fields, hidden_size)
        self.policy_model.to(device)
        self.energy_model = self.get_energy_model(n_fields, hidden_size)
        self.energy_model.to(device)

        self.batch_size = batch_size

        self.sample_env = self.get_environment(n_fields,
                                               trajectory_len, 
                                               self.energy_model,
                                               batch_size)
        self.train_env = self.get_environment(n_fields, 
                                              trajectory_len, 
                                              self.energy_model,
                                              batch_size)

        self.output_folder = "trajectories/" \
                             f"{env_folder}/" \
                             f"{n_fields}_fields"
        
        self.fields = fields
        self.energies = energies
    
    @abstractmethod
    def get_environment(self, 
                        n_fields: int,
                        trajectory_len: int, 
                        model: torch.nn.Module,
                        batch_size: int):
        pass

    @abstractmethod
    def get_gflownet(self, n_fields: int, hidden_size: int):
        pass

    @abstractmethod
    def get_energy_model(self, n_fields: int, hidden_size: int):
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
        # log_probs = F.log_softmax(masked_flow, dim=1)
        # probs = log_probs.exp()
        # return probs
        probs = F.softmax(masked_flow, dim=1)
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
        # log_probs = F.log_softmax(masked_flow, dim=1)
        # probs = log_probs.exp()
        # return probs
        probs = F.softmax(masked_flow, dim=1)
        return probs

    def get_action(self, probs: torch.Tensor, eps: float):
        batch_size = probs.shape[0]

        rand_vals = torch.rand(batch_size, 1)
        random_mask = rand_vals < eps
        uniform_actions = torch.multinomial((probs > 1e-10).float(), 1)
        probabilistic_actions = torch.multinomial(probs, 1)

        actions = torch.where(random_mask, uniform_actions, probabilistic_actions)
        return actions.squeeze(1)

    def forward_sample_trajectory(self, eps: float, K: int, state: torch.Tensor = None):
        """
        Samples trajectories using the training policy. 
        The training policy uses the learned policy 1-epsilon 
        of the time and samples uniformly epsilon of the time.
        Starts at the initial state by default.
        """
        if state is None:
            state = self.sample_env.reset()
        self.sample_env.state = state

        trajectory = torch.empty((K+1, self.batch_size, self.n_fields+1))
        trajectory[0] = state
        forward_probs = torch.empty((K, self.batch_size))
        backward_probs = torch.empty((K, self.batch_size))
        actions = torch.empty((K, self.batch_size))
        # trajectory = [state]
        # forward_probs = []
        # backward_probs = []
        # actions = []

        for t in range(K+1):
            if t < K:
                state = state.to(self.device)
                log_forward_flow_values, log_backward_flow_values = self.policy_model(state)
                log_forward_flow_values.cpu()
                log_backward_flow_values.cpu()
                state.cpu()

                probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)
                action = self.get_action(probs, eps)
                
                next_state, log_reward, done = self.sample_env.step(action)
                state = next_state

                forward_probs[t] = probs[torch.arange(self.batch_size), action]
                actions[t] = action
                trajectory[t+1] = next_state
                # forward_probs.append(probs[torch.arange(self.batch_size), action])
                # actions.append(action)
                # trajectory.append(next_state)

            if t > 0:
                prev_state = trajectory[t-1]
                curr_state = trajectory[t]
                diff = (curr_state - prev_state)[:, :-1]
                back_action = diff.abs().argmax(dim=1)
                prev_probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)
                backward_probs[t-1] = prev_probs[torch.arange(self.batch_size), back_action]
                # backward_probs.append(prev_probs[torch.arange(self.batch_size), back_action])

            # if done:
            #     break
            
        return trajectory, forward_probs, backward_probs, actions, log_reward

    def back_sample_trajectory(self, eps: float, K: int, state: torch.Tensor = None):
        """
        Samples backwards trajectories using the training policy. 
        The training policy uses the learned policy 1-epsilon 
        of the time and samples uniformly epsilon of the time.
        """
        if state is None:
            state = self.sample_env.state
        self.sample_env.state = state
            
        log_reward = -self.sample_env.energy_fn(self.sample_env.state[:, :-1].reshape(self.batch_size, 
                                                                                      self.n_fields))
        
        trajectory = torch.empty((K+1, self.batch_size, self.n_fields+1))
        trajectory[0] = state
        forward_probs = torch.empty((K, self.batch_size))
        backward_probs = torch.empty((K, self.batch_size))
        actions = torch.empty((K, self.batch_size))
        # trajectory = [state]
        # forward_probs = []
        # backward_probs = []
        # actions = []
        
        for t in range(K+1):
            if t < K:
                state = state.to(self.device)
                log_forward_flow_values, log_backward_flow_values = self.policy_model(state)
                log_forward_flow_values.cpu()
                log_backward_flow_values.cpu()
                state.cpu()

                probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)
                action = self.get_action(probs, eps)
                
                next_state, _, done = self.sample_env.reverse_step(action)
                state = next_state

                backward_probs[t] = probs[torch.arange(self.batch_size), action]
                actions[t] = action
                trajectory[t+1] = next_state
                # backward_probs.append(probs[torch.arange(self.batch_size), action])
                # actions.append(action)
                # trajectory.append(next_state)

            if t > 0:
                prev_state = trajectory[t-1]
                curr_state = trajectory[t]
                diff = (curr_state - prev_state)[:, :-1]
                prev_action = diff.abs().argmax(dim=1)
                prev_probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)
                forward_probs[t-1] = prev_probs[torch.arange(self.batch_size), prev_action]
                # forward_probs.append(prev_probs[torch.arange(self.batch_size), prev_action])

            # if done:
            #     break

        return trajectory, forward_probs, backward_probs, actions, log_reward

    # useful if using a replay buffer
    def compute_probabilities(self, batch: list[tuple], K: int):
        trajectory = torch.empty((self.batch_size, K+1, self.n_fields+1))
        actions = torch.empty((self.batch_size, K))
        log_reward = torch.empty((self.batch_size, K))
        # trajectory = []
        # actions = []
        # log_reward = []
        for i, (traj, act, log_rew) in enumerate(batch):
            trajectory[i] = traj
            actions[i] = act
            log_reward[i] = log_rew
            # trajectory.append(traj)
            # actions.append(act)
            # log_reward.append(log_rew)
        # trajectory = torch.stack(trajectory)
        # actions = torch.stack(actions)
        # log_reward = torch.stack(log_reward)
            
        state = self.train_env.reset()
        forward_probs = torch.empty((self.batch_size, K))
        backward_probs = torch.empty((self.batch_size, K))
        # forward_probs = []
        # backward_probs = []
        for t in range(K+1):
            if t < K:
                state = state.to(self.device)
                log_forward_flow_values, log_backward_flow_values = self.policy_model(state)
                log_forward_flow_values.cpu()
                log_backward_flow_values.cpu()
                state.cpu()

                probs = self.mask_and_norm_forward_actions(state, log_forward_flow_values)

                next_state, log_reward, done = self.train_env.step(actions[:, t])
                state = next_state
                assert torch.equal(state, trajectory[:, t+1])
                forward_probs[:, t] = probs[torch.arange(self.batch_size), actions[:, t]]
                # forward_probs.append(probs[torch.arange(self.replay_batch_size), actions[:, t]])

            if t > 0:
                prev_state = trajectory[:, t-1]
                curr_state = trajectory[:, t]
                diff = (curr_state - prev_state)[:, :-1]
                back_action = diff.abs().argmax(dim=1)
                prev_probs = self.mask_and_norm_backward_actions(state, log_backward_flow_values)
                backward_probs[:, t-1] = prev_probs[torch.arange(self.batch_size), back_action]
                # backward_probs.append(prev_probs[torch.arange(self.replay_batch_size), back_action])

            if done:
                break
        
        return trajectory, forward_probs, backward_probs, actions, log_reward
    
    def trajectory_balance_loss(self, 
                                forward_probs: torch.Tensor, 
                                backward_probs: torch.Tensor, 
                                log_reward: torch.Tensor):
        """
        Calculate Trajectory Balance Loss function as described in https://arxiv.org/abs/2201.13259.
        """
        forward_log_prob = torch.sum(torch.log(torch.clamp(forward_probs, min=1e-10)), dim=0)
        backward_log_prob = torch.sum(torch.log(torch.clamp(backward_probs, min=1e-10)), dim=0)
        log_ratio = self.policy_model.log_Z + forward_log_prob - log_reward - backward_log_prob
        loss = log_ratio ** 2
        return loss.mean()

    def update_ebm(self, 
                   optimizer: torch.optim.Optimizer, 
                   K: int):
        indices = torch.randint(0, len(self.fields), (self.batch_size,))
        fields = self.fields[indices]
        # energy = self.energies[indices]
        assert(fields.shape == self.sample_env.state[:, :-1].shape)

        state = torch.cat([fields, torch.full((self.batch_size, 1), self.trajectory_len)], dim=1)
        self.sample_env.state = state
        _, forward_probs_to_x, backward_probs_from_x, _, _ = self.back_sample_trajectory(eps=0, K=K, state=state)
        _, forward_probs_to_x_fake, backward_probs_from_x_fake, _, _ = self.forward_sample_trajectory(eps=0, K=K, state=self.sample_env.state)

        x_energy = self.energy_model(fields)
        x_fake_energy = self.energy_model(self.sample_env.state[:, :-1])

        rand_vals = torch.rand(self.batch_size)
        transition_probs = torch.exp(x_energy-x_fake_energy+ 
                                     torch.sum(torch.log(torch.clip(forward_probs_to_x_fake, min=1e-10))-
                                               torch.log(torch.clip(backward_probs_from_x_fake, min=1e-10))-
                                               torch.log(torch.clip(forward_probs_to_x, min=1e-10))+
                                               torch.log(torch.clip(backward_probs_from_x, min=1e-10)), axis=0))
        transition_mask = (rand_vals < transition_probs).detach()
        diff = torch.where(
            transition_mask,
            x_energy - x_fake_energy,
            0
        )
        l2 = x_energy ** 2 + x_fake_energy ** 2
        # l2 = 0
        loss = torch.mean(diff + 0.1 * l2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.energy_model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss

    def train_gflownet(self, iterations, warmup_k):
        gflownet_optimizer = torch.optim.Adam([
            {'params': self.policy_model.log_Z, 'lr': 1e-1},
            {'params': self.policy_model.network.parameters(), 'lr': 1e-3},
            {'params': self.policy_model.fwp.parameters(), 'lr': 1e-3},
            {'params': self.policy_model.bwp.parameters(), 'lr': 1e-3},])

        ebm_optimizer = torch.optim.Adam(self.energy_model.parameters(), lr=1e-4)

        # training_lattices = []

        tqdm_bar = tqdm(range(iterations))
        for i in tqdm_bar:
            _, fwd_probs, bwd_probs, _, log_rew = self.forward_sample_trajectory(eps=1e-2, K=self.trajectory_len)
            
            gflownet_optimizer.zero_grad()
            loss = self.trajectory_balance_loss(fwd_probs, bwd_probs, log_rew)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
            gflownet_optimizer.step()

            ebm_loss = self.update_ebm(ebm_optimizer, K = min(self.n_fields, 1+int(i / warmup_k * self.n_fields)))

            trajectory, _, _, _, _ = self.forward_sample_trajectory(eps=0, K=self.trajectory_len)
            fields = [state[:, :-1].reshape((self.batch_size, self.nx, self.ny)) for state in trajectory]
            # training_lattices.append(fields[-1][0])

            tqdm_bar.set_postfix(gflownet_loss=f"{loss:.4f}", ebm_loss=f"{ebm_loss:.4f}", energy=f"{-log_rew.mean():.4f}")
            if i % 100 == 0:
                print(f"EBM loss: {ebm_loss:.4f}, Gflownet loss: {loss:.4f}, Avg energy: {-log_rew.mean():.4f}")
        # visualize_terminal_states(
        #     lattices=training_lattices,
        #     filename=os.path.join(self.output_folder, f"training_fields.png"),
        #     cols=16
        # )