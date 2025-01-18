import torch
import heapq
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, beta):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.upper_capacity = int(self.capacity * beta)
        self.lower_capacity = self.capacity - int(self.capacity * beta)
        # self.buffer = deque(maxlen=capacity)
        # self.priorities = deque(maxlen=capacity)
        self.upper_buffer = [] # for the top alpha rewards
        self.lower_buffer = [] # for the bottom (1-alpha) rewards
        self.counter = 0

    def push(self, trajectory, actions, log_reward): 
        for i in range(len(trajectory[0])):
            # self.buffer.append((trajectory[:, i], actions[:, i], log_reward[i]))
            # self.priorities.append(log_reward[i].item())
            replay = (log_reward[i].item(),
                      self.counter,
                      (trajectory[:, i], actions[:, i], log_reward[i]))
            if len(self.upper_buffer) >= self.upper_capacity:
                popped_replay = heapq.heappushpop(self.upper_buffer, replay)
                if len(self.lower_buffer) >= self.lower_capacity:
                    heapq.heappushpop(self.lower_buffer, popped_replay)
                else:
                    heapq.heappush(self.lower_buffer, popped_replay)
            else:
                heapq.heappush(self.upper_buffer, replay)
            self.counter += 1

    def get_probabilities(self):
        if len(self.priorities) == 0:
            return None
            
        rewards = torch.tensor(list(self.priorities), dtype=torch.float64)
        min_reward, max_reward = rewards.min(), rewards.max()
        priorities = (rewards - min_reward) / (max_reward - min_reward)
        priorities = priorities ** 2
        probs = priorities / priorities.sum()
        return probs

    def sample(self, batch_size):
        # probs = self.get_probabilities()
        # indices = torch.multinomial(probs, batch_size, replacement=False)
        # samples = [self.buffer[idx.item()] for idx in indices]
        
        # return samples
        if batch_size <= len(self.upper_buffer):
            samples = random.choices(self.upper_buffer, k=batch_size)
            return [sample[-1] for sample in samples]
        else:
            upper_size = int(batch_size * self.alpha)
            lower_size = batch_size - upper_size

            upper_samples = random.choices(self.upper_buffer, k=upper_size)
            lower_samples = random.choices(self.lower_buffer, k=lower_size)
            return [sample[-1] for sample in upper_samples] + [sample[-1] for sample in lower_samples]

    def __len__(self):
        return len(self.upper_buffer) + len(self.lower_buffer)