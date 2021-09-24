import random
import torch
import numpy as np


class ReplayMemory:
    def __init__(self, buffer_size=1_000_000, device='cpu'):
        self.data = []
        self.position = 0
        self.buffer_size = buffer_size
        self.device = device

        self.i_episode = []
        self.frequency_sampled = []

    def sample(self, batch_size):
        def merge(var):
            if type(var[0]) is np.ndarray:
                return torch.Tensor(np.stack(var)).to(self.device)
            elif type(var[0]) is torch.Tensor:
                return torch.cat(var, dim=0).to(self.device)
            else:
                return torch.Tensor([[v] for v in var]).to(self.device)

        batch_idx = random.sample(range(len(self.data)), k=batch_size)
        batch = [self.data[idx] for idx in batch_idx]

        for idx in batch_idx:
            self.frequency_sampled[idx] += 1

        data = []
        for var in zip(*batch):
            if type(var[0]) is tuple:
                data.append(tuple([merge(v) for v in zip(*var)]))
            else:
                data.append(merge(var))

        stats = {
            # 'batch_recency': sum(self.i_episode[idx] for idx in batch_idx) / batch_size,
            # 'batch_repeated': sum(self.frequency_sampled[idx] > 1 for idx in batch_idx) / batch_size,
            # 'experience_utilization': sum(self.frequency_sampled[idx] > 0 for idx in range(len(self.data))) / len(self.data),
        }

        return tuple(data), stats

    def push(self, transition, i_episode=None):
        if len(self.data) == self.position:
            self.data.append(None)
            self.frequency_sampled.append(0)
            self.i_episode.append(i_episode)

        self.data[self.position] = transition
        self.frequency_sampled[self.position] = 0
        self.i_episode[self.position] = i_episode

        self.position = (self.position + 1) % self.buffer_size

    def __len__(self):
        return len(self.data)
