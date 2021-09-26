import gym
import envs
from rlss.agents.sac_discrete import DSACAgent
from rlss.utils import ArgsManager, state_action_dims
from rlss.nets import TrTwinQNet, TrPolicyNet, BaseNet, BasePolicyNet
from typing import Tuple, List, Callable
from torch.utils.data import IterableDataset, DataLoader
import collections
import gym
import numpy as np
import time
import envs
import random
import operator
import functools
import torch
import threading
from torch.multiprocessing import Queue, Process
from multiprocessing.shared_memory import SharedMemory

CHUNK_SIZE = 10

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'state2', 'state3', 'state4', 'action', 'reward', 'done', 'new_state', 'new_state2', 'new_state3', 'new_state4'])

class ReplayBuffer:
    def __init__(self, dtypes, shapes, buffer_size: int = 1_000):
        sizes = [functools.reduce(operator.mul, shape) for shape in shapes]
        elem_nbytes = [np.dtype(dtype).itemsize * size for dtype, size in zip(dtypes, sizes)]

        self.shms = [
            SharedMemory(create=True, size=nbytes * buffer_size)
            for nbytes in elem_nbytes
        ]

        self.buffer = []

        for dtype, shape, shm in zip(dtypes, shapes, self.shms):
            shape = (shape[0] * buffer_size, *shape[1:])

            buf_arr = np.ndarray(shape, buffer=shm.buf, dtype=dtype)
            buf_arr[:] = np.zeros(shape)[:]

            self.buffer.append(buf_arr)

        self.buffer_size = buffer_size

        self._len = 0
        self._pos = 0

    def __len__(self):
        return self._len

    def append(self, item):
        for i, elem in item:
            self.buffer[i][self._pos] = elem

        self._len = min(self._len + 1, self.buffer_size)
        self._pos += 1
        self._pos %= self.buffer_size

    def get_shared_memories(self):
        return self.shms

    def sample(self, batch_size: int):
        while len(self) == 0:
            pass

        idx = np.random.choice(len(self), size=batch_size)
        return [buf[idx] for buf in self.buffer]

    def close(self):
        for buffer in self.buffer:
            buffer.close()
            buffer.unlink()

# class ReplayBuffer:
#     def __init__(self, capacity: int) -> None:
#         self.capacity = capacity
#         self.buffer = collections.deque(maxlen=capacity)

#     def __len__(self) -> None:
#         return len(self.buffer)

#     def append(self, experience: Experience) -> Tuple:
#         self.buffer.append(experience)

#     def sample(self, batch_size: int) -> Tuple:
#         return random.choices(self.buffer, k=batch_size)



# class ReplayBuffer:
#     def __init__(self, capacity: int) -> None:
#         self.buffer = collections.deque(maxlen=capacity)

#     def __len__(self) -> None:
#         return len(self.buffer)

#     def append(self, experience: Experience) -> Tuple:
#         self.buffer.append(experience)

#     def merge(self, items):
#         if len(items) == 1:
#             return items[0]

#         if isinstance(items[0], tuple):
#             return tuple(self.merge(field) for field in zip(*items))
#         elif isinstance(items[0], torch.Tensor):
#             return torch.cat(items, axis=0)
#         elif isinstance(items[0], list):
#             return list(functools.reduce(operator.add, items))
#         else:
#             return np.array(items)

#     # def merge(self, item, prefetch=False):
#     #     if len(item) == 1:
#     #         return item[0]

#     #     if isinstance(item[0], tuple):
#     #         return tuple(self.merge(i, prefetch=prefetch) for i in zip(*item))
#     #     elif isinstance(item[0], torch.Tensor):
#     #         if prefetch:
#     #             return torch.cat(item, axis=0).pin_memory()
#     #         else:
#     #             return torch.cat(item, axis=0)
#     #     elif isinstance(item[0], list):
#     #         return list(reduce(add, item))
#     #     else:
#     #         return list(item)

    # def sample(self, batch_size: int, merge: bool = False) -> Tuple:
    #     sampled = random.choices(self.buffer, k=batch_size)
    #     if not merge:
    #         return sampled

    #     return self.merge(sampled)
    #     # states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

    #     # return (np.array(states),
    #     #         np.array(actions),
    #     #         np.array(rewards, dtype=np.float32),
    #     #         np.array(dones, dtype=np.bool),
    #     #         np.array(next_states))



class ExplorerProcess(torch.multiprocessing.Process):
    def __init__(self, in_queue, shms, dtypes, shapes, env_create_fn):
        super(ExplorerProcess, self).__init__()

        self.env = env_create_fn()
        self.buffer = [
            np.ndarray(shape, buffer=shm.buf, dtype=dtype)
            for shm, shape, dtype in zip(shms, shapes, dtypes)
        ]

        self.transitions = []
        self.in_queue = in_queue
        self.stop = False

    def run(self):
        ExplorerIOThread(self).start()

        state = self.env._obs()
        while not self.stop:
            if len(self.transitions) >= 100:
                continue

            action = self.env.action_space.sample()
            next_state, _, reward, done = self.env.step(action)

            self.transitions.append((*state, action, reward, done, *next_state))
            if done:
                self.env.reset()


class ExplorerIOThread(threading.Thread):
    def __init__(self, process):
        super(ExplorerIOThread, self).__init__()
        self.process = process

    def run(self):
        while True:
            if not self.process.in_queue.empty():
                ins = self.process.in_queue.get()
                if ins is None:
                    self.stop = True
                    break

                idx_start, items = ins

                for idx in range(idx_start, idx_start + items):
                    while len(self.process.transitions) == 0:
                        pass

                    transition = self.process.transitions.pop(0)
                    for i, item in enumerate(transition):
                        self.process.buffer[i][idx] = item

                self.process.in_queue.put((True, True))
# class Agent:
#     def __init__(self, env_create_fn: Callable):
#         self.env = env_create_fn()

#     def get_action(self, state):
#         return self.env.action_space.sample()

# class Explorer:
#     def __init__(self, env_create_fn: Callable, buffer_size: int, num_workers: int = 0):
#         process_buffer_size: int = buffer_size // num_workers
#         self.num_workers = num_workers
#         self.n_samples_queues = [Queue() for _ in range(num_workers)]
#         self.result_queue = Queue()

#         self.workers = [
#             ExplorerProcess(self.n_samples_queues[i], self.result_queue, Agent(env_create_fn), process_buffer_size)
#             for i in range(num_workers)
#         ]

#         for worker in self.workers:
#             worker.start()

#     def sample_from_worker(self, idx, n_samples):
#         self.n_samples_queues[idx].put(n_samples)

# class RLDataset(IterableDataset):
#     def __init__(self, explorer: Explorer, sample_size: int = 200) -> None:
#         self.sample_size = sample_size
#         self.explorer = explorer
#         self.samples = []

#     def _spawn_thread(self):
#         def fn():
#             while True:
#                 if not self.explorer.result_queue.empty():
#                     self.samples.extend(self.explorer.result_queue.get())
#                     print('received something', len(self.samples))

#         self.thread = threading.Thread(target=fn)
#         self.thread.start()

#     def __iter__(self) -> Tuple:
#         self._spawn_thread()

#         prop = np.random.random(self.explorer.num_workers)
#         prop /= prop.sum()

#         n_samples = np.floor(prop * self.sample_size).astype(np.int)
#         n_samples[-1] = self.sample_size - n_samples[:-1].sum()

#         for i, n in enumerate(n_samples):
#             self.explorer.sample_from_worker(i, n)

#         n_samples = 0
#         while n_samples != self.sample_size:
#             if len(self.samples) != 0:
#                 n_samples += 1
#                 yield self.samples.pop(0)

#         self.thread.join()

class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200):
        self.sample_size = sample_size
        self.buffer = buffer

    def __iter__(self) -> Tuple:
        samples = self.buffer.sample(self.sample_size)

        for i in range(self.sample_size):
            yield tuple(item[i] for item in samples)

# class RLDataset(IterableDataset):
#     def __init__(self, env, sample_size: int=200):
#         self.sample_s

def train_dataloader(buffer: ReplayBuffer, max_ep_steps: int,
                     batch_size: int = 64, num_workers: int = 4) -> DataLoader:
    """"a"""
    dataset = RLDataset(buffer, max_ep_steps * batch_size // max(num_workers, 1))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    return dataloader

# class explorer_process(env_create_fn, out_queue):
#     env = env_create_fn()
#     print(env)

#     state = env.reset()
#     while True:
#         action = env.action_space.sample()
#         next_state, _, reward, done = env.step(action)

#         out_queue.put(Experience(*state, action, reward, done, *next_state))
#         print('put')
#         state = next_state

#         if done:
#             state = env.reset()

def process_thread(buffer, queues, num_workers):
    current_worker = 0

    while True:
        idx = buffer._pos
        n_items = min(buffer._pos + CHUNK_SIZE, buffer.buffer_size) - buffer._pos

        queues[current_worker].put((idx, n_items))

        while queues[current_worker].empty():
            pass

        queues[current_worker].get()

        buffer._len += n_items
        buffer._pos += n_items
        buffer._pos %= buffer.buffer_size

        if buffer._len > buffer.buffer_size:
            buffer._len = buffer.buffer_size

        current_worker += 1
        current_worker %= num_workers


def create_pnet(*args, **kwargs) -> BasePolicyNet:
    return TrPolicyNet(*args, **kwargs).to(kwargs['device'])


def create_qnet(*args, **kwargs) -> BaseNet:
    return TrTwinQNet(*args, **kwargs).to(kwargs['device'])


if __name__ == "__main__":
    am = ArgsManager()
    kwargs = am.parse()

    num_workers = 2
    buffer_size = 1_000
    env_create_fn = lambda: gym.make('StopSkip-v1')
    # explorer = Explorer(env_create_fn, buffer_size=100_000, num_workers=2)

    # Initialize buffer
    env = gym.make('StopSkip-v1')
    state = env.reset()
    action = env.action_space.sample()
    next_state, _, reward, done = env.step(action)

    transition = (
        *state,
        np.array([[action]], dtype=np.uint8),
        np.array([[reward]], dtype=np.float32),
        np.array([[done]], dtype=np.bool),
        *next_state
    )

    dtypes = [item.dtype.type for item in transition]
    shapes = [item.shape for item in transition]
    buffer = ReplayBuffer(dtypes, shapes)

    # buffers = [ReplayBuffer(100) for _ in range(4)]
    # buffer = ReplayBuffer(100_000)
    # buffer_queue = Queue()

    shapes = [(shape[0] * buffer_size, *shape[1:]) for shape in shapes]
    workers = []
    in_queues = []
    for _ in range(num_workers):
        queue = Queue()
        worker = ExplorerProcess(
            queue,
            buffer.get_shared_memories(),
            dtypes,
            shapes,
            env_create_fn
        )
        worker.start()

        workers.append(worker)
        in_queues.append(queue)

    thread = threading.Thread(target=process_thread, args=(buffer, in_queues, num_workers))
    thread.start()

    time.sleep(5)


    # # spwan processes to populate buffer

    # for i in range(4):
    # state = env.reset()
    # done = False
    # while len(buffer) != 100:
    #     action = env.action_space.sample()
    #     next_state, _, reward, done = env.step(action)

    #     buffer.append(Experience(*state, action, reward, done, *next_state))
    #     state = next_state

    #     if done:
    #         state = env.reset()

    dataloader = train_dataloader(buffer, 200, 128, num_workers=2)
    env_dim = state_action_dims(env)
    policy = create_pnet(*env_dim, env.pos_enc_dim, **kwargs)
    critic = create_qnet(*env_dim, env.pos_enc_dim, **kwargs)

    agent = DSACAgent(env, policy, critic, **kwargs)

    t = time.time()
    n_batches = 0
    for i in range(10):
        print(f'Epoch {i}')
        for idx_batch, batch in enumerate(dataloader):
            print(idx_batch)

            state = (batch[:3])
            action = batch[3]
            reward = batch[4]
            next_state = batch[6:]
            done = batch[5]

            batch_size = state[0].shape[0]
            state = (state[0], state[1], agent.env._edge_indices.repeat(batch_size, 1, 1), state[2])
            next_state = (next_state[0], next_state[1], agent.env._edge_indices.repeat(batch_size, 1, 1), next_state[2])

            state = tuple(item.to(agent.qnet.device) for item in state)
            next_state = tuple(item.to(agent.qnet.device) for item in next_state)

            done = done.int().to(agent.qnet.device)
            reward = reward.to(agent.qnet.device)
            action = action.to(agent.qnet.device)

            batch = (state, action, reward, next_state, done)

            q1_loss, q2_loss = agent._calc_q_loss(*batch)
            agent.qnet.backward_with_loss(q1_loss + q2_loss)
            agent._soft_update_target(agent.qnet, agent.qnet_target)

            p_loss, entropy = agent._calc_policy_loss(*batch)
            agent.pnet.backward_with_loss(p_loss)

            a_loss = agent._calc_alpha_loss(*batch)
            agent.temp.backward_with_loss(a_loss)

            n_batches += 1

    t = (time.time() - t)
    sps = (n_batches * 128) / t
    print(f'Sampling rate: {sps:.1f}/s')

    for queue in in_queues:
        queue.put(None)

    [w.join() for w in workers]
    thread.join()

    buffer.close()
