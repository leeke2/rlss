"""s"""
import collections
import time
import operator
import functools
import itertools
import threading
from typing import Tuple, Callable, Optional
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import torch
import gym
import envs
from rlss.agents.sac_discrete import DSACAgent
from rlss.utils import ArgsManager, state_action_dims
from rlss.nets import TrTwinQNet, TrPolicyNet, BaseNet, BasePolicyNet
from torch.utils.data import IterableDataset, DataLoader
from torch.multiprocessing import Queue
import tqdm

CHUNK_SIZE = 10

class ReplayBuffer: # pylint: disable=missing-class-docstring
    def __init__(self, dtypes, shapes, buffer_size: int = 1_000): # pylint: disable=missing-function-docstring
        self.dtypes = dtypes
        self.shapes = shapes

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

    def added(self, n_items):
        self._len += n_items
        self._pos += n_items

        self._pos %= self.buffer_size
        self._len = min(self._len, self.buffer_size)

    def __len__(self): # pylint: disable=missing-function-docstring
        return self._len

    def append(self, item): # pylint: disable=missing-function-docstring
        for i, elem in item:
            self.buffer[i][self._pos] = elem

        self._len = min(self._len + 1, self.buffer_size)
        self._pos += 1
        self._pos %= self.buffer_size

    def get_shared_memories(self): # pylint: disable=missing-function-docstring
        return self.shms

    def sample(self, batch_size: int): # pylint: disable=missing-function-docstring
        # while len(self) == 0:
        #     pass

        idx = np.random.choice(len(self), size=batch_size)
        return [buf[idx] for buf in self.buffer]

    def close(self): # pylint: disable=missing-function-docstring
        for buffer in self.buffer:
            buffer.close()
            buffer.unlink()


class ExplorerProcess(torch.multiprocessing.Process): # pylint: disable=missing-class-docstring
    def __init__(self, in_queue, buffer, env_create_fn, policy): # a]pylint: disable=missing-function-docstring
        super(ExplorerProcess, self).__init__()
        shms = buffer.get_shared_memories()
        dtypes = buffer.dtypes
        shapes = buffer.shapes
        shapes = [(shape[0] * buffer.buffer_size, *shape[1:]) for shape in shapes]

        self.env = env_create_fn()
        self.buffer = [
            np.ndarray(shape, buffer=shm.buf, dtype=dtype)
            for shm, shape, dtype in zip(shms, shapes, dtypes)
        ]

        self.transitions = []
        self.in_queue = in_queue
        self.stop = False
        self.policy = policy

    def run(self): # pylint: disable=missing-function-docstring
        ExplorerIOThread(self).start()

        state = self.env._obs()
        while not self.stop:
            if len(self.transitions) >= CHUNK_SIZE * 20:
                continue

            # aug_state = (torch.Tensor(state[0]),
            #          torch.Tensor(state[1]),
            #          self.env._edge_indices.repeat(1, 1, 1),
            #          torch.Tensor(state[2]))
            # action, _, _ = self.policy.get_action(aug_state)

            action = self.env.action_space.sample()
            next_state, _, reward, done = self.env.step(action)

            self.transitions.append((*state, action, reward, done, *next_state))
            state = next_state
            if done:
                self.env.reset()


class ExplorerIOThread(threading.Thread): # pylint: disable=missing-class-docstring
    def __init__(self, process): # pylint: disable=missing-function-docstring
        super(ExplorerIOThread, self).__init__()
        self.process = process

    def run(self): # pylint: disable=missing-function-docstring
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

                self.process.in_queue.put((True, len(self.process.transitions)))

class RLDataset(IterableDataset): # pylint: disable=missing-class-docstring
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200): # pylint: disable=missing-function-docstring
        self.sample_size = sample_size
        self.buffer = buffer

    def __iter__(self) -> Tuple: # pylint: disable=missing-function-docstring
        samples = self.buffer.sample(self.sample_size)

        for i in range(self.sample_size):
            yield tuple(item[i] for item in samples)


def train_dataloader(buffer: ReplayBuffer, max_ep_steps: int,
                     batch_size: int = 64, num_workers: int = 4, prefetch_factor: int = 2) -> DataLoader:
    """"a"""
    dataset = RLDataset(buffer, max_ep_steps * batch_size // max(num_workers, 1))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor)

    return dataloader

def dispatcher_thread(buffer, queues, num_workers): # pylint: disable=missing-function-docstring
    for queue in itertools.cycle(queues):
        idx = buffer._pos
        n_items = min(buffer.buffer_size - buffer._pos, CHUNK_SIZE)

        queue.put((idx, n_items))
        while queue.empty(): pass # wait for job to complete
        _, n_transitions = queue.get()

        buffer.added(n_items)

def create_pnet(*args, **kwargs) -> BasePolicyNet: # pylint: disable=missing-function-docstring
    return TrPolicyNet(*args, **kwargs).to(kwargs['device'])


def create_qnet(*args, **kwargs) -> BaseNet: # pylint: disable=missing-function-docstring
    return TrTwinQNet(*args, **kwargs).to(kwargs['device'])

class Explorer:
    def __init__(
        self,
        create_env_fn: Callable,
        policy: Optional[BasePolicyNet] = None,
        num_workers: int = 2,
        buffer_size: int = 1_000
    ): # pylint: disable=too-many-arguments

        dtypes, shapes = self.get_experience_dtypes_shapes(create_env_fn())
        self.buffer = ReplayBuffer(dtypes, shapes, buffer_size=buffer_size)

        self.instruction_queues = []
        self.workers = []

        for _ in range(num_workers):
            queue = Queue()
            worker = ExplorerProcess(queue, self.buffer, create_env_fn, policy)
            worker.start()

            self.workers.append(worker)
            self.instruction_queues.append(queue)

        self.dispatcher = threading.Thread(
            target=dispatcher_thread,
            args=(self.buffer, self.instruction_queues, num_workers)
        )
        self.dispatcher.start()

    def get_experience_dtypes_shapes(self, env):
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

        return dtypes, shapes


    def get_buffer_queues(self): # pylint: disable=missing-function-docstring
        return self.buffer, self.instruction_queues

if __name__ == "__main__":
    am = ArgsManager()
    kwargs = am.parse()

    create_env_fn = lambda: gym.make('StopSkip-v1')

    env = create_env_fn()
    env_dim = state_action_dims(env)
    policy = create_pnet(*env_dim, env.pos_enc_dim, **kwargs)
    rollout_policy = create_pnet(*env_dim, env.pos_enc_dim, **kwargs).cpu()
    critic = create_qnet(*env_dim, env.pos_enc_dim, **kwargs)
    agent = DSACAgent(env, policy, critic, **kwargs)

    policy.share_memory()
    rollout_policy.load_state_dict(policy.state_dict())
    rollout_policy.share_memory().eval()

    explorer = Explorer(create_env_fn, policy=rollout_policy, num_workers=kwargs['rollout_workers'])
    buffer, _ = explorer.get_buffer_queues()
    dataloader = train_dataloader(
        buffer, 200, 64, num_workers=kwargs['dataloader_workers'],
        prefetch_factor=kwargs['prefetch_factor']
    )

    time.sleep(10)

    # t = time.time()
    # n_batches = 0
    # for i in range(10):
    #     print(f'Epoch {i}')
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=200)
    for idx_batch, batch in progress_bar:
        # print(idx_batch, len(buffer))

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

    #     n_batches += 1

    # t = (time.time() - t)
    # sps = (n_batches * 128) / t
    # print(f'Sampling rate: {sps:.1f}/s')