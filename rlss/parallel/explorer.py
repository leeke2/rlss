# pylint: disable=missing-module-docstring

import threading
import itertools
import typing as T
import numpy as np
import gym
import torch
from torch.multiprocessing import Queue, Process
from rlss.nets import BasePolicyNet
from .memory import ReplayMemory
import os
import signal

CHUNK_SIZE = 10

class DispatcherProcess(Process):  # pylint: disable=missing-class-docstring, too-few-public-methods
    def __init__(self, in_queues, out_queues, buffer_size, recreate_buffer_len_fn):
        super().__init__()

        self.in_queues = in_queues
        self.out_queues = out_queues

        self.buffer_size = buffer_size
        self.buffer_len_shm, self.buffer_len = recreate_buffer_len_fn()

    def run(self):  # pylint: disable=missing-function-docstring
        busy = [False for _ in self.in_queues]
        buffer_pos = 0

        queues = itertools.cycle(enumerate(zip(self.in_queues, self.out_queues)))
        for worker_idx, (in_queue, out_queue) in queues:
            if not busy[worker_idx]:
                n_items = min(self.buffer_size - buffer_pos, CHUNK_SIZE)
                in_queue.put((buffer_pos, n_items))

                busy[worker_idx] = True

                buffer_pos += n_items
                buffer_pos %= self.buffer_size
            elif not out_queue.empty():
                out = out_queue.get()

                if out is None:
                    break

                busy[worker_idx] = False
                if self.buffer_len < self.buffer_size:
                    self.buffer_len += min(CHUNK_SIZE, self.buffer_len - self.buffer_size)

class ExplorerProcess(Process): # pylint: disable=missing-class-docstring, too-many-instance-attributes, too-few-public-methods
    def __init__(
        self,
        in_queue: torch.multiprocessing.Queue,
        out_queue: torch.multiprocessing.Queue,
        recreate_buffer_fn: T.Callable[..., T.Callable[[], np.ndarray]],
        recerate_ready_fn: T.Callable[..., T.Callable[[], np.ndarray]],
        create_env_fn: T.Callable[..., T.Any],
        policy: BasePolicyNet,
        random_sampling_steps: int
    ): # pylint: disable=missing-function-docstring, too-many-arguments
    # buffer_shms, buffer_ready_shm, buffer_shapes, buffer_dtypes,

    # def __init__(self, in_queue, out_queue, buffer_shms, buffer_ready_shm, buffer_shapes,
    # buffer_dtypes, create_env_fn, policy): # a]pylint: disable=missing-function-docstring
        super().__init__()

        self.env = create_env_fn()
        self.buffer_shm, self.buffer = recreate_buffer_fn()
        self.buffer_ready_shm, self.buffer_ready = recerate_ready_fn()

        self.transitions = []
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop = False
        self.policy = policy
        self.random_sampling_steps = random_sampling_steps
        self.i_step = 0

        self.io_thread = ExplorerIOThread(self)

    def run(self): # pylint: disable=missing-function-docstring
        self.io_thread.start()

        state = self.env._obs()
        while not self.stop:
            if len(self.transitions) >= CHUNK_SIZE * 20:
                continue

            if self.i_step < self.random_sampling_steps:
                action = self.env.action_space.sample()
            else:
                processed_state = (
                    torch.from_numpy(state[0]),
                    torch.from_numpy(state[1]),
                    self.env._edge_indices.repeat(1, 1, 1),
                    torch.from_numpy(state[2])
                )

                with torch.no_grad():
                    action = self.policy(*processed_state).argmax().item()

            next_state, _, reward, done = self.env.step(action)

            self.transitions.append((*state, action, reward, done, *next_state))
            self.i_step += 1

            state = next_state
            if done:
                self.env.reset()

class ExplorerIOThread(threading.Thread): # pylint: disable=missing-class-docstring
    def __init__(self, process): # pylint: disable=missing-function-docstring
        super().__init__()
        self.process = process

    def run(self): # pylint: disable=missing-function-docstring
        while not self.process.stop:
            if not self.process.in_queue.empty():
                ins = self.process.in_queue.get()

                if ins is None:
                    self.process.stop = True
                    break

                idx_start, items = ins
                self.process.buffer_ready[idx_start:idx_start + items] = False

                for idx in range(idx_start, idx_start + items):
                    while len(self.process.transitions) == 0:
                        pass

                    transition = self.process.transitions.pop(0)
                    for i, item in enumerate(transition):
                        self.process.buffer[i][idx] = item
                    self.process.buffer_ready[idx] = True

                self.process.out_queue.put((True, True))

class Explorer:  # pylint: disable=missing-class-docstring
    def __init__(
        self,
        create_env_fn: T.Callable,
        policy: T.Optional[BasePolicyNet] = None,
        num_workers: int = 2,
        buffer_size: int = 1_000,
        random_sampling_steps: int = 10_000
    ): # pylint: disable=too-many-arguments

        dtypes, shapes = Explorer.get_experience_dtypes_shapes(create_env_fn())
        self.memory = ReplayMemory(dtypes, shapes, buffer_size=buffer_size)

        self.instruction_queues = []
        self.out_queues = []
        self.workers = []

        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        for _ in range(num_workers):
            queues = Queue(), Queue()
            worker = ExplorerProcess(
                *queues,
                self.memory.recreate_buffer_fn,
                self.memory.recerate_ready_fn,
                create_env_fn,
                policy,
                random_sampling_steps
            )

            worker.start()

            self.workers.append(worker)
            self.instruction_queues.append(queues[0])
            self.out_queues.append(queues[1])

        self.dispatcher = DispatcherProcess(
            self.instruction_queues,
            self.out_queues,
            buffer_size,
            self.memory.recreate_buffer_len_fn
        )
        self.dispatcher.start()

        signal.signal(signal.SIGINT, sigint_handler)

    @staticmethod
    def get_experience_dtypes_shapes(env):  # pylint: disable=missing-function-docstring
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

    def join(self):  # pylint: disable=missing-function-docstring
        for queue in self.instruction_queues:
            queue.put(None)

        for queue in self.out_queues:
            while not queue.empty():
                _ = queue.get()

        for queue in self.out_queues:
            queue.put(None)

        for worker in self.workers:
            worker.join()

        self.dispatcher.join()
        self.memory.close()

if __name__ == '__main__':
    pass
