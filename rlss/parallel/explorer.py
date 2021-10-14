# pylint: disable=missing-module-docstring

import threading
import itertools
import typing as T
import numpy as np
import gym
import torch
from torch.multiprocessing import Queue, Process, SimpleQueue
from multiprocessing.shared_memory import SharedMemory
from rlss.nets import BasePolicyNet
from .memory import ReplayMemory
import functools
import operator
import os
import signal
import time

CHUNK_SIZE = 10
SPS_SAMPLING_BUFFER = 30

SharedMemoriesReconstructionFn = T.Callable[..., T.Callable[[], np.ndarray]]
EnvironmentCreationFn = T.Callable[..., T.Any]

class DispatcherProcess(Process):  # pylint: disable=missing-class-docstring, too-few-public-methods
    def __init__(self, in_queues, out_queues, buffer_size, recreate_buffer_len_fn, recreate_sps_fn):
        super().__init__()

        self.in_queues = in_queues
        self.out_queues = out_queues

        self.buffer_size = buffer_size
        self.buffer_len_shm, self.buffer_len = recreate_buffer_len_fn()
        self.sps_shm, self.sps = recreate_sps_fn()

        self.times = []

    def run(self):  # pylint: disable=missing-function-docstring
        self.times.append((time.time(), 0))
        n_samples = 0

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

                n_samples += CHUNK_SIZE
                busy[worker_idx] = False
                if self.buffer_len < self.buffer_size:
                    self.buffer_len += min(CHUNK_SIZE, self.buffer_len - self.buffer_size)

                self.times.append((time.time(), n_samples))
                if len(self.times) > SPS_SAMPLING_BUFFER:
                    self.times = self.times[1:]

                if len(self.times) > 1:
                    self.sps *= 0
                    self.sps += (self.times[-1][1] - self.times[0][1]) / (self.times[-1][0] - self.times[0][0])

        self.buffer_len_shm.close()
        self.sps_shm.close()

class ExplorerProcess(Process): # pylint: disable=missing-class-docstring, too-many-instance-attributes, too-few-public-methods
    def __init__(
        self,
        worker_id: int,
        in_queue: torch.multiprocessing.SimpleQueue,
        out_queue: torch.multiprocessing.SimpleQueue,
        recreate_buffer_fn: T.Callable[..., T.Callable[[], np.ndarray]],
        recerate_ready_fn: T.Callable[..., T.Callable[[], np.ndarray]],
        create_env_fn: T.Callable[..., T.Any],
        policy: BasePolicyNet,
        random_sampling_steps: int,
    ): # pylint: disable=missing-function-docstring, too-many-arguments
    # buffer_shms, buffer_ready_shm, buffer_shapes, buffer_dtypes,

    # def __init__(self, in_queue, out_queue, buffer_shms, buffer_ready_shm, buffer_shapes,
    # buffer_dtypes, create_env_fn, policy): # a]pylint: disable=missing-function-docstring
        super().__init__()

        self.worker_id = worker_id
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
        num_envs: int = 200,
        buffer_size: int = 1_000,
        random_sampling_steps: int = 10_000,
        max_steps_per_episode: int = 200,
        device: str = 'cpu'
    ): # pylint: disable=too-many-arguments

        self.device = device
        dtypes, shapes = Explorer.get_experience_dtypes_shapes(create_env_fn())
        self.memory = ReplayMemory(dtypes, shapes, buffer_size=buffer_size)
        self.buffer_size = buffer_size

        # sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        if device == 'cpu':
            self.instruction_queues = []
            self.out_queues = []
            self.workers = []

            # RolloutWorkers
            for i in range(num_workers):
                queues = SimpleQueue(), SimpleQueue()
                worker = ExplorerProcess(
                    i,
                    *queues,
                    self.memory.recreate_buffer_fn,
                    self.memory.recerate_ready_fn,
                    create_env_fn,
                    policy,
                    random_sampling_steps,
                )

                worker.start()

                self.workers.append(worker)
                self.instruction_queues.append(queues[0])
                self.out_queues.append(queues[1])

            self.dispatcher = DispatcherProcess(
                self.instruction_queues,
                self.out_queues,
                buffer_size,
                self.memory.recreate_buffer_len_fn,
                self.memory.recreate_sps_fn
            )
            self.dispatcher.start()
        else:
            self.ins_queue = Queue()
            self.explorer = CudaExplorerProcess(
                policy,
                self.memory.recreate_buffer_fn,
                self.memory.recreate_buffer_len_fn,
                self.memory.recreate_sps_fn,
                create_env_fn,
                num_workers,
                num_envs,
                max_steps_per_episode,
                buffer_size,
                random_sampling_steps,
                self.ins_queue
            ).start()

        # signal.signal(signal.SIGINT, sigint_handler)

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
        if self.device == 'cpu':
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
        else:
            self.ins_queue.put(None)

        self.memory.close()

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

def get_state_dtypes_shapes(env, batch_size=1):
    state = env.reset()

    dtypes = [item.dtype.type for item in state]
    shapes = [(item.shape[0] * batch_size, *item.shape[1:]) for item in state]

    return dtypes, shapes

class CudaExplorerProcess(Process):
    def __init__(
        self,
        policy: BasePolicyNet,
        buffer_reconstructor,
        buffer_len_reconstructor,
        sps_reconstructor,
        env_fn: EnvironmentCreationFn,
        num_cpu_reward_workers: int,
        num_envs: int,
        max_steps_per_episode: int,
        buffer_size: int,
        random_sampling_steps: int,
        ins_queue
    ):
        super().__init__()

        assert num_cpu_reward_workers < num_envs

        self.env = env_fn()

        self.policy = policy
        self.shms = SharedMemoryManager()
        self.buffer_size = buffer_size
        self.num_envs = num_envs

        self.worker_env_map = []
        self.buffer_len = buffer_len_reconstructor()[1]
        self.sps = sps_reconstructor()[1]
        self.ins_queue = ins_queue

        self._create_shared_memories(env_fn, num_envs, num_cpu_reward_workers)
        self._spawn_cpu_reward_workers(
            self.shms.reconstructors,
            env_fn,
            buffer_reconstructor,
            num_cpu_reward_workers,
            num_envs,
            max_steps_per_episode
        )

        self.i_step = 0
        self.random_sampling_steps = random_sampling_steps

    def _create_shared_memories(
        self,
        env_fn: EnvironmentCreationFn,
        num_envs: int,
        num_cpu_reward_workers: int
    ):

        env = env_fn()
        dtypes, shapes = get_state_dtypes_shapes(env, batch_size=num_envs)

        self.state = self.shms.create('state', shapes, dtypes, init=True)
        self.ready = self.shms.create('ready', (num_cpu_reward_workers, ), np.bool, init=True)

###############################################################################
# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v #

    def _spawn_cpu_reward_workers(
        self,
        shm_reconstructors,
        env_fn: EnvironmentCreationFn,
        buffer_reconstructor,
        num_cpu_reward_workers: int,
        num_envs: int,
        max_steps_per_episode: int
    ):
        self.workers = []
        self.action_queues = []

        for worker_id in range(num_cpu_reward_workers):
            action_queue = Queue()
            env_ids = list(range(worker_id, num_envs, num_cpu_reward_workers))
            self.worker_env_map.append(env_ids)

            worker = CpuRewardWorker(
                worker_id,
                env_fn,
                buffer_reconstructor,
                shm_reconstructors,
                action_queue,
                env_ids,
                max_steps_per_episode
            )
            worker.start()

            self.workers.append(worker)
            self.action_queues.append(action_queue)

    def run(self):
        dcount = torch.cuda.device_count()
        if dcount > 1:
            self.device = f'cuda:{dcount - 1}'
        else:
            self.device = 'cpu'

        print(self.device)
        self.policy.to(self.device)

        times = [time.time()]
        buffer_pos = 0
        while True:
            if not self.ins_queue.empty() and self.ins_queue.get() is None:
                break

            if np.all(self.ready):
                self.buffer_len += min(self.buffer_size - self.buffer_len, self.num_envs)
                self.ready[:] = False

                times.append(time.time())
                if len(times) > SPS_SAMPLING_BUFFER:
                    times = times[1:]

                if len(times) > 1:
                    self.sps *= 0
                    self.sps += (len(times) - 1) * self.num_envs / (times[-1] - times[0])

                if self.i_step < self.random_sampling_steps:
                    self.i_step += self.num_envs

                    actions = [-1] * self.num_envs
                else:
                    state = self.env.process_state(self.state, device=self.device)

                    values = self.policy(*state).cpu()
                    actions = values.argmax(axis=1).long().numpy().tolist()

                worker_envs = zip(self.action_queues, self.worker_env_map)
                for worker_id, (job_queue, env_ids) in enumerate(worker_envs):
                    self.ready[worker_id] = False

                    job_queue.put([
                        (env_id, actions[env_id], (env_id + buffer_pos) % self.buffer_size)
                        for env_id in env_ids
                    ])

                buffer_pos += self.num_envs
                buffer_pos %= self.buffer_size

        for queue in self.action_queues:
            queue.put(None)

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ #
###############################################################################

class CpuRewardWorker(Process):
    def __init__(
        self,
        worker_id: int,
        env_fn: EnvironmentCreationFn,
        buffer_reconstructor,
        shm_reconstructors,
        action_queue: torch.multiprocessing.Queue,
        env_ids: T.List[int],
        max_steps_per_episode: int
    ):

        super().__init__()

        self.envs = {
            env_id: env_fn()
            for env_id in env_ids
        }

        self.steps_remaining = {
            env_id: max_steps_per_episode
            for env_id in env_ids
        }

        self.max_steps_per_episode = max_steps_per_episode
        self.action_queue = action_queue
        self.env_ids = env_ids
        self.worker_id = worker_id

        # Reconstruct state from shared memory for exploration
        self.state = shm_reconstructors['state']()
        self.ready = shm_reconstructors['ready']()

        # Reconstruct buffer from shared memory for storing transitions
        self.buffer = buffer_reconstructor()[1]

    def reset_envs(
        self,
        env_ids: T.Optional[T.Union[T.List[int], int]] = None
    ):

        if env_ids is None:
            env_ids = self.envs.keys()
        elif isinstance(env_ids, int):
            env_ids = [env_ids]

        for env_id in env_ids:
            self.envs[env_id].reset()
            self.steps_remaining[env_id] = self.max_steps_per_episode

    def update_states(
        self,
        env_ids: T.Optional[T.Union[T.List[int], int]] = None
    ):
        if env_ids is None:
            env_ids = self.envs.keys()
        elif isinstance(env_ids, int):
            env_ids = [env_ids]

        for env_id in env_ids:
            for i, item in enumerate(self.envs[env_id].current_state):
                self.state[i][env_id] = item[0][:]

    def set_ready(self):
        self.ready[self.worker_id] = True

    def save_transition(self, transition, buffer_pos):
        for i, item in enumerate(transition):
            self.buffer[i][buffer_pos] = item

    def step_env(self, env_id, action):
        if action == -1:
            action = self.envs[env_id].action_space.sample()

        state = self.envs[env_id].current_state
        next_state, _, reward, done = self.envs[env_id].step(action)

        self.steps_remaining[env_id] -= 1
        done = done or self.steps_remaining[env_id] == 0

        transition = (
            *state,
            np.array([[action]], dtype=np.uint8),
            np.array([[reward]], dtype=np.float32),
            np.array([[done]], dtype=np.bool),
            *next_state
        )

        return transition, done

    def run(self):
        self.reset_envs()
        self.update_states()
        self.set_ready()

        while True:
            if not self.action_queue.empty():
                jobs = self.action_queue.get()

                if jobs is None:
                    break

                for env_id, action, buffer_pos in jobs:
                    transition, done = self.step_env(env_id, action)

                    self.update_states(env_id)
                    self.save_transition(transition, buffer_pos)

                    if done:
                        self.reset_envs(env_id)

                self.set_ready()

def reconstructors_wrapper(reconstructors):
    def wrapped_fn():
        return tuple(fn() for fn in reconstructors)

    return wrapped_fn

def create_shared_memory_wrapper(shape, dtype, shm):
    def wrapped_fn():
        return create_shared_memory(shape, dtype, shm=shm)

    return wrapped_fn

def create_shared_memory(
    shape: T.Union[T.List, T.Tuple],
    dtype: T.Union[T.List, np.dtype],
    shm: T.Optional[T.Union[T.List, SharedMemory]] = None,
    init: bool = False,
):

    if isinstance(shape, list):
        if shm is not None:
            jobs = zip(shape, dtype, shm)
        else:
            jobs = zip(shape, dtype)

        out = tuple(zip(*[
            create_shared_memory(*args, init=init)
            for args in jobs
        ]))

        return (*out[:-1], reconstructors_wrapper(out[-1]))

    if shm is not None:  # create from existing shared memory
        arr = np.ndarray(shape, buffer=shm.buf, dtype=dtype)
        return arr

    num_elem = functools.reduce(operator.mul, shape)
    mem_size = num_elem * np.dtype(dtype).itemsize

    shm = SharedMemory(create=True, size=mem_size)
    arr = np.ndarray(shape, buffer=shm.buf, dtype=dtype)

    if init:
        arr[:] = np.zeros(shape)[:]

    return (shm, arr, create_shared_memory_wrapper(shape, dtype, shm))

class SharedMemoryManager:
    def __init__(self) -> None:
        self.shms = {}

    def create(
        self,
        name,
        shape: T.Union[T.List, T.Tuple],
        dtype: T.Union[T.List, np.dtype],
        init: bool = False
    ) -> T.Tuple[T.Union[T.List, SharedMemory], T.Union[T.List, np.ndarray]]:

        shm, arr, recon_fn = create_shared_memory(
            shape, dtype, init=init
        )

        self.shms[name] = {
            'memoryview': shm,
            'arr': arr,
            'shape': shape,
            'dtype': dtype,
            'recon_fn': recon_fn
        }

        return arr

    @property
    def reconstructors(self):
        return {
            key: self.get_reconstructor(key)
            for key in self.shms
        }

    def get_reconstructor(self, name):
        assert name in self.shms
        return self.shms[name]['recon_fn']

    def get_array(self, name):
        assert name in self.shms
        return self.shms[name]['arr']

    def get_memoryview(self, name):
        assert name in self.shms
        return self.shms[name]['memoryview']

    def join(self):
        for shm in self.shms.values():
            if isinstance(shm['memoryview'], tuple):
                for item in shm['memoryview']:
                    item.close()
                    item.unlink()
            else:
                shm['memoryview'].close()
                shm['memoryview'].unlink()



if __name__ == '__main__':
    pass
