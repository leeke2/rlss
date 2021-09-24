import numpy as np
import gym
from gym import spaces
# from gym.utils import seeding
from .reward_njit import *
from numba import types
from numba.typed import Dict
import torch
import math
from .utils import seed_all
from typing import Optional
from enum import Enum


class StopSkippingEnvActions(Enum):
    Terminate = 0


class StopSkippingEnvAllocationActions(Enum):
    Add = 0
    Remove = 1


class StopSkipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_nodes: int = 15,
                 n_buses: int = 10,
                 min_travel_time: int = 3,
                 max_travel_time: int = 15,
                 max_perturb: float = 0.3,
                 seed: int = 9112,
                 allow_node_removal: bool = True,
                 allow_allocation_removal: bool = True,
                 enforce_order: bool = False,
                 tensor: bool = True,
                 var_tt: bool = False,
                 var_nodes: bool = False,
                 ttmat: Optional[torch.Tensor] = None,
                 ignore_structure: bool = False,
                 done_reward: float = 0.0,
                 incremental_reward_step: float = 0.1,
                 incremental_reward: float = 0.0):
        """
        An express service design problem given an existing alignment and fleet size. This
        environment considers the case where buses travel only in a single direction. The base route
        serves all stops on the corridor.

        Observation:
            Type: Tuple(Box(n_links), Box(n_links), Box(n_nodes), Box(1))
            #   #elems      Observation                     Min     Max
            0   n_links     OD demands (flattened triu)     0       1
            1   n_links     Travel time (flattened triu)    0       1
            2   n_nodes     Express alignment               0       1
            3   1           Express allocation              0       1

        Actions:
            Type: Discrete(n_nodes + 2) or Discrete(n_nodes + 3)
            #               Action
            0               Terminate
            [1, n_nodes]    Select / deselect (if allow_node_removal is True) node
            n_nodes + 1     Add bus to express route
            n_nodes + 2     Remove bus from express route (if allow_allocation_removal is True)

            Note: The environment considers unidirectional route, thus the nodes selected are sorted
            before calculating the rewards.

        Reward:
            Reward is -0.001 for each step taken (penalty) plus the objective difference compared to
            the previous state. The termination step does not incur a -0.001 penalty.

        State:
            Travel time is invariant across states sampled from the environment, and is normalized
            by the maximum link travel time. OD demands are sampled randomly such that the OD
            demands sum to 1.

        Episode termination:
            Action 0 is selected.
            Number of buses allocated to the express route exceeds permissible range.
            Node repeated (when allow_node_removal is False).
        """

        self.n_nodes = n_nodes
        self.n_buses = n_buses
        # self.max_steps = max_steps
        self.allow_node_removal = allow_node_removal
        self.allow_allocation_removal = allow_allocation_removal
        self.enforce_order = enforce_order
        self.tensor = tensor
        self.var_tt = var_tt
        self.var_nodes = var_nodes
        self.min_travel_time = min_travel_time
        self.max_travel_time = max_travel_time
        self.max_perturb = max_perturb
        self.ignore_structure = ignore_structure
        self.done_reward = done_reward
        self.incremental_reward = incremental_reward
        self.incremental_reward_step = incremental_reward_step

        n_links = self.n_nodes * (self.n_nodes - 1) // 2
        self.action_space = spaces.Discrete(self.n_nodes + 2 + self.allow_allocation_removal)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(n_links, ), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(n_links, ), dtype=np.float32),
            spaces.Box(low=0, high=self.n_nodes, shape=(self.n_nodes, ), dtype=np.uint),
            spaces.Box(low=0, high=1, shape=(1, ), dtype=np.uint))
        )

        # Always seed before generating ttmat
        self.seed(seed)
        if ttmat is None:
            self.travel_time = self._ttmat(n_nodes, min_travel_time, max_travel_time, max_perturb, var_nodes, ignore_structure)
            self.ttmat = None
        else:
            self.travel_time = ttmat
            self.ttmat = ttmat

        self.existing_allocation = np.array([[self.n_buses]])
        self.base_alignment = np.zeros((2, self.n_nodes, self.n_nodes))
        self.base_alignment[0] = np.eye(self.n_nodes)

        # Caching variables for calculation of objective
        self.triu_indices = np.triu_indices(self.n_nodes, k=1)
        self.log_costs = Dict.empty(
            key_type=types.UniTuple(types.int64, 2),
            value_type=types.double[:, :]
        )

        self.base_objective = None
        self.reset()

    def calculate_base_objective(self):

        allocation = np.array([[[self.allocation]]])
        alignment = np.zeros((1, 1, self.n_nodes), dtype=np.uint8)
        if len(self.alignment) >= 2:
            alignment[0, 0, :len(self.alignment)] = self.alignment

        if self.allocation == 0:
            alignment *= 0

        objective, log_costs, seq_ids, alc_ids, frequencies = reward_njit(
            alignment,
            allocation,
            demands=np.expand_dims(self.demands, axis=0),
            existing_allocation=self.existing_allocation,
            log_costs=self.log_costs,
            base_alignment=self.base_alignment,
            ttmat=self.travel_time,
            unidirectional=True)

        if self.best > objective[0]:
            self.best = objective[0]

        return objective[0]

    def calculate_objective(self, allocation, alignment):
        objective, _, _, _, _ = reward_njit(
            alignment,
            allocation,
            demands=np.expand_dims(self.demands, axis=0),
            existing_allocation=self.existing_allocation,
            log_costs=self.log_costs,
            base_alignment=self.base_alignment,
            ttmat=self.travel_time,
            unidirectional=True)

        return objective

    def reset(self, demands=None):
        if demands is not None:
            self.demands = demands
        else:
            self.demands = self._rmat(sum_to_one=True)

        self.alignment = [i for i in range(1, self.n_nodes + 1)]
        self.allocation = 1
        self.n_steps = 0

        self.last_reward = 0
        self.last_objective = self.calculate_base_objective()
        self.last_action = ' '

        self.best = self.last_objective
        self.last_incremental_reward_obj = 1

        if self.var_tt:
            if self.ttmat is None:
                self.travel_time = self._ttmat(
                    self.n_nodes,
                    self.min_travel_time,
                    self.max_travel_time,
                    self.max_perturb,
                    self.var_nodes,
                    self.ignore_structure)
            else:
                perturb = (np.random.rand(self.n_nodes, self.n_nodes) - 0.5) * 2 * self.max_perturb
                self.travel_time = self.ttmat + perturb * (self.ttmat > 0) * self.ttmat
                self.travel_time = self.travel_time / self.travel_time.max()

            self.log_costs = Dict.empty(
                key_type=types.UniTuple(types.int64, 2),
                value_type=types.double[:, :]
            )

        return self._obs()

    def execute(self, action):
        if action == 0:
            # step_reward += 1
            return True
        elif action < self.n_nodes + 1:
            # Change alignment
            if self.alignment == []:
                self.alignment.append(action)
            elif action not in self.alignment:
                if self.enforce_order:
                    if action > self.alignment[-1]:
                        self.alignment.append(action)
                    # else:
                    #     done = True
                else:
                    self.alignment.append(action)
                    self.alignment = sorted(self.alignment)
            elif self.allow_node_removal and action in self.alignment:
                self.alignment.remove(action)
            # elif action in self.alignment:
            #     done = True
        else:
            cmd = action - self.n_nodes - 1
            if cmd == BUSADD:
                if self.allocation < self.n_buses - 1:
                    self.allocation += 1
                # else:
                    # done = True
            elif cmd == BUSREM:
                if self.allocation > 0:
                    self.allocation -= 1
                # else:
                    # done = True

        return False

    def _get_partial_solution(self):
        allocation = np.array([[[self.allocation]]])
        alignment = np.zeros((1, 1, self.n_nodes), dtype=np.uint8)

        if self.allocation == 0:
            return allocation, alignment

        if len(self.alignment) >= 2:
            alignment[0, 0, :len(self.alignment)] = self.alignment

        return allocation, alignment

    def step(self, action, evaluate=True):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # assert self.n_steps <= self.max_steps, '#steps exceeded'
        step_reward = 0

        done = self.execute(action)

        if evaluate:
            allocation, alignment = self._get_partial_solution()
            objective, log_costs, seq_ids, alc_ids, frequencies = reward_njit(
                alignment,
                allocation,
                demands=np.expand_dims(self.demands, axis=0),
                existing_allocation=self.existing_allocation,
                log_costs=self.log_costs,
                base_alignment=self.base_alignment,
                ttmat=self.travel_time,
                unidirectional=True)

            if self.best > objective[0]:
                self.best = objective[0]

            objective_diff = self.last_objective - objective[0]

            step_reward += objective_diff

            self.last_reward = step_reward
            self.last_objective = objective[0]
            self.log_costs = log_costs

        self.last_action = action
        self.n_steps += 1

        if done or action == StopSkippingEnvActions.Terminate:
            step_reward += self.reward_termination

        step_reward += self.reward_step * (not done)

        return self._obs(), step_reward, done, None

    def render(self, mode='human', close=False):
        line = ['○' if i not in self.alignment else '●' for i in range(1, self.n_nodes + 1)]
        line = ' '.join([c for c in line])

        last_objective = self.last_objective or 0
        if last_objective == self.best:
            obj_str = f'\033[1;30;47m{last_objective:6.3f}\033[0m'
        else:
            obj_str = f'{last_objective:6.3f}'

        out = (f'> {str(self.last_action):>3} | '
               f'Reward: {self.last_reward:6.3f} | '
               f'Objective: {obj_str} | '
               f'{self.allocation}x | '
               f'{line}')

        print(out, end='\n')

    def seed(self, seed=None):
        seed = seed or 0

        seed_all(seed)
        return [seed]

        # if seed is None:
        #     seed = 0

        # torch.manual_seed(seed)
        # self.np.random, seed = seeding.np.random(seed)
        # return [seed]

    def _pad(self, x):
        return x + [0] * (self.n_nodes - len(x))

    def _obs(self):
        x, y = self.triu_indices

        exp_allocation = (self.allocation + 1) / (self.n_buses + 1)
        state = np.concatenate((
            self.demands[x, y],
            self.travel_time[x, y],
            self._pad([x / self.n_nodes for x in self.alignment]),
            [exp_allocation]))

        if not self.tensor:
            return state

        alignment = torch.from_numpy(self.base_alignment).transpose(2, 0).transpose(0, 1) * (1 - exp_allocation)

        if len(self.alignment) > 0:
            exp_alignment = torch.tensor(self.alignment).int() - 1
            alignment[(*zip(*enumerate(exp_alignment))), 1] = exp_allocation

        state_tensor = torch.zeros((self.n_nodes, self.n_nodes, 4))
        state_tensor[:, :, 0] = torch.from_numpy(self.demands)
        state_tensor[:, :, 1] = torch.from_numpy(self.travel_time)
        state_tensor[:, :, 2:] = alignment
        state_tensor = state_tensor.unsqueeze(0)

        return state_tensor

    def _rmat(self, sum_to_one=False, triu=True, mask=True):
        tt = np.random.rand(self.n_nodes, self.n_nodes)

        if triu:
            tt = np.triu(tt, k=1)

        if mask:
            tt = (self.travel_time > 0) * tt

        if sum_to_one:
            tt = np.where(tt > 1e-20, -np.log(tt + 1e-20), 0)
            tt = tt / np.sum(tt)
        else:
            tt = tt / np.max(tt)

        return tt

    def _ttmat(self, n_nodes, min_travel_time, max_travel_time, max_perturb, var_nodes, ignore_structure):
        def build_mat(vector):
            vector_len = len(vector)
            shape = (vector_len + 1, vector_len + 1)
            out = np.zeros(shape)

            diag = vector
            for i in range(1, vector_len + 1):
                if i > 1:
                    vecs_to_stack = tuple([vector[j:vector_len - i + j + 1] for j in range(i)])
                    diag = np.stack(vecs_to_stack, axis=1).sum(axis=1)

                    perturb_pct = np.clip(np.random.rand(*diag.shape), 0, max_perturb)
                    perturb = diag * perturb_pct
                    diag = diag - perturb

                out += np.diag(diag, k=i)

            if var_nodes:
                n = max(2, np.random.randint(n_nodes + 1))
                out[n:, :] = 0
                out[:, n:] = 0
                
            return out / out.max().max()

        if not ignore_structure:
            shape = (n_nodes - 1, )

            vectors = torch.randint(min_travel_time, max_travel_time, shape).numpy()
            return build_mat(vectors)
        else:
            out = np.triu(np.random.rand(n_nodes, n_nodes), 1)
            out = out / out.max().max()

            return out


