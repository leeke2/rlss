import gym
from gym import spaces
from enum import Enum
from .utils import seed_all
from typing import Optional, List
import random
import math
from .reward_njit import *
import numpy as np
import torch
import itertools


class StopSkipEnvActions(Enum):
    """a"""
    Terminate = 0


class StopSkipEnvAllocationActions(Enum):
    """a"""
    Add = 0
    Remove = 1


class StopSkipEnvDirection(Enum):
    Indeterministic = 0
    Forward = 1
    Backward = 2


class StopSkipEnvNodeMode(Enum):
    __order__ = "NotSelected Forward Backward Both"
    NotSelected = 0
    Forward = 1
    Backward = 2
    Both = 3


class StopSkipEnvNodeModeIterator:
    def __init__(self):
        self._iterator = itertools.cycle(StopSkipEnvNodeMode)
        self._cur = next(self._iterator)

    @property
    def cur(self):
        return self._cur
    
    def next(self):
        self._cur = next(self._iterator)


class StopSkipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        n_nodes: int = 15,
        n_buses: int = 10,
        min_travel_time: int = 3,
        max_travel_time: int = 15,
        max_perturb: float = 0.3,
        seed: int = 9112,
        reward_done: float = 0.0,
        reward_step: float = -0.0001,
        reward_step_nonaction: float = -0.01,
        ttmat: Optional[List[List[float]]] = None,
        var_tt: bool = False,
        n_attempts_per_problem: int = 1
    ) -> None:

        self.n_nodes = n_nodes
        self.n_buses = n_buses

        self.ttmat = ttmat
        self.min_travel_time = min_travel_time
        self.max_travel_time = max_travel_time
        self.max_perturb = max_perturb
        self.var_tt = var_tt
        self.n_attempts_per_problem = n_attempts_per_problem

        self.reward_done = reward_done
        self.reward_step = reward_step
        self.reward_step_nonaction = reward_step_nonaction

        self.action_space = spaces.Discrete(self.n_nodes + 3)
        self.observation_space = None  # TODO: Fix this

        # Initialize travel time matrix
        self.seed_ = seed
        self.seed(seed)
        self.travel_time = ttmat if ttmat is not None else self._ttmat()
        self.pos_enc_dim = min(20, self.n_nodes - 3)

        self.log_costs = {}

        src = torch.arange(n_nodes).repeat(n_nodes)
        dst = torch.arange(n_nodes).repeat_interleave(n_nodes)
        valid = src != dst
        src = src[valid]
        dst = dst[valid]
        self._edge_indices = torch.stack((src, dst))

        self.i_attempt = -1

        self.reset()

    def copy(self):
        return StopSkipEnv(
            n_nodes=self.n_nodes,
            n_buses=self.n_buses,
            min_travel_time=self.min_travel_time,
            max_travel_time=self.max_travel_time,
            max_perturb=self.max_perturb,
            seed=self.seed_,
            reward_done=self.reward_done,
            reward_step=self.reward_step,
            reward_step_nonaction=self.reward_step_nonaction,
            ttmat=self.ttmat,
            var_tt=self.var_tt,
            n_attempts_per_problem=self.n_attempts_per_problem
        )

    def _ttmat(self):
        def add_perturb(vec, perturb):
            return [(a + b) * (1 - min(random.random(), perturb))
                    for a, b in zip(vec[:-1], vec[1:])]

        def build_mat(forward, backward):
            size = len(forward) + 1

            out = [[0 for _ in range(size)] for _ in range(size)]
            for offset in range(1, size):
                if offset != 1:
                    forward = add_perturb(forward, self.max_perturb)
                    backward = add_perturb(backward, self.max_perturb)

                for idx in range(size - offset):
                    out[idx + offset][idx] = backward[idx]
                    out[idx][idx + offset] = forward[idx]

            max_value = max(forward + backward)

            for i in range(size):
                for j in range(size):
                    out[i][j] /= max_value

            return out

        forward = [random.randint(self.min_travel_time, self.max_travel_time)
                   for _ in range(self.n_nodes - 1)]
        backward = [random.randint(self.min_travel_time, self.max_travel_time)
                    for _ in range(self.n_nodes - 1)]

        return build_mat(forward, backward)

    def _rmat(self):
        n_values = self.n_nodes * (self.n_nodes - 1)
        values = [math.log(random.random()) for _ in range(n_values)]
        sum_values = sum(values)

        idx = 0

        out = []
        for i in range(self.n_nodes):
            out.append([])

            for j in range(self.n_nodes):
                if i != j:
                    out[i].append(values[idx] / sum_values)
                    idx += 1
                else:
                    out[i].append(0)

        return out

    def reset_problem(self, demands=None):
        self.demands = demands if demands is not None else self._rmat()
        self.best = 1

        if self.var_tt:
            if self.ttmat is None:
                self.travel_time = self._ttmat()
            else:
                self.travel_time = self.ttmat

                max_value = 0
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i == j:
                            continue

                        self.travel_time[i][j] *= 1 + (random.random() - 0.5) * 2 * self.max_perturb
                        if self.travel_time[i][j] > max_value:
                            max_value = self.travel_time[i][j]

                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        self.travel_time[i][j] /= max_value

        self.calculate_base_objective()

    def reset_solution(self):
        self.exp_allocation = 1
        self.exp_node_modes = [StopSkipEnvNodeModeIterator() for _ in range(self.n_nodes)]

        self.last_objective = 1

    def reset(self, demands=None):
        # if self.i_attempt == self.n_attempts_per_problem - 1 or self.i_attempt == -1:
        #     self.reset_problem(demands=demands)
        #     self.i_attempt = -1

        # self.i_attempt += 1
        self.reset_problem(demands=demands)
        self.reset_solution()

        return self._obs()

    def calculate_objective(self, exp_allocation, exp_alignment):
        alignments = self.base_alignment + [exp_alignment]
        allocations = [self.n_buses - exp_allocation, exp_allocation]

        objective, us, log_costs_new = reward(alignments,
                                              allocations,
                                              self.travel_time,
                                              self.demands,
                                              self.log_costs)

        self.log_costs = log_costs_new
        return objective

    def calculate_base_objective(self):
        alignments = [node for node in range(1, self.n_nodes + 1)]
        alignments = [alignments + alignments[-2::-1]]
        allocations = [self.n_buses]

        self.base_alignment = alignments

        us, log_costs_new = reward(alignments,
                                   allocations,
                                   self.travel_time,
                                   self.demands,
                                   log_costs=self.log_costs)

        self.us = us
        self.log_costs = log_costs_new

    def _obs(self):
        def shape(mat):
            out = ()
            while type(mat) is list:
                out = *out, len(mat)
                mat = mat[0]

            return out

        def flatten(*mats):
            size = shape(mats[0])[0]

            out = []
            for i in range(size):
                for j in range(size):
                    if i != j:
                        out.append([mat[i][j] for mat in mats])

            return out

        def segregate_forward_backward(alignment):
            exp_alignment_forward = []
            exp_alignment_backward = []

            prev_node = None
            for node in alignment:
                if prev_node is not None:
                    if node - prev_node > 0:
                        if exp_alignment_forward == []:
                            exp_alignment_forward.append(prev_node)

                        exp_alignment_forward.append(node)
                    else:
                        if exp_alignment_backward == []:
                            exp_alignment_backward.append(prev_node)

                        exp_alignment_backward.append(node)

                prev_node = node

            return exp_alignment_forward, exp_alignment_backward

        # edge_shape = self.n_nodes * (self.n_nodes - 1), 3
        # node_shape = self.n_nodes, 4 + min(20, self.n_nodes - 1)

        # edge features: demands, travel_time, current_od_travel_time
        edge_features = flatten(self.demands, self.travel_time, self.us)

        # node features: alignment for each direction (frequency), positional encoding
        exp_allocation = self.exp_allocation if len(self.exp_alignment) != 0 else 0
        full_allocation = self.n_buses - exp_allocation

        exp_allocation /= self.n_buses
        full_allocation /= self.n_buses

        exp_alg_for, exp_alg_bac = segregate_forward_backward(self.exp_alignment)

        full_forward = [full_allocation] * self.n_nodes
        full_backward = full_forward
        exp_forward = [exp_allocation if i in exp_alg_for else 0
                       for i in range(1, self.n_nodes + 1)]
        exp_backward = [exp_allocation if i in exp_alg_bac else 0
                        for i in range(1, self.n_nodes + 1)]
        pos_enc = self._lap_pos_enc(self.exp_alignment)

        node_features = list(map(list, zip(full_forward, full_backward, exp_forward, exp_backward)))

        return (np.expand_dims(np.array(node_features, dtype=np.float32), axis=0),
                np.expand_dims(np.array(edge_features, dtype=np.float32), axis=0),
                np.expand_dims(np.array(pos_enc, dtype=np.float32), axis=0))

        # return (torch.Tensor(node_features).unsqueeze(0),
        #         torch.Tensor(edge_features).unsqueeze(0),
        #         self._edge_indices.unsqueeze(0),
        #         torch.Tensor(pos_enc).unsqueeze(0))

    def _lap_pos_enc(self, alignment):
        A = np.zeros((self.n_nodes, self.n_nodes))

        if len(alignment) > 1:
            alignment_mo = np.array(alignment) - 1
            A[alignment_mo[:-1], alignment_mo[1:]] = 1
        # else:
        #     return [
        #         [1 if j - 1 == i else 0
        #          for i in range(self.pos_enc_dim)]
        #         for j in range(self.n_nodes)
        #     ]

        # No need to normalize, always normalized
        # N = np.diag(A.sum(1).clip(1) ** -0.5) #, dtype=float)
        # L = np.eye(self.n_nodes) - N @ A @ N
        L = np.eye(self.n_nodes) - A

        try:
            EigVal, EigVec = np.linalg.eig(L)  # k=self.pos_enc_dim + 1, which='SR', tol=1e-2)
        except:
            print(alignment)
            print(L)
            exit()

        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        EigVec = EigVec[:, EigVal.argsort()]

        return EigVec[:, 1:self.pos_enc_dim + 1].astype(float).tolist()

    def seed(self, seed):
        seed = seed or 0

        seed_all(seed)
        return [seed]

    def execute(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if action == 0:
            return True

        if action > self.n_nodes:
            action = action - self.n_nodes - 1

            if action == StopSkipEnvAllocationActions.Add.value:
                self.exp_allocation = min(self.n_buses - 1, self.exp_allocation + 1)
            elif action == StopSkipEnvAllocationActions.Remove.value:
                self.exp_allocation = max(1, self.exp_allocation - 1)

            return False

        self.exp_node_modes[action - 1].next()

        return False

    @property
    def exp_alignment(self):
        forward = [
            node + 1
            for node in range(self.n_nodes)
            if self.exp_node_modes[node].cur in [
                StopSkipEnvNodeMode.Forward, 
                StopSkipEnvNodeMode.Both
            ]
        ]

        backward = [
            node + 1
            for node in range(self.n_nodes)
            if self.exp_node_modes[node].cur in [
                StopSkipEnvNodeMode.Backward, 
                StopSkipEnvNodeMode.Both
            ]
        ][::-1]

        if len(forward) > 0 and len(backward) > 0 and backward[0] == forward[-1]:
            backward = backward[1:]

        return forward + backward

    def step(self, action):
        step_reward = 0 if action != 0 else self.reward_done

        old_alignment, old_allocation = self.exp_alignment.copy(), self.exp_allocation
        done = self.execute(action)
        changed = (old_alignment != self.exp_alignment) or (old_allocation != self.exp_allocation)

        if not done:
            step_reward += self.reward_step if changed else self.reward_step_nonaction

        if self.exp_allocation == 0 or len(self.exp_alignment) < 2:
            step_reward += self.last_objective - 1
            self.last_objective = 1

            return self._obs(), 1, step_reward, done

        alignments = self.base_alignment + [self.exp_alignment]
        allocations = [self.n_buses - self.exp_allocation, self.exp_allocation]

        objective, us, log_costs_new = reward(alignments,
                                              allocations,
                                              self.travel_time,
                                              self.demands,
                                              self.log_costs)

        self.log_costs = log_costs_new
        self.us = us

        if self.best > objective:
            self.best = objective

        step_reward += self.last_objective - objective
        self.last_objective = objective

        return self._obs(), objective, step_reward, done

    def process_state(self, state, device='cuda'):
        batch_size = state[0].shape[0]
        state = (
            state[0],
            state[1],
            self._edge_indices.repeat(batch_size, 1, 1),
            state[2]
        )

        to_tensor = lambda x: x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        return tuple(to_tensor(item).to(device) for item in state)

    def process_batch(self, batch, device='cuda'):  # pylint: disable=missing-function-docstring
        state = (batch[:3])
        action = batch[3]
        reward = batch[4]
        next_state = batch[6:]
        done = batch[5]

        batch_size = state[0].shape[0]
        state = self.process_state(state, device=device)
        next_state = self.process_state(next_state, device=device)

        done = done.int().to(device)
        reward = reward.to(device)
        action = action.to(device)

        return (state, action, reward, next_state, done)
