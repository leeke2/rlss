import csv
import numpy as np
import torch
from typing import Optional
from timeit import default_timer as timer
from .instance import StopSkipInstance


class OptimizerResults:
    def __init__(self):
        self._time = []
        self._value = []
        self._instance = []
        self._action = []
        self._attempt = []

    def log(self, time: float, value: float, instance: int, attempt: Optional[int] = None, action: Optional[int] = None):
        self._time.append(time)
        self._value.append(value)
        self._instance.append(instance)

        if action is not None:
            self._action.append(action)
            self._attempt.append(attempt)

    def to_csv(self, fn):
        with open(fn, 'w+') as f:
            writer = csv.writer(f)

            if len(self._action) != 0:
                data = [self._time, self._value, self._instance, self._attempt, self._action]
            else:
                data = [self._time, self._value, self._instance]

            for row in zip(*data):
                writer.writerow(row)


class BaseOptimizer:
    def __init__(self, max_time_budget: int = 60):
        self._results = OptimizerResults()

        self._start_time = None
        self._max_time_budget = max_time_budget

        self._env = None

        self._log = self._results.log

    def optimize(self, state):
        raise NotImplementedError

    # def _log(self, time: float, value: float, instance: int = 0, action: Optional[int] = None):
    #     self._results.log(time, value, instance, action)

    def _state_by_injecting(self, instance: StopSkipInstance) -> torch.Tensor:
        self._env.reset(demands=instance.demands)
        # self._env.reset()
        # self._env.demands = instance.demands
        
        if instance.has_travel_time:
            self._env.travel_time = instance.travel_time

        return self._env._obs()

    @property
    def _current_time(self):
        return timer()

    @property
    def _elapsed(self):
        return self._current_time - self._start_time

    @property
    def _proceed(self):
        return self._current_time - self._start_time < self._max_time_budget

    def _start(self) -> None:
        self._start_time = timer()
