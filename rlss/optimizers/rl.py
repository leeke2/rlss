from typing import List

from .base import BaseOptimizer, OptimizerResults
from ..nets import BasePolicyNet
from .instance import StopSkipInstance


class RLOptimizer(BaseOptimizer):
    def __init__(
        self,
        policy: BasePolicyNet,
        max_time_budget: int = 60,
        max_steps_per_episode: int = 200
    ) -> None:

        super(RLOptimizer, self).__init__(
            max_time_budget=max_time_budget
        )

        self._max_steps_per_episode = max_steps_per_episode

        self.policy = policy
        self.device = 'cpu'

    def optimize(self, env, instances: List[StopSkipInstance], max_steps_without_improving=10) -> OptimizerResults:
        self._env = env

        for i, instance in enumerate(instances):
            self._start()

            best = 1
            attempt = 1

            while self._proceed:
                state = self._state_by_injecting(instance)

                if isinstance(state, tuple):
                    state = tuple(item.to(self.device) for item in state)
                else:
                    state = state.to(self.device)
                done = False
                i_step = 0
                plateau_steps = 0

                while not done and i_step < self._max_steps_per_episode and plateau_steps < max_steps_without_improving:
                    action, _, _ = self.policy.get_action(state)
                    action = action.cpu().item()

                    state, _, _, done = self._env.step(action)
                    if isinstance(state, tuple):
                        state = tuple(item.to(self.device) for item in state)
                    else:
                        state = state.to(self.device)
                    i_step += 1

                    elapsed = self._elapsed

                    new_obj = min(self._env.best, 1)
                    if new_obj < best:
                        best = new_obj
                        plateau_steps = 0
                    else:
                        plateau_steps += 1

                    if elapsed >= self._max_time_budget:
                        break

                    self._log(elapsed, best, i, attempt, action)

                attempt += 1

        return self._results

    def to(self, device):
        self.device = device
        self.policy = self.policy.to(device)

        return self
