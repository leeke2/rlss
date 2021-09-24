import typing as T
import numpy as np
from copy import copy
import random

from .base import BaseOptimizer, OptimizerResults
from .instance import StopSkipInstance


"""
Tabu Search Class
https://towardsdatascience.com/optimization-techniques-tabu-search-36f197ef8e25
"""


class TabuSearch:
    def __init__(
            self,
            initialSolution,
            solutionEvaluator,
            neighborOperator,
            isTerminationCriteriaMet,
            acceptableScoreThreshold=None,
            aspirationCriteria=None,
            tabuTenure=20,
            callback=None) -> None:

        self.currSolution = initialSolution
        self.bestSolution = initialSolution
        self.evaluate = solutionEvaluator
        self.aspirationCriteria = aspirationCriteria
        self.neighborOperator = neighborOperator
        self.acceptableScoreThreshold = acceptableScoreThreshold
        self.tabuTenure = tabuTenure
        self.isTerminationCriteriaMet = isTerminationCriteriaMet
        self._callback = callback

        self.oldCost = 1
        self._best = 1

    # def isTerminationCriteriaMet(self):
    #     # can add more termination criteria
    #     return self.evaluate(self.bestSolution) < self.acceptableScoreThreshold \
    #         or self.neighborOperator(self.currSolution) == 0

    def run(self):
        tabuList = {}

        while not self.isTerminationCriteriaMet():
            # get all of the neighbors
            neighbors = self.neighborOperator(self.currSolution)
            # find all tabuSolutions other than those
            # that fit the aspiration criteria
            tabuActions = tabuList.keys()
            # find all neighbors that are not part of the Tabu list
            neighbors = [(n, a) for n, a in neighbors if a not in tabuActions]

            if len(neighbors) != 0:
                # neighbors = filter(lambda n: self.aspirationCriteria(n), neighbors)
                # pick the best neighbor solution
                costs = [self.evaluate(n) for n, a in neighbors]

                minCost = min(costs)

                neighbors = [neighbors[i] for i, c in enumerate(costs) if c == minCost]
                # newSolution = sorted(neighbors, key=lambda n: self.evaluate(n))[0]
                if len(neighbors) != 1:
                    newSolution = random.choice(neighbors)
                else:
                    newSolution = neighbors[0]

                # get the cost between the two solutions
                newCost = minCost
                cost = newCost - self.oldCost
                # if the new solution is better,
                # update the best solution with the new solution

                self.update(newSolution[0], newCost, newSolution[1])

                if cost <= 0:
                    self.oldCost = newCost
                    self.bestSolution = newSolution[0]

                # update the current solution with the new solution
                self.currSolution = newSolution[0]

                tabuList[newSolution[1]] = self.tabuTenure

            # decrement the Tabu Tenure of all tabu list solutions
            toDelete = []
            for action in tabuList:
                tabuList[action] -= 1

                if tabuList[action] == 0:
                    toDelete.append(action)

            for action in toDelete:
                del tabuList[action]
            # add new solution to the Tabu list

        # return best solution found
        return self.bestSolution

    def update(self, *args, **kwargs):
        _, obj, action = args

        if obj < self._best:
            self._best = obj

        if self._callback is not None:
            self._callback(obj, action)


# def _callback(self, model, obj):
#         elapsed = self._elapsed
#         new_obj = min(obj, 1)

#         if elapsed >= self._max_time_budget:
#             model.iterate = 0  # Stop execution
#         else:
#             self._log(elapsed, new_obj, instance=self._i_instance)


class TSOptimizer(BaseOptimizer):
    def __init__(self, max_time_budget: int = 60) -> None:

        super(TSOptimizer, self).__init__(
            max_time_budget=max_time_budget
        )

        self._i_instance = 0

    def _callback(self, obj, action):
        elapsed = self._elapsed
        new_obj = min(obj, 1)

        self._log(
            time=elapsed,
            value=new_obj,
            instance=self._i_instance,
            attempt=1,
            action=action
        )

    @staticmethod
    def _fitness_fn_generator(env):
        def _fitness(x):
            if len(x) < 3:
                return 1

            alignment = x[:-1]
            allocation = x[-1]

            if allocation == 0:
                return 1

            obj = env.calculate_objective(allocation, alignment)
            if np.isnan(obj):
                exit()

            return obj

        return _fitness

    @staticmethod
    def _neighbor_op_generator(env):
        def _neighbors(sol):
            sol = list(sol)
            neighbors = []

            idx_turn_around = [i for i in range(1, len(sol) - 1)
                               if sol[i] - sol[i - 1] < 0]
            idx_turn_around = len(sol) - 1 if len(idx_turn_around) == 0 else idx_turn_around[0]

            for n in range(1, env.n_nodes + 1):
                sol_cp = copy(sol)
                if n in sol_cp[:idx_turn_around]:
                    idx = sol_cp[:idx_turn_around].index(n)

                    if idx == idx_turn_around - 1 and idx_turn_around > 1:
                        sol_cp = sol_cp[:idx - 1] + sol_cp[idx + 1:]
                    else:
                        sol_cp = sol_cp[:idx] + sol_cp[idx + 1:]
                else:
                    if sol_cp[idx_turn_around] == n:
                        continue

                    sol_cp = (
                        sorted(sol_cp[:idx_turn_around] + [n])
                        + sol_cp[idx_turn_around:]
                    )

                neighbors.append((sol_cp, n))

            for n in range(1, env.n_nodes + 1):
                sol_cp = copy(sol)
                if n in sol_cp[idx_turn_around:-1]:
                    idx = sol_cp[idx_turn_around:].index(n)
                    sol_cp = sol_cp[:idx + idx_turn_around] + sol_cp[idx + idx_turn_around + 1:]
                else:
                    if len(sol_cp) == 1:
                        continue

                    if sol_cp[idx_turn_around - 1] == n:
                        continue

                    if sol_cp[idx_turn_around - 1] < n:
                        continue

                    sol_cp = (
                        sol_cp[:idx_turn_around]
                        + sorted(sol_cp[idx_turn_around:] + [n])[::-1]
                    )

                neighbors.append((sol_cp, n + env.n_nodes))

            if sol[-1] > 0:
                sol_cp = copy(sol)
                sol_cp[-1] -= 1

                neighbors.append((sol_cp, env.n_nodes * 2 + 1))

            if sol[-1] < env.n_buses - 1:
                sol_cp = copy(sol)
                sol_cp[-1] += 1

                neighbors.append((sol_cp, env.n_nodes * 2 + 2))

            return neighbors

        return _neighbors

    def optimize(self, env, instances: T.List[StopSkipInstance]) -> OptimizerResults:
        self._env = env

        for i, instance in enumerate(instances):
            self._start()
            self._i_instance = i

            state = self._state_by_injecting(instance)
            if isinstance(state, tuple):
                state = tuple(item.cpu() for item in state)
            else:
                state = state.cpu()

            fitness_fn = TSOptimizer._fitness_fn_generator(self._env)
            neighbor_op = TSOptimizer._neighbor_op_generator(self._env)

            model = TabuSearch(
                [1],
                fitness_fn,
                neighbor_op,
                lambda: not self._proceed,
                callback=self._callback
            )

            model.run()

        return self._results
