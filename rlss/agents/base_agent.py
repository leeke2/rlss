"""Summary
"""
import typing
from typing import Union
import numpy as np

import torch
from torch import Tensor
from torch.distributions import Categorical
from gym.spaces import Tuple
from torch.multiprocessing import Process

from ..utils.base import BaseClass
from ..utils.replay import ReplayMemory
from ..utils.logger import Logger, EpisodeEntry, StepEntry
from ..utils.logger import CallbackTrigger as CT


class BaseAgent(Process):

    """
    Base implementation for Reinforcement Learning agents
    """

    def __init__(self, **kwargs) -> None:
        """Summary

        Args:
            **kwargs: Description
        """
        super().__init__()

        self._REQUIRED_PARAMS = ['update_print_interval', 'identifier', 'device']
        BaseClass.__init__(self, **kwargs)

    def run(self):
        self._memory = ReplayMemory(device=self.device)
        self._logger = Logger(self.identifier)

        self._log = self._logger.log

        self.start_procedures()

    def start_procedures(self):
        raise NotImplementedError

    @property
    def alpha(self) -> torch.Tensor:
        """Summary

        Returns:
            torch.Tensor: Description
        """
        return self.temp.log_alpha.exp()

    @property
    def log_alpha(self) -> torch.Tensor:
        """Summary

        Returns:
            torch.Tensor: Description
        """
        return self.temp.log_alpha

    @property
    def _random_sampling(self) -> bool:
        """Summary

        Returns:
            bool: Description
        """
        return False
        # return self._logger.i_step < self.random_sampling_steps

    def _soft_update_target(self, net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        """Summary

        Args:
            net (torch.nn.Module): Description
            target_net (torch.nn.Module): Description
        """
        for critic, critic_targ in zip(net.parameters(), target_net.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            critic_targ.data.mul_(1 - self.tau)
            critic_targ.data.add_(self.tau * critic.data)

    def _get_action(
        self,
        state: Union[torch.Tensor, np.ndarray],
        deterministic=False
    ) -> typing.Tuple[int, typing.Tuple[torch.Tensor, torch.Tensor]]:
        """Summary

        Args:
            state (Union[torch.Tensor, np.ndarray]): Description
            deterministic (bool, optional): Description
        """
        if self._random_sampling:
            action = self.env.action_space.sample()
            return action, None

        # if type(state) is not Tensor:
        #     state = Tensor(state)

        if type(state) is tuple:
            state = tuple(el.to(self.device) for el in state)
        else:
            state = state.to(self.device)

        action, probs, log_probs = self.pnet.get_action(state, deterministic=deterministic)

        try:
            action = action.item()
        except ValueError:
            pass

        return action, (probs, log_probs)

    def save(self) -> None:
        """Summary
        """
        torch.save(
            self.pnet.state_dict(),
            f'checkpoints/{self.identifier}_{self._logger.i_step}_pnet.pt'
        )

        torch.save(
            self.qnet.state_dict(),
            f'checkpoints/{self.identifier}_{self._logger.i_step}_qnet.pt'
        )

    def load(self, name: str) -> None:
        """Summary

        Args:
            name (str): Description
        """
        self.pnet.load_state_dict(torch.load(f'checkpoints/{name}_pnet.pt'))
        self.qnet.load_state_dict(torch.load(f'checkpoints/{name}_qnet.pt'))

    def _register_callbacks(self) -> None:
        """Summary
        """
        # Update trigger
        # self._logger.register_callback(
        #     CT.Step.End,
        #     lambda _: self._update(),
        #     interval=self.update_interval,
        #     offset=self.steps_before_updating)

        # Save trigger
        self._logger.register_callback(
            CT.Step.End,
            lambda _: self.save(),
            interval=self.save_interval,
            offset=self.steps_before_updating)

        # Print triggers
        def ep_print(log: EpisodeEntry):
            """Summary

            Args:
                log (EpisodeEntry): Description
            """
            print(
                f'> E {log.i_episode: 7} | '
                f'SRwd: {log.ep_reward:10.3f} | '
                f'BObj: {log.best_obj:10.3f}'
            )

        def update_print(log: StepEntry):
            """Summary

            Args:
                log (StepEntry): Description
            """
            print(
                f'> U {log.i_step:7}                                         '
                f'Q1: {log.loss_q1:10.2f} | '
                f'Q2: {log.loss_q2:10.2f} | '
                f'Pol: {log.loss_p:10.2f} | '
                f'Alpha: {log.loss_alpha:10.2f}')

        self._logger.register_callback(
            CT.Step.End,
            update_print,
            interval=self.update_interval * 50,
            offset=self.steps_before_updating
        )

        # self._logger.register_callback(CT.Episode.End,
        #                               ep_print,
        #                               interval=10,
        #                               offset=self.random_sampling_steps)

    def process_state(self, *args):
        return self.env.process_state(*args, device=self.device)

    def process_batch(self, *args):
        return self.env.process_batch(*args, device=self.device)
