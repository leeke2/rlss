"""Summary
"""
import torch
from torch.nn import functional as F
import numpy as np
from .base_agent import BaseAgent
from ..nets.base import BaseNet
from ..utils import state_action_dims
from ..utils.base import BaseClass
from ..utils.logger import InfoType
from ..nets import Temperature
import time


class DSACAgentPar(BaseAgent):

    """Summary

    Attributes:
        entropy_target (TYPE): Description
        env (TYPE): Description
        pnet (TYPE): Description
        qnet (TYPE): Description
        qnet_target (TYPE): Description
        state (TYPE): Description
        temp (TYPE): Description
    """

    def __init__(self, env, pnet: BaseNet, qnet: BaseNet, **kwargs):
        """Summary

        Args:
            env (TYPE): Description
            **kwargs: Description
        """
        super(DSACAgentPar, self).__init__(**kwargs)

        self._REQUIRED_PARAMS = [
            'graph_transformer_encoder', 'max_steps_per_episode', 'batch_size', 'gamma', 'tau',
            'steps_before_updating', 'save_interval', 'update_interval', 'random_sampling_steps']
        BaseClass.__init__(self, **kwargs)

        self.env = env
        self.pnet = pnet
        self.qnet = qnet
        self.qnet_target = self.qnet.target_copy()

        _, _, out_dim = state_action_dims(env)

        self.temp = Temperature(**kwargs).to(self.device)
        self.entropy_target = -0.98 * np.log(1 / out_dim)

        self._memory_populated = False
        self.kwargs = kwargs

    def _log(self, *args, **kwargs):
        """Summary

        Args:
            *args: Description
            **kwargs: Description
        """
        self._logger.log(*args, **kwargs)

    def _gather_experience(self, eval_ep=False) -> None:
        """Summary

        Args:
            eval_ep (bool, optional): Description

        Returns:
            None: Description
        """
        action, _ = self._get_action(self.state)
        new_state, _, reward, done = self.env.step(action)

        self._memory.push((self.state, action, reward, new_state, done), i_episode=self._logger.i_episode)
        self.state = new_state

        self._log(InfoType.Step.StepReward, reward)
        self._log(InfoType.Episode.EpDone, done)
        self._log(InfoType.Episode.BestObj, self.env.best)

        return done

    def _calc_q_loss(self, state, action, reward, next_state, done):
        """Summary

        Args:
            state (TYPE): Description
            action (TYPE): Description
            reward (TYPE): Description
            next_state (TYPE): Description
            done (TYPE): Description

        Returns:
            TYPE: Description
        """
        q1, q2 = self.qnet(*state)

        with torch.no_grad():
            _, (probs, logprobs) = self._get_action(next_state)
            q1_next, q2_next = self.qnet_target(*next_state)
            q = torch.min(q1_next, q2_next)

            v = (probs * (q - self.alpha * logprobs)).sum(dim=-1).unsqueeze(-1)
            backup = reward + self.gamma * (1 - done) * v

        q1 = q1.gather(1, action.long())
        q2 = q2.gather(1, action.long())

        loss1 = F.mse_loss(q1, backup)
        loss2 = F.mse_loss(q2, backup)

        return loss1, loss2

    def _calc_policy_loss(self, state, action, reward, next_state, done):
        """Summary

        Args:
            state (TYPE): Description
            action (TYPE): Description
            reward (TYPE): Description
            next_state (TYPE): Description
            done (TYPE): Description

        Returns:
            TYPE: Description
        """
        q1, q2 = self.qnet(*state)
        q = torch.min(q1, q2)

        _, (probs, logprobs) = self._get_action(state)
        loss = (probs * (self.alpha * logprobs - q)).sum(dim=-1).mean()
        entropy = -(probs * logprobs).sum(dim=-1).mean()

        return loss, entropy

    def _calc_alpha_loss(self, state, action, reward, next_state, done):
        """Summary

        Args:
            state (TYPE): Description
            action (TYPE): Description
            reward (TYPE): Description
            next_state (TYPE): Description
            done (TYPE): Description

        Returns:
            TYPE: Description
        """
        _, (probs, logprobs) = self._get_action(state)
        entropy = -(probs * logprobs).sum(dim=-1)

        return (self.log_alpha * (entropy - self.entropy_target).detach()).mean()

    def _update(self) -> None:
        """Summary
        """
        print('update')
        t = time.time()
        batch, stats = self._memory.sample(self.batch_size)
        t = time.time() - t
        print(f'sample done: {t:.1f}s')
        # Update critic

        t = time.time()
        q1_loss, q2_loss = self._calc_q_loss(*batch)
        t = time.time() - t
        print(f'calc_q_loss: {t:.1f}s')

        t = time.time()
        self.qnet.backward_with_loss(q1_loss + q2_loss)
        t = time.time() - t
        print(f'update_qnet: {t:.1f}s')
        self._soft_update_target(self.qnet, self.qnet_target)
        t = time.time() - t

        # Update actor
        # self.qnet.freeze()
        t = time.time()
        p_loss, entropy = self._calc_policy_loss(*batch)
        self.pnet.backward_with_loss(p_loss)
        t = time.time() - t
        print(f'update_policy: {t:.1f}s')

        t = time.time()
        a_loss = self._calc_alpha_loss(*batch)
        self.temp.backward_with_loss(a_loss)
        t = time.time() - t
        print(f'update_alpha: {t:.1f}s')
        # self.qnet.unfreeze()

        self._log(InfoType.Step.Update)
        # self._log(InfoType.Step.BatchRecency, stats['batch_recency'])
        # self._log(InfoType.Step.BatchRepeated, stats['batch_repeated'])
        # self._log(InfoType.Step.ExperienceUtilization, stats['experience_utilization'])
        self._log(InfoType.Step.LossQ1, q1_loss.item())
        self._log(InfoType.Step.LossQ2, q2_loss.item())
        self._log(InfoType.Step.LossPolicy, p_loss.item())
        self._log(InfoType.Step.LossAlpha, a_loss.item())
        self._log(InfoType.Step.Entropy, entropy.item())
        self._log(InfoType.Step.Alpha, self.alpha.item())

    def train(self, n_episodes=10):
        """Summary

        Args:
            n_episodes (int, optional): Description
        """
        self._register_callbacks()

        for _ in self._logger.ep_counter(n_episodes):
            # self.state = self.env.reset()

            for _ in self._logger.step_counter(self.max_steps_per_episode):
                if not self._memory_populated:
                    while self._memory.position < self.random_sampling_steps:
                        print(self._memory.position)
                        time.sleep(1)
                        continue

                    self._memory_populated = True
                    print('here')
                # done = self._gather_experience()

                # if done:
                #     self._logger.episode_done()
                #     break
