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
import tqdm


class DSACAgent(BaseAgent):

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

    def __init__(self, env, pnet: BaseNet, rollout_pnet: BaseNet, qnet: BaseNet, **kwargs):
        """Summary

        Args:
            env (TYPE): Description
            **kwargs: Description
        """
        super(DSACAgent, self).__init__(**kwargs)

        self._REQUIRED_PARAMS = [
            'graph_transformer_encoder', 'max_steps_per_episode', 'batch_size', 'gamma', 'tau',
            'steps_before_updating', 'save_interval', 'update_interval']
        BaseClass.__init__(self, **kwargs)

        self.env = env
        self.rollout_pnet = rollout_pnet
        self.pnet = pnet
        self.qnet = qnet
        self.qnet_target = self.qnet.target_copy()

        _, _, out_dim = state_action_dims(env)

        self.temp = Temperature(**kwargs).to(self.device)
        self.entropy_target = -0.98 * np.log(1 / out_dim)

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

        # self._memory.push((self.state, action, reward, new_state, done), i_episode=self._logger.i_episode)
        self.state = self.process_state(new_state)

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

    def _update(self, batch) -> None:
        """Summary
        """
        # Update critic
        q1_loss, q2_loss = self._calc_q_loss(*batch)
        self.qnet.backward_with_loss(q1_loss + q2_loss)
        self._soft_update_target(self.qnet, self.qnet_target)

        # Update actor
        # self.qnet.freeze()
        p_loss, entropy = self._calc_policy_loss(*batch)
        self.pnet.backward_with_loss(p_loss)

        a_loss = self._calc_alpha_loss(*batch)
        self.temp.backward_with_loss(a_loss)
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

        self.rollout_pnet.load_state_dict(self.pnet.state_dict())
        print(f'Updated: {torch.sum(list(self.pnet.state_dict().items())[0][1])}')

    def train(self, dataloader, explorer, n_episodes=10):
        """Summary

        Args:
            n_episodes (int, optional): Description
        """
        self._register_callbacks()

        for _ in self._logger.ep_counter(n_episodes):
            self.state = self.process_state(self.env.reset())

            dataloader_iter = iter(dataloader)
            progress_bar = tqdm.tqdm(
                self._logger.step_counter(self.max_steps_per_episode), 
                total=self.max_steps_per_episode,
                ascii=True
            )

            for idx_batch in progress_bar:
                progress_bar.set_postfix(sps=f'{explorer.memory.sps:.1f}', refresh=False)
                batch = next(dataloader_iter)

                self._update(self.process_batch(batch))

                done = self._gather_experience()
                if done:
                    self._logger.episode_done()
                    break

            # for idx_batch, batch in progress_bar:
            #     progress_bar.set_postfix(sps=f'{explorer.memory.sps:.1f}', refresh=False)
            #     self._update(self.process_batch(batch))

            #     done = self._gather_experience()

            #     if done:
            #         self._logger.episode_done()

                    
