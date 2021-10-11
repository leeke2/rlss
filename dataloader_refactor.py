"""s"""
import typing as T
import time
import torch
import gym
import envs
from rlss.agents.sac_discrete import DSACAgent
from rlss.utils import ArgsManager, state_action_dims
from rlss.nets import TrTwinQNet, TrPolicyNet, BaseNet, BasePolicyNet
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.multiprocessing import Queue
import tqdm
from rlss.parallel.memory import ReplayMemory
from rlss.parallel.explorer import Explorer
torch.set_num_threads(1)


class RLDataset(IterableDataset): # pylint: disable=missing-class-docstring, too-few-public-methods
    def __init__(self, memory: ReplayMemory, sample_size: int = 200) -> None: # pylint: disable=missing-function-docstring
        self.sample_size = sample_size
        self.memory = memory

    def __iter__(self) -> T.Tuple: # pylint: disable=missing-function-docstring
        samples = self.memory.sample(self.sample_size)

        for i in range(self.sample_size):
            yield tuple(item[i] for item in samples)

def train_dataloader(
    memory: ReplayMemory,
    max_ep_steps: int,
    batch_size: int = 64,
    num_workers: int = 4,
    prefetch_factor: int = 2
) -> torch.utils.data.DataLoader:
    """"a"""
    dataset = RLDataset(memory, max_ep_steps * batch_size // max(num_workers, 1))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor
    )

    return dataloader

def create_pnet(*args, **kwargs) -> BasePolicyNet: # pylint: disable=missing-function-docstring
    return TrPolicyNet(*args, **kwargs).to(kwargs['device'])


def create_qnet(*args, **kwargs) -> BaseNet: # pylint: disable=missing-function-docstring
    return TrTwinQNet(*args, **kwargs).to(kwargs['device'])


if __name__ == "__main__":
    am = ArgsManager()
    kwargs = am.parse()

    create_env_fn = lambda: gym.make('StopSkip-v1')

    env = create_env_fn()
    env_dim = state_action_dims(env)
    policy = create_pnet(*env_dim, env.pos_enc_dim, **kwargs)
    rollout_policy = create_pnet(*env_dim, env.pos_enc_dim, **kwargs).cpu()
    critic = create_qnet(*env_dim, env.pos_enc_dim, **kwargs)
    agent = DSACAgent(env, policy, critic, **kwargs)

    policy.share_memory()
    rollout_policy.load_state_dict(policy.state_dict())
    rollout_policy.share_memory().eval()

    explorer = Explorer(
        create_env_fn,
        policy=rollout_policy,
        num_workers=kwargs['rollout_workers'],
        buffer_size=kwargs['buffer_size'],
        random_sampling_steps=kwargs['random_sampling_steps']
    )

    dataloader = train_dataloader(
        explorer.memory, 200, kwargs['batch_size'], num_workers=kwargs['dataloader_workers'],
        prefetch_factor=kwargs['prefetch_factor']
    )

    try:
        # wait for enough steps to be sampled
        while len(explorer.memory) < kwargs['batch_size']:
            print(f'Current replay memory size: {len(explorer.memory)}')
            time.sleep(1)

        progress_bar = tqdm.tqdm(enumerate(dataloader), total=200)
        for idx_batch, batch in progress_bar:
        # for idx_batch, batch in enumerate(dataloader):
            # print(idx_batch, len(buffer))
            batch = agent.process_batch(batch)

            q1_loss, q2_loss = agent._calc_q_loss(*batch)
            agent.qnet.backward_with_loss(q1_loss + q2_loss)
            agent._soft_update_target(agent.qnet, agent.qnet_target)

            p_loss, entropy = agent._calc_policy_loss(*batch)
            agent.pnet.backward_with_loss(p_loss)

            a_loss = agent._calc_alpha_loss(*batch)
            agent.temp.backward_with_loss(a_loss)
    except KeyboardInterrupt:
        print('Exiting...')
        explorer.join()