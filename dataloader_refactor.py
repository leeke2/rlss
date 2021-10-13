"""s"""
# import typing as T
# import time
# import torch
# import gym
# import envs
# from rlss.agents.sac_discrete import DSACAgent
# from rlss.utils import ArgsManager, state_action_dims
# from rlss.nets import TrTwinQNet, TrPolicyNet, BaseNet, BasePolicyNet
# import torch
# import torch.multiprocessing as mp
# from torch.utils.data import IterableDataset, DataLoader
# from torch.multiprocessing import Queue, Process
# import tqdm
# from rlss.parallel.memory import ReplayMemory
# from rlss.parallel.explorer import Explorer

import typing as T
import time
import torch
import gym
import envs
from torch.multiprocessing import Process
from rlss.agents.sac_discrete import DSACAgent
from torch.utils.data import IterableDataset, DataLoader
from rlss.utils import ArgsManager, state_action_dims
from rlss.nets import TrTwinQNet, TrPolicyNet, BaseNet, BasePolicyNet
from rlss.parallel.memory import ReplayMemory
from rlss.parallel.explorer import Explorer

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
    return TrPolicyNet(*args, **kwargs)

def create_qnet(*args, **kwargs) -> BaseNet: # pylint: disable=missing-function-docstring
    return TrTwinQNet(*args, **kwargs)

class proc(Process):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def run(self):
        self.policy.to('cuda:0')
        while True:
            print('ok')
            time.sleep(5)

if __name__ == "__main__":
    am = ArgsManager()
    kwargs = am.parse()

    create_env_fn = lambda: gym.make('StopSkip-v1')

    kwargs = {
        'embedding_size': 128,
        'n_heads': 4,
        'n_encoder_layers': 2,
        'rollout_workers': 1,
        'dataloader_workers': 1,
        'buffer_size': 100,
        'random_sampling_steps': 10,
        'device': 'cuda',
        'max_steps_per_episode': 200,
        'batch_size': 32,
        'prefetch_factor': 2,
        'identifier': 'test'
    }

    env = create_env_fn()
    env_dim = state_action_dims(env)
    policy = create_pnet(*env_dim, env.pos_enc_dim, **kwargs)
    rollout_policy = create_pnet(*env_dim, env.pos_enc_dim, **kwargs)
    critic = create_qnet(*env_dim, env.pos_enc_dim, **kwargs)

    policy.share_memory()
    rollout_policy.load_state_dict(policy.state_dict())
    rollout_policy.share_memory().eval()

    print(f'rollout_policy: {hash(rollout_policy.state_dict().values)}')
    print(f'policy: {hash(policy.state_dict().values)}')

    explorer = Explorer(
        create_env_fn,
        policy=rollout_policy,
        num_workers=kwargs['rollout_workers'],
        **kwargs
    )

    dataloader = train_dataloader(
        explorer.memory,
        kwargs['max_steps_per_episode'],
        kwargs['batch_size'],
        num_workers=kwargs['dataloader_workers'],
        prefetch_factor=kwargs['prefetch_factor']
    )

    proc(rollout_policy).start()

    # agent = DSACAgent(env, policy, rollout_policy, critic, dataloader, explorer, **kwargs)
    # agent.start()

    # # try:
    # #     # wait for enough steps to be sampled
    # #     while len(explorer.memory) < 20:#kwargs['batch_size']:
    # #         print(f'Current replay memory size: {len(explorer.memory)}')
    # #         time.sleep(1)

    # #     agent.train(dataloader, explorer)

    # #     # progress_bar = tqdm.tqdm(enumerate(dataloader), total=200, ascii=True)
    # #     # for idx_batch, batch in progress_bar:
    # #     #     progress_bar.set_postfix(sps=f'{explorer.memory.sps:.1f}', refresh=False)
            
    # #     #     for idx_batch, batch in enumerate(dataloader):
    # #     #         batch = agent.process_batch(batch)
    # #     #         print(len(batch))
    # #         # print(idx_batch, len(buffer))
            

    # #     #     q1_loss, q2_loss = agent._calc_q_loss(*batch)
    # #     #     agent.qnet.backward_with_loss(q1_loss + q2_loss)
    # #     #     agent._soft_update_target(agent.qnet, agent.qnet_target)

    # #     #     p_loss, entropy = agent._calc_policy_loss(*batch)
    # #     #     agent.pnet.backward_with_loss(p_loss)

    # #     #     a_loss = agent._calc_alpha_loss(*batch)
    # #     #     agent.temp.backward_with_loss(a_loss)

    # #     explorer.join()
    # # except KeyboardInterrupt:
    # #     print('Exiting...')
    # #     explorer.join()


