from typing import Optional
from copy import deepcopy

import torch
from torch.nn import Module, Linear, ReLU, Sequential
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from ..utils.base import BaseClass


def ff_net(in_dim: int,
           out_dim: int,
           hidden_dim: int = 256,
           num_layers: int = 2,
           activation: str = 'relu'):

    assert activation == 'relu', 'only ReLU works'
    modules = []

    modules.append(Linear(in_dim, hidden_dim))
    modules.append(ReLU())

    for _ in range(num_layers):
        modules.append(Linear(hidden_dim, hidden_dim))
        modules.append(ReLU())
    modules.append(Linear(hidden_dim, out_dim))

    return Sequential(*modules)


class BaseNet(Module):
    def __init__(self, **kwargs):
        super(BaseNet, self).__init__()

        self._REQUIRED_PARAMS = ['lr', 'max_grad_norm']
        BaseClass.__init__(self, **kwargs)

    def backward_with_loss(self, loss):
        if not hasattr(self, '_optim'):
            self._optim = Adam(self.parameters(), lr=self.lr, eps=1e-4)
            self._scheduler = StepLR(
                self._optim,
                step_size=1000,
                gamma=0.96
            )

        self._optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self._optim.step()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def target_copy(self):
        net = deepcopy(self)
        for param in net.parameters():
            param.requires_grad = False

        return net

    def load(self, fn: str, device: Optional[str]) -> None:
        device = device or self.device

        self.load_state_dict(
            torch.load(fn, map_location=device)
        )


class BasePolicyNet(BaseNet):
    def __init__(self, **kwargs):
        super(BasePolicyNet, self).__init__(**kwargs)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic=False,
        mask=None
    ) -> torch.Tensor:

        if type(state) in [tuple, list]:
            probs = self(*state)
        else:
            probs = self(state)

        if mask is not None:
            probs = probs * mask

        z = probs == 0.0
        z = z.float() * 1e-20

        log_probs = torch.log(probs + z)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = Categorical(probs).sample()

        return action, probs, log_probs
