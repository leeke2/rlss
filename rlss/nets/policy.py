from torch.nn import Softmax
from .base import ff_net, BasePolicyNet


class PolicyNet(BasePolicyNet):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(PolicyNet, self).__init__(**kwargs)

        self.out_dim = out_dim

        self.net = ff_net(in_dim, out_dim)
        self.sm = Softmax(dim=-1)

    def forward(self, x):
        x = self.net(x)
        x = self.sm(x)

        return x
