from torch import Tensor
from torch.nn import Parameter

from .base import BaseNet


class Temperature(BaseNet):
    def __init__(self, **kwargs):
        super(Temperature, self).__init__(**kwargs)

        self.log_alpha = Parameter(Tensor([0]))

    def forward(self, x):
        return None
