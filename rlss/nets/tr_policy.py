from torch import cat, argmax
from torch.nn import Softmax
from torch.nn.functional import pad

from .base import ff_net, BasePolicyNet
from .encoder import GraphTransformerEncoder


class TrPolicyNet(BasePolicyNet):
    def __init__(self, node_features_dim, edge_features_dim, out_dim, pos_enc_dim, **kwargs):

        super(TrPolicyNet, self).__init__(**kwargs)

        self.out_dim = out_dim

        self.encoder = GraphTransformerEncoder(
            node_features_dim,
            edge_features_dim,
            embedding_size=kwargs['embedding_size'],
            n_heads=kwargs['n_heads'],
            n_encoder_layers=kwargs['n_encoder_layers'],
            use_positional_encoding=True,
            positional_encoding_dims=pos_enc_dim)

        self.net = ff_net(kwargs['embedding_size'], out_dim)
        self.sm = Softmax(dim=-1)

    def forward(self, nodes, edges, edge_index, pos_enc):
        x = self.encoder(nodes, edges, edge_index, pos_enc)
        x = self.net(x)
        x = self.sm(x)

        return x
