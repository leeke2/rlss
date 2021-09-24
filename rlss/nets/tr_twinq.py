from copy import deepcopy
from torch import argmax
from torch.nn import Sequential, ModuleList

from .base import ff_net, BaseNet, BaseClass
from .encoder import GraphTransformerEncoder


class TrTwinQNet(BaseNet):
    def __init__(self, node_features_dim, edge_features_dim, out_dim, pos_enc_dim, **kwargs):
        super(TrTwinQNet, self).__init__(**kwargs)

        self._REQUIRED_PARAMS = ['distinct_qnet_encoders']
        BaseClass.__init__(self, **kwargs)

        if self.distinct_qnet_encoders:
            self.encoders = ModuleList()

            for i in range(2):
                self.encoders.append(GraphTransformerEncoder(
                    node_features_dim,
                    edge_features_dim,
                    embedding_size=kwargs['embedding_size'],
                    n_heads=kwargs['n_heads'],
                    n_encoder_layers=kwargs['n_encoder_layers'],
                    use_positional_encoding=True,
                    positional_encoding_dims=pos_enc_dim))
        else:
            encoder = GraphTransformerEncoder(
                node_features_dim,
                edge_features_dim,
                embedding_size=kwargs['embedding_size'],
                n_heads=kwargs['n_heads'],
                n_encoder_layers=kwargs['n_encoder_layers'],
                use_positional_encoding=True,
                positional_encoding_dims=pos_enc_dim)

            self.encoders = ModuleList([encoder, encoder])

        self.q1_ff = ff_net(kwargs['embedding_size'], out_dim)
        self.q2_ff = ff_net(kwargs['embedding_size'], out_dim)

    def forward(self, nodes, edges, edge_index, pos_enc):
        q1 = self.encoders[0](nodes, edges, edge_index, pos_enc)
        q1 = self.q1_ff(q1)

        q2 = self.encoders[1](nodes, edges, edge_index, pos_enc)
        q2 = self.q1_ff(q2)

        # x_sum = x.sum(axis=-1)
        # emp1 = x_sum.sum(axis=-1) == 0
        # emp2 = x_sum.sum(axis=-2) == 0
        # emp = (emp1 & emp2) * 1

        # emp_rolled = emp.roll(1, 1)
        # emp_rolled[:, 0] = emp[:, 0]
        # oh_n_nodes = (emp != emp_rolled) * 1

        # valid = oh_n_nodes.sum(1) == 1
        # n_nodes = argmax(oh_n_nodes)

        # for i in range(len(valid)):
        #     if valid[i]:
        #         print('twinq x')
        #         q1[i, n_nodes[i] + 1:-2] = 0
        #         q2[i, n_nodes[i] + 1:-2] = 0

        return q1, q2


