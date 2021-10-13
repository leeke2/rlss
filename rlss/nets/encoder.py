import torch
import numpy as np
from torch.nn import (
    Module, Linear, LayerNorm, ModuleList, BatchNorm1d
)
from torch.nn import functional as F

class MultiHeadAttentionLayer(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_heads: int,
            use_bias: bool
    ) -> None:

        super(MultiHeadAttentionLayer, self).__init__()

        self.out_channels = out_channels
        self.num_heads = num_heads

        self.Q = Linear(in_channels, out_channels * num_heads, bias=use_bias)
        self.K = Linear(in_channels, out_channels * num_heads, bias=use_bias)
        self.V = Linear(in_channels, out_channels * num_heads, bias=use_bias)
        self.E = Linear(in_channels, out_channels * num_heads, bias=use_bias)

    def forward(self, x, e, edge_index):
        Q_h = self.Q(x).view(-1, self.num_heads, self.out_channels)
        K_h = self.K(x).view(-1, self.num_heads, self.out_channels)
        V_h = self.V(x).view(-1, self.num_heads, self.out_channels)
        E_e = self.E(e).view(-1, self.num_heads, self.out_channels)

        src, dst = edge_index[0], edge_index[1]
        e_out = K_h[src] * Q_h[dst] * E_e / np.sqrt(self.out_channels)

        score_sum = e_out.sum(dim=-1, keepdim=True)
        score_sm = scatter_softmax(score_sum, src, dim=0)

        attn = score_sm * V_h[dst]
        attn = attn.view(-1, self.num_heads * self.out_channels)
        h_out = scatter_sum(attn, src, dim=0, dim_size=x.shape[0])

        return h_out, e_out.view(-1, self.num_heads * self.out_channels)


class GraphTransformerLayer(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_heads: int,
            dropout: float = 0.0,
            layer_norm: bool = False,
            batch_norm: bool = True,
            residual: bool = True,
            use_bias: bool = False
    ) -> None:

        super(GraphTransformerLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        # self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(
            in_channels, out_channels // num_heads, num_heads, use_bias)

        self.O_h = Linear(out_channels, out_channels)
        self.O_e = Linear(out_channels, out_channels)

        if self.layer_norm:
            self.layer_norm1_h = LayerNorm(out_channels)
            self.layer_norm1_e = LayerNorm(out_channels)

        if self.batch_norm:
            self.batch_norm1_h = BatchNorm1d(out_channels)
            self.batch_norm1_e = BatchNorm1d(out_channels)

        self.FFN_h_layer1 = Linear(out_channels, out_channels * 2)
        self.FFN_h_layer2 = Linear(out_channels * 2, out_channels)

        self.FFN_e_layer1 = Linear(out_channels, out_channels * 2)
        self.FFN_e_layer2 = Linear(out_channels * 2, out_channels)

        if self.layer_norm:
            self.layer_norm2_h = LayerNorm(out_channels)
            self.layer_norm2_e = LayerNorm(out_channels)

        if self.batch_norm:
            self.batch_norm2_h = BatchNorm1d(out_channels)
            self.batch_norm2_e = BatchNorm1d(out_channels)

    def forward(self, h, e, edge_index):
        h_in1, e_in1 = h, e

        h, e = self.attention(h, e, edge_index)

        # h = F.dropout(h, self.dropout, training=self.training)
        h = self.O_h(h)

        # e = F.dropout(e, self.dropout, training=self.training)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h
            e = e_in1 + e

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2, e_in2 = h, e

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        # h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        # e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h
            e = e_in2 + e

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'heads={self.num_heads}, '
                f'residual={self.residual})')


class MLPReadout(Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            Linear(input_dim // 2**ly, input_dim // 2**(ly + 1), bias=True)
            for ly in range(L)]

        list_FC_layers.append(Linear(input_dim // 2**L, output_dim, bias=True))
        self.FC_layers = ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x

        for ly in range(self.L):
            y = self.FC_layers[ly](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)

        return y


class GraphTransformerEncoder(Module):
    def __init__(
        self,
        node_faetures: int,
        edge_features: int,
        embedding_size: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        use_positional_encoding: bool = True,
        positional_encoding_dims: int = 20
    ) -> None:

        super(GraphTransformerEncoder, self).__init__()

        self.use_positional_encoding = use_positional_encoding

        self.embedding_h = Linear(node_faetures, embedding_size)
        self.embedding_e = Linear(edge_features, embedding_size)

        if use_positional_encoding:
            self.embedding_pe = Linear(positional_encoding_dims,
                                       embedding_size)

        self.layers = ModuleList([
            GraphTransformerLayer(embedding_size,
                                  embedding_size,
                                  n_heads)
            for _ in range(n_encoder_layers)
        ])

        self.mlp = MLPReadout(embedding_size, embedding_size)

        self._batch_size = None

    def forward(self, nodes, edges, edge_index, pos_enc):
        from torch_scatter import scatter_softmax, scatter_sum, scatter

        batch_size, n_nodes, _ = nodes.shape
        nodes = nodes.flatten(start_dim=0, end_dim=1)
        edges = edges.flatten(start_dim=0, end_dim=1)
        pos_enc = pos_enc.flatten(start_dim=0, end_dim=1)

        self._batch = torch.arange(batch_size).repeat_interleave(n_nodes).to(nodes.device)

        h = self.embedding_h(nodes)
        e = self.embedding_e(edges)

        if self.use_positional_encoding:
            h = h + self.embedding_pe(pos_enc)

        for idx, net in enumerate(self.layers):
            h, e = net(h, e, edge_index[0].repeat(1, batch_size))

        size = int(self._batch.max().item() + 1)
        hg = scatter(h, self._batch, dim=0, dim_size=size, reduce='mean')
        return self.mlp(hg)
        # n_nodes, node_features = nodes.shape

        # h = self.embedding_h(nodes)
        # e = self.embedding_e(edges)

        # if self.use_positional_encoding:
        #     pe = self.embedding_pe(pos_enc)
        #     h = h + pe

        # for net in self.layers:
        #     h, e = net(h, e, edge_index)

        # size = int(self._batch.max().item() + 1)
        # hg = scatter(h, self._batch, dim=0, dim_size=size, reduce='mean')

        # return self.mlp(hg)
