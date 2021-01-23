import torch
import torch.nn as nn
import numpy as np
from torch.nn import LSTM
from torch_geometric.nn import GCNConv

from ComGCN.utility.dyn_graph_reader import DynGraphReader
from ComGCN.mul_sco_emb.comb_emb import ComEmb


class ComGCN(torch.nn.Module):

    def __init__(self, in_channels: int, improved: bool = False, cached: bool = False,
                 normalize: bool = True, add_self_loops: bool = True):
        super(ComGCN, self).__init__()

        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self._create_layers()

    def _create_layers(self):
        self.com_emb = ComEmb(in_channels=self.in_channels)

        self.recurrent_layer = LSTM(input_size=self.in_channels,
                                    hidden_size=self.in_channels,
                                    num_layers=1,
                                    bidirectional=True)

        self.conv_layer = GCNConv(in_channels=self.in_channels,
                                  out_channels=self.in_channels,
                                  improved=self.improved,
                                  cached=self.cached,
                                  normalize=self.normalize,
                                  add_self_loops=self.add_self_loops,
                                  bias=False)

        self.fusion_linear_layer = nn.Linear(self.in_channels, self.in_channels)
        self.fusion_linear_layer_no_bias = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.fusion_sigmoid_layer = nn.Sigmoid()

    def forward(self,
                graphReader: DynGraphReader,
                snapshot: int,
                X: torch.FloatTensor,
                edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:
        W = self.conv_layer.weight[None, :, :]

        W, _ = self.recurrent_layer(W)

        W_forward = W[0, :, :self.in_channels]
        W_backword = W[0, :, self.in_channels:]

        W_max_pool = torch.FloatTensor(np.max(np.array([W_forward.numpy(), W_backword.numpy()]), axis=0))

        self.conv_layer.weight = torch.nn.Parameter(W_max_pool.squeeze())

        X_node_emb = torch.FloatTensor(graphReader.network_snapshot_embs[snapshot])
        X_com_emb = self.com_emb(graphReader, snapshot, X, edge_index, edge_weight)
        X_ma_emb = self.conv_layer(X, edge_index, edge_weight)

        X = self.linear_fusion_function(X_node_emb, X_com_emb, X_ma_emb)
        # X = self.nonlinear_fusion_function(X_node_emb, X_com_emb, X_ma_emb)

        return X

    def linear_fusion_function(self, X_node_emb, X_com_emb, X_ma_emb):
        return self.fusion_linear_layer(X_node_emb + X_com_emb + X_ma_emb)

    def nonlinear_fusion_function(self, X_node_emb, X_com_emb, X_ma_emb):
        _1 = self.fusion_linear_layer(X_node_emb + X_com_emb + X_ma_emb)
        _2 = self.fusion_sigmoid_layer(_1)
        _3 = self.fusion_linear_layer_no_bias(_2)
        return self.fusion_sigmoid_layer(_3)
