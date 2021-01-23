import numpy as np
import torch
from torch_geometric.nn import GCNConv

from ComGCN.utility.dyn_graph_reader import DynGraphReader


class ComEmb(torch.nn.Module):

    def __init__(self, in_channels: int, improved: bool = False, cached: bool = False,
                 normalize: bool = True, add_self_loops: bool = True):
        super(ComEmb, self).__init__()

        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self._create_layers()

    def _create_layers(self):
        self.conv_layer = GCNConv(in_channels=self.in_channels,
                                  out_channels=self.in_channels,
                                  improved=self.improved,
                                  cached=self.cached,
                                  normalize=self.normalize,
                                  add_self_loops=self.add_self_loops,
                                  bias=False)

    def forward(self,
                DynGraphReader: DynGraphReader,
                snapshot: int,
                X: torch.FloatTensor,
                edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:

        g = DynGraphReader.network_snapshots[snapshot]
        g_node_embs = DynGraphReader.network_snapshot_embs[snapshot]
        snapshot_community_graph_dict = DynGraphReader.network_snapshot_communities[snapshot]
        stacked_community_gcn_W = np.ones(shape=(len(g.nodes), self.in_channels))

        for community_idx in snapshot_community_graph_dict.keys():
            g_community = snapshot_community_graph_dict[community_idx]
            X_com, edge_index, edge_weight, mapped_node_idx_dict, mapped_idx_node_dict = \
                self.parse_community_graph_data(g_community, g_node_embs)
            community_gcn_W = self.conv_layer(X_com, edge_index, edge_weight)
            for idx, node_emb in enumerate(community_gcn_W):
                stacked_community_gcn_W[mapped_idx_node_dict[idx]] = node_emb.tolist()

        stacked_community_gcn_W = torch.FloatTensor(stacked_community_gcn_W)
        print(stacked_community_gcn_W.shape)
        W = torch.FloatTensor(stacked_community_gcn_W)
        return torch.mul(W, X)

    def parse_community_graph_data(self, g_community, g_node_embs):
        nodes = [node_id for node_id in g_community.nodes]
        nodes.sort()
        mapped_node_idx_dict = {}
        mapped_idx_node_dict = {}
        for idx, node_id in enumerate(nodes):
            mapped_node_idx_dict[node_id] = idx
            mapped_idx_node_dict[idx] = node_id

        mapped_edges = [(mapped_node_idx_dict[src_node_id], mapped_node_idx_dict[des_node_id]) for
                        (src_node_id, des_node_id) in
                        g_community.edges()]
        if len(mapped_edges) == 0:
            mapped_edges = [(0, 0)]

        edge_index = torch.LongTensor(np.array(mapped_edges).T)
        edge_weight = torch.FloatTensor(np.ones(edge_index.shape[1]))

        g_community_node_embs = [g_node_embs[node_id] for node_id in nodes]
        X = torch.FloatTensor(g_community_node_embs)
        return X, edge_index, edge_weight, mapped_node_idx_dict, mapped_idx_node_dict
