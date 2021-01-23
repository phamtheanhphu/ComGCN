import networkx as nx
import pandas as pd
import random
import gensim
from datetime import datetime
from community import community_louvain


class DynGraphReader():

    def __init__(self, data_file, node_embs_root_dir, limit_training_pair_each_class=500):

        self.data_file = data_file
        self.node_embs_root_dir = node_embs_root_dir
        self.limit_training_pair_each_class = limit_training_pair_each_class
        self.group_by_month_dict = {}

        self.network_snapshots = {}
        self.network_snapshot_embs = {}
        self.network_snapshot_communities = {}

        self.nodes = None
        self.training_labels = {}
        self.training_pairs = {}

    def read(self):

        dataset_csv = pd.read_csv(self.data_file, encoding='utf-8')
        nodes = []
        for row in dataset_csv.values:
            if row[0] not in nodes:
                nodes.append(int(row[0]))
            if row[1] not in nodes:
                nodes.append(int(row[1]))
            record_dt_obj = datetime.fromtimestamp(row[2])
            if record_dt_obj.month not in self.group_by_month_dict.keys():
                self.group_by_month_dict[record_dt_obj.month] = []
            self.group_by_month_dict[record_dt_obj.month].append((row[0] - 1, row[1] - 1))

        self.nodes = [(i - 1) for i in nodes]
        self.nodes.sort()

        for month in self.group_by_month_dict.keys():
            g = nx.Graph()
            g.add_nodes_from(self.nodes)
            g.add_edges_from(self.group_by_month_dict[month])
            self.network_snapshots[month] = g

        for month in self.network_snapshots.keys():
            g_snapshot = self.network_snapshots[month]
            self.training_labels[month] = []
            self.training_pairs[month] = []
            pos_label_count = 0
            neg_label_count = 0
            for (src_node_id, des_node_id) in g_snapshot.edges:
                if pos_label_count >= self.limit_training_pair_each_class:
                    break
                self.training_labels[month].append(1)
                self.training_pairs[month].append((int(src_node_id), int(des_node_id)))
                pos_label_count += 1

            nodes = [node_id for node_id in g_snapshot.nodes]
            random.shuffle(nodes)
            for src_node_id in nodes:
                if neg_label_count >= self.limit_training_pair_each_class:
                    break
                random.shuffle(nodes)
                for des_node_id in nodes:
                    if neg_label_count >= self.limit_training_pair_each_class:
                        break
                    if not g_snapshot.has_edge(src_node_id, des_node_id):
                        self.training_labels[month].append(0)
                        self.training_pairs[month].append((int(src_node_id), int(des_node_id)))
                        neg_label_count += 1

    def read_network_snapshot_embs(self):
        for snapshot in self.network_snapshots.keys():
            snapshot_node_emb_file_path = './{}/{}.emb'.format(self.node_embs_root_dir, snapshot)
            node_embs = gensim.models.KeyedVectors.load_word2vec_format(snapshot_node_emb_file_path)
            nodes = [int(float(i)) for i in node_embs.vocab.keys()]
            node_emb_list = []
            for node in nodes:
                node_emb_list.append(node_embs.get_vector(str(node)))
            self.network_snapshot_embs[snapshot] = node_emb_list

    def parse_community_structure(self):
        for snapshot in self.network_snapshots.keys():
            g = self.network_snapshots[snapshot]
            community_node_dict = {}
            community_graph_dict = {}
            node_community_dict = community_louvain.best_partition(g)
            for node_id in node_community_dict.keys():
                community_idx = node_community_dict[node_id]
                if community_idx not in community_node_dict.keys():
                    community_node_dict.update({community_idx: [node_id]})
                else:
                    community_node_dict[community_idx].append(node_id)
            for community_idx in community_node_dict:
                nodes_in_community = community_node_dict[community_idx]
                g_community = nx.Graph()
                g_community.add_nodes_from(nodes_in_community)
                for src_node_id in nodes_in_community:
                    for des_node_id in nodes_in_community:
                        if g.has_edge(src_node_id, des_node_id):
                            g_community.add_edge(src_node_id, des_node_id)
                community_graph_dict[community_idx] = g_community
            self.network_snapshot_communities[snapshot] = community_graph_dict
