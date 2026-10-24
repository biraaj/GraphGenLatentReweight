import os.path as osp

import numpy as np
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
import networkx as nx

from .utils import (
    consume_preprocess_times,
    init_preprocess_time_totals,
    load_pickle,
    log_preprocess_times,
    node_counts,
    save_pickle,
)


class SBMModifiedDataset(InMemoryDataset):
    """"Extended" SBM benchmark: same 8192/256/256 train/val/test scale as
    PlanarDataset (n_graphs=8704), for scale-matched comparison across the
    "extended" synthetic benchmarks. Unlike planar/tree, node count here is
    NOT fixed -- it varies per graph.

    Same MECHANISM as the canonical SBM generator (github.com/KarolisMart/SPECTRE,
    data.py's SBMDataset): per graph, num_blocks ~ randint(2, 5) inclusive, and each
    block's size drawn INDEPENDENTLY ~ randint(a, b) (not a partition of a
    fixed total) -- so node count genuinely varies per graph, same as the
    original DiGress/SPECTRE fixed benchmark (confirmed variable 44-187 there).
    The reference itself uses size ~ randint(20, 40) (mean 30), giving
    E[total] = E[num_blocks] * E[size] = 3.5 * 30 = 105. To land "more
    towards 128" while keeping the reference's exact shape (same 1:2
    min:max ratio, same width-to-mean proportions, same num_blocks range),
    the size range is scaled up: a=24, b=48 (mean 36, same 1:2 ratio as
    20:40) -> E[total] = 3.5 * 36 = 126. p_intra=0.3, p_inter=0.005 --
    identical to the reference (also confirmed against DeFoG's own
    `is_sbm_graph` validity check, src/analysis/spectre_utils.py:732,
    which uses these exact values as its ground-truth SBM parameters).

    This is a MODIFIED node-count/scale variant of the canonical SBM
    benchmark (see docstring above), not the original fixed benchmark --
    hence the "modified" name.
    """

    def __init__(
            self,
            root,
            split,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.n_graphs = 8704
        self.num_blocks_range = (2, 5)  # inclusive
        self.p_intra = 0.3
        self.p_inter = 0.005
        self.block_size_range = (24, 48)  # inclusive
        super().__init__(root, transform, pre_transform, pre_filter)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def _sample_sbm_graph(self):
        num_blocks = np.random.randint(self.num_blocks_range[0], self.num_blocks_range[1] + 1)
        # Independent per-block draw, no fixed-total constraint -- same mechanism as
        # github.com/KarolisMart/SPECTRE's data.py, just a rescaled range (see docstring).
        sizes = np.random.randint(self.block_size_range[0], self.block_size_range[1] + 1, size=num_blocks).tolist()
        p = np.full((num_blocks, num_blocks), self.p_inter)
        np.fill_diagonal(p, self.p_intra)
        return nx.stochastic_block_model(sizes, p.tolist())

    def process(self):
        preprocess_times = init_preprocess_time_totals()
        data_list = []
        for i in range(self.n_graphs):
            G = self._sample_sbm_graph()
            adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
            n = adj.shape[-1]
            x = torch.zeros(n, 1, dtype=torch.long)
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], dtype=torch.long).unsqueeze(-1)
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=n_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                consume_preprocess_times(data, preprocess_times)

            data_list.append(data)

        train_data_list = data_list[:8192]
        val_data_list = data_list[8192:8448]
        test_data_list = data_list[8448:]
        counts = {'train': 0, 'val': 0, 'test': 0}
        for split, data_list in zip(['train', 'val', 'test'], [train_data_list, val_data_list, test_data_list]):
            node_count = node_counts(data_list)
            counts[split] = node_count

            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))
        save_pickle(counts, osp.join(self.processed_dir, 'node_counts.pickle'))
        log_preprocess_times(self.__class__.__name__, preprocess_times)
