"""
PlanarDatasetOriginal — the exact same 64-node planar graphs used by DiGress / GenGNN
(train=128, val=32, test=40).  Downloads from the Bergmeister et al. repository
if not already present locally, then processes with LG-Flow's standard pipeline.
"""

import os
import os.path as osp
import pickle

import torch
import torch_geometric
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

DIGRESS_PLANAR_URL = (
    "https://raw.githubusercontent.com/AndreasBergmeister/"
    "graph-generation/main/data/planar.pkl"
)

GENGNN_PLANAR_PKL = (
    "/home/biraaj/Projects/graph_diffusion/"
    "gengnn_analysis_with_encoders/GenGNN_KDD/data/planar/raw/planar.pkl"
)


def _resolve_pkl(raw_dir: str, pkl_path: str | None) -> str:
    if pkl_path and osp.exists(pkl_path):
        return pkl_path
    if osp.exists(GENGNN_PLANAR_PKL):
        return GENGNN_PLANAR_PKL
    import urllib.request
    dest = osp.join(raw_dir, "planar_digress.pkl")
    if not osp.exists(dest):
        os.makedirs(raw_dir, exist_ok=True)
        print(f"Downloading DiGress planar dataset from:\n  {DIGRESS_PLANAR_URL}")
        urllib.request.urlretrieve(DIGRESS_PLANAR_URL, dest)
        print(f"  Saved to {dest}")
    return dest


def _nx_to_pyg(G: nx.Graph) -> torch_geometric.data.Data:
    n         = G.number_of_nodes()
    adj       = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
    x         = torch.zeros(n, 1, dtype=torch.long)
    edge_attr = torch.zeros(edge_index.shape[-1], dtype=torch.long).unsqueeze(-1)
    n_nodes   = n * torch.ones(1, dtype=torch.long)
    return torch_geometric.data.Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=n_nodes
    )


class PlanarDatasetOriginal(InMemoryDataset):
    """
    DiGress / GenGNN original planar dataset — 200 graphs (128 train / 32 val / 40 test),
    all 64-node planar graphs (Delaunay triangulation), processed with LG-Flow's
    standard LPE pipeline.

    Source: https://github.com/AndreasBergmeister/graph-generation
    """

    def __init__(
            self,
            root,
            split,
            pkl_path=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self._pkl_path = pkl_path
        self.n_nodes   = 64
        super().__init__(root, transform, pre_transform, pre_filter)

        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)
        self.num_nodes = load_pickle(
            osp.join(self.processed_dir, 'node_counts.pickle')
        )[split]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        pkl_path = _resolve_pkl(self.raw_dir, self._pkl_path)
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)
        print(f"  Loaded DiGress planar dataset: "
              f"train={len(raw['train'])}  val={len(raw['val'])}  test={len(raw['test'])}")

        preprocess_times = init_preprocess_time_totals()
        counts = {}

        for split in ['train', 'val', 'test']:
            data_list = []
            for G in raw[split]:
                data = _nx_to_pyg(G)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    consume_preprocess_times(data, preprocess_times)
                data_list.append(data)

            counts[split] = node_counts(data_list)
            torch.save(
                self.collate(data_list),
                osp.join(self.processed_dir, f'{split}.pt'),
            )

        save_pickle(counts, osp.join(self.processed_dir, 'node_counts.pickle'))
        log_preprocess_times(self.__class__.__name__, preprocess_times)
