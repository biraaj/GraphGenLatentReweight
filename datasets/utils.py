from typing import Sequence, Any
import time
import tqdm
import torch
import torch.nn.functional as F

import networkx as nx
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.utils import (to_dense_adj, to_dense_batch, cumsum, degree, coalesce, remove_self_loops,
                                   dense_to_sparse, to_networkx, to_undirected)
from torch_geometric.transforms import BaseTransform
import pickle
from collections import Counter
from loguru import logger

LAPLACIAN_TIME_ATTR = "_laplacian_preprocess_time_s"
ORBIT_TIME_ATTR = "_orbit_preprocess_time_s"


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def init_preprocess_time_totals() -> dict[str, float]:
    return {"laplacian_time_s": 0.0, "orbit_time_s": 0.0}


def pop_preprocess_times(data: Data) -> dict[str, float]:
    laplacian_time_s = float(getattr(data, LAPLACIAN_TIME_ATTR, 0.0))
    orbit_time_s = float(getattr(data, ORBIT_TIME_ATTR, 0.0))
    if hasattr(data, LAPLACIAN_TIME_ATTR):
        delattr(data, LAPLACIAN_TIME_ATTR)
    if hasattr(data, ORBIT_TIME_ATTR):
        delattr(data, ORBIT_TIME_ATTR)
    return {
        "laplacian_time_s": laplacian_time_s,
        "orbit_time_s": orbit_time_s,
    }


def consume_preprocess_times(data: Data, totals: dict[str, float]) -> None:
    times = pop_preprocess_times(data)
    totals["laplacian_time_s"] += times["laplacian_time_s"]
    totals["orbit_time_s"] += times["orbit_time_s"]


def log_preprocess_times(dataset_name: str, totals: dict[str, float]) -> None:
    logger.info(
        f"{dataset_name} preprocessing times | laplacian: {totals['laplacian_time_s']:.2f}s | "
        f"orbits: {totals['orbit_time_s']:.2f}s"
    )


def node_counts(data_list):
    print("Computing node counts...")
    all_node_counts = Counter()
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def to_list(value):
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def transform_dataset(dataset, transform):
    data_list = []
    for data in tqdm.tqdm(dataset, miniters=len(dataset) / 50):
        data_list.append(transform(data))
    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset._data, dataset.slices = dataset.collate(data_list)
    return dataset


def compute_laplacian_eigen(
    edge_index,
    num_nodes,
    max_freq,
    normalized=False,
    normalize=False,
    large_graph=False,
):
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float64)  # (e,)
    row, col = edge_index[0], edge_index[1]
    degree = torch.zeros(num_nodes, dtype=torch.float64)  # (n,)
    degree.scatter_add_(0, row, edge_weight)

    if large_graph:
        if normalized:
            inv_sqrt_degree = degree.clamp_min(1).pow(-0.5)  # (n,)
            laplacian_values = -edge_weight * inv_sqrt_degree[row] * inv_sqrt_degree[col]  # (e,)
            diagonal_values = torch.ones(num_nodes, dtype=torch.float64)  # (n,)
        else:
            laplacian_values = -edge_weight  # (e,)
            diagonal_values = degree  # (n,)

        diagonal_index = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)  # (2, n)
        laplacian_index = torch.cat((edge_index, diagonal_index), dim=1)  # (2, e + n)
        laplacian_values = torch.cat((laplacian_values, diagonal_values), dim=0)  # (e + n,)
        L = torch.sparse_coo_tensor(
            laplacian_index,
            laplacian_values,
            (num_nodes, num_nodes),
        ).coalesce()

        k = min(max_freq, num_nodes)  # ()
        if num_nodes >= 3 * k:
            eigvals, eigvecs = torch.lobpcg(
                L,
                k=k,
                largest=False,
                method="ortho",
            )
        else:
            L_dense = L.to_dense()  # (n, n)
            eigvals, eigvecs = torch.linalg.eigh(L_dense)
    else:
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)  # (n, n)
        A[edge_index[0], edge_index[1]] = 1

        if normalized:
            D12 = torch.diag(A.sum(1).clip(1) ** -0.5)  # (n, n)
            I = torch.eye(A.size(0), dtype=torch.float64)  # (n, n)
            L = I - D12 @ A @ D12  # (n, n)
        else:
            D = torch.diag(A.sum(1))  # (n, n)
            L = D - A  # (n, n)
        eigvals, eigvecs = torch.linalg.eigh(L)

    idx = torch.argsort(eigvals)[:max_freq]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = torch.real(eigvals).clamp_min(0)

    if normalize:
        eignorm: torch.Tensor = eigvecs.norm(p=2, dim=0, keepdim=True)
        eigvecs = eigvecs / eignorm.clamp_min(1e-12).expand_as(eigvecs)

    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))
    eigvals = eigvals.unsqueeze(0).repeat(num_nodes, 1)  # (n, k)
    return eigvals.float(), eigvecs.float()


def compute_magnetic_laplacian_eigen(
    edge_index,
    num_nodes,
    max_freq,
    q=0.25,                 # charge parameter (q=0 -> undirected case)
    normalized=False,       # normalized magnetic Laplacian (MagNet)
    normalize=False,        # L2-normalize eigenvectors (column-wise)
    large_graph=False,
):
    """
    Returns:
        eigvals:  (num_nodes x k) real tensor (k = selected eigen-count)
        eigvecs:  (num_nodes x 2k) real tensor with interleaved [Re, Im, ...]
    """

    # Build (directed) adjacency A \in {0,1}^{n x n}
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    A[edge_index[0], edge_index[1]] = 1.0

    # m
    m = (A != A.t()).sum(dim=(-2, -1)) / 2

    min_m_n = torch.where(m > num_nodes, num_nodes, m)

    # safe division
    q = q / torch.where(min_m_n > 1, min_m_n, torch.tensor(1, dtype=min_m_n.dtype))

    # Symmetrized adjacency and antisymmetric phase (MagNet)
    As = ((A + A.t()) > 0).float()                                   # real symmetric
    Theta = 2.0 * torch.pi * q * (A - A.t())                 # real, antisymmetric
    phase = torch.exp(1j * Theta)                            # complex phase
    # Hermitian adjacency: H = As ⊙ exp(i Theta)
    H = As.to(torch.cdouble) * phase                         # Hadamard product

    n = num_nodes

    if normalized:
        # L_N^{(q)} = I - (D_s^{-1/2} As D_s^{-1/2}) ⊙ exp(i Theta)
        ds = As.sum(1)                                       # real degrees of As
        inv_sqrt = (ds.clamp_min(1).pow(-0.5)).to(torch.cdouble)
        Dm = torch.diag(inv_sqrt)
        As_norm = (Dm @ As.to(torch.cdouble) @ Dm)           # real, symmetric
        H_norm = As_norm * phase                             # Hermitian
        L = torch.eye(n, dtype=torch.cdouble) - H_norm
    else:
        # L_U^{(q)} = D_s - H, with D_s = diag(As 1)
        Ds = torch.diag(As.sum(1).to(torch.cdouble))
        L = Ds - H                                           # Hermitian PSD

    # Hermitian eigendecomposition: eigenvalues are real, vectors complex
    eigvals, eigvecs = torch.linalg.eigh(L)                  # shapes: (n,), (n,n)

    idx = torch.argsort(eigvals.real)[: max_freq]

    eigvals = eigvals[idx].real.clamp_min(0)                 # (k,)
    eigvecs = eigvecs[:, idx]                                # (n, k) complex

    # Pad when num_nodes < max_freq
    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))

    # Repeat eigvals per node as in your original function
    eigvals = eigvals.unsqueeze(0).repeat(num_nodes, 1)      # (n, k_padded)

    return eigvals.float(), eigvecs, q


class Transform:
    def __init__(
        self,
        directed,
        normalized_laplacian,
        normalize_eigenvecs,
        num_vecs,
        large_graph=False,

    ):
        self.directed = directed
        self.normalized = normalized_laplacian
        self.normalize = normalize_eigenvecs
        self.num_vecs = num_vecs
        self.large_graph = large_graph

    def __call__(self, data):
        start_time = time.perf_counter()
        # Keep a fixed spectral width across graphs so variable-size datasets collate cleanly.
        max_freq = self.num_vecs
        if self.directed:
            eigvals, eigvecs, q = compute_magnetic_laplacian_eigen(
                data.edge_index,
                data.num_nodes,
                max_freq,
                0.1,
                self.normalized,
                self.normalize,
                self.large_graph,
            )

            data.eigvals = eigvals
            data.eigvecs_real = eigvecs.real.float()
            data.eigvecs_imag = eigvecs.imag.float()
            data.q = q
        else:
            data.eigvals, data.eigvecs = compute_laplacian_eigen(
                data.edge_index,
                data.num_nodes,
                max_freq,
                self.normalized,
                self.normalize,
                self.large_graph,
            )
        setattr(data, LAPLACIAN_TIME_ATTR, time.perf_counter() - start_time)
        return data


def compute_rrwp(edge_index, num_nodes, ksteps):
    """Even-step RRWP node features (T^2, T^4, ..., T^{2*ksteps} diagonals), adapted
    from GenGNN; even-only avoids the bipartite-graph zero-odd-step problem."""
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    if edge_index.shape[1] > 0:
        A[edge_index[0], edge_index[1]] = 1.0

    deg = A.sum(dim=1)                                               # (n,)
    inv_deg = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    T = inv_deg.unsqueeze(1) * A                                     # row-normalised (n, n)
    T2 = T @ T                                                        # T^2

    diag_idx = torch.arange(num_nodes)
    rrwp_list = []
    Tk = T2                                                           # starts at T^2
    for _ in range(ksteps):
        rrwp_list.append(Tk[diag_idx, diag_idx])                     # T^2, T^4, … T^{2*ksteps}
        Tk = Tk @ T2

    return torch.stack(rrwp_list, dim=1)                             # (n, ksteps)


class RRWPTransform:
    """Replaces Laplacian eigenvectors with even-step RRWP node features, stored in
    data.eigvecs so existing encoders work unchanged; data.eigvals is unused zeros."""
    def __init__(self, ksteps: int):
        self.ksteps = ksteps

    def __call__(self, data):
        start_time = time.perf_counter()
        rrwp = compute_rrwp(data.edge_index, data.num_nodes, self.ksteps)
        data.eigvecs = rrwp                                           # (n, ksteps)
        data.eigvals = torch.zeros(data.num_nodes, self.ksteps)       # dummy
        setattr(data, LAPLACIAN_TIME_ATTR, time.perf_counter() - start_time)
        return data


def compute_rrwp_full(edge_index, num_nodes, ksteps):
    """Consecutive powers T^0..T^{ksteps-1} of T=D^-1 A, matching GenGNN's RRWPFeatures
    exactly (node diagonals + sparse edge features, unlike compute_rrwp's even-only)."""
    n = num_nodes
    A = torch.zeros(n, n, dtype=torch.float32)
    if edge_index.shape[1] > 0:
        A[edge_index[0], edge_index[1]] = 1.0

    deg = A.sum(dim=1)
    inv_deg = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    T = inv_deg.unsqueeze(1) * A                                     # row-normalised (n, n)

    powers = [torch.eye(n, dtype=torch.float32)]                    # T^0
    for _ in range(ksteps - 1):
        powers.append(powers[-1] @ T)                               # T^1, T^2, ..., T^{ksteps-1}
    stacked = torch.stack(powers, dim=-1)                            # (n, n, ksteps)

    diag_idx = torch.arange(n)
    node_rrwp = stacked[diag_idx, diag_idx, :]                       # (n, ksteps)

    if edge_index.shape[1] > 0:
        edge_rrwp = stacked[edge_index[0], edge_index[1], :]         # (e, ksteps)
    else:
        edge_rrwp = torch.zeros(0, ksteps, dtype=torch.float32)

    return node_rrwp, edge_rrwp


class RRWPFullTransform:
    """GenGNN-parity RRWP pre-transform with both node (data.eigvecs) and sparse
    edge (data.rrwp_edge_attr) features."""
    def __init__(self, ksteps: int):
        self.ksteps = ksteps

    def __call__(self, data):
        start_time = time.perf_counter()
        node_rrwp, edge_rrwp = compute_rrwp_full(data.edge_index, data.num_nodes, self.ksteps)
        data.eigvecs = node_rrwp                                      # (n, ksteps)
        data.eigvals = torch.zeros(data.num_nodes, self.ksteps)       # dummy
        data.rrwp_edge_attr = edge_rrwp                                # (e, ksteps)
        setattr(data, LAPLACIAN_TIME_ATTR, time.perf_counter() - start_time)
        return data


class DegreeTransform(BaseTransform):
    def __call__(self, data):
        idx = data.edge_index[0]
        degrees = degree(idx).unsqueeze(-1)
        data.x = degrees
        return data


def compute_1wl_orbits(g):
    if not nx.is_connected(g):
        logger.info(f'Graph is disconnected !')
        # Get the largest connected component as a subgraph
        largest_cc = max(nx.connected_components(g), key=len)
        subgraph = g.subgraph(largest_cc)

        # Compute its diameter
        diameter = nx.diameter(subgraph)
    else:
        diameter = nx.diameter(g)

    # tiny/degenerate graphs (e.g. a 1-atom molecule) can have diameter 0; WL needs
    # iterations >= 1 and such graphs have no non-trivial orbits anyway.
    hashes = nx.weisfeiler_lehman_subgraph_hashes(g, iterations=max(diameter, 1))
    final_hashes = {key: value[-1] for key, value in hashes.items()}

    list_hashes = list(final_hashes.values())
    orbits = {}
    for i, hash_value in enumerate(list_hashes):
        if hash_value not in orbits:
            orbits[hash_value] = []
        orbits[hash_value].append(i)

    non_trivial_orbits = [value for key, value in orbits.items() if len(value) > 1]
    return non_trivial_orbits


class OrbitTransform(BaseTransform):
    def __call__(self, data):
        start_time = time.perf_counter()
        g = to_networkx(
            data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
            to_undirected=True
        )
        non_trivial_orbits = compute_1wl_orbits(g)
        orbits_attr = torch.zeros_like(data.x[..., [0]])

        for i, idxs in enumerate(non_trivial_orbits):
            orbits_attr[idxs] = i + 1
        data.orbits = orbits_attr
        setattr(data, ORBIT_TIME_ATTR, time.perf_counter() - start_time)
        return data


def _tree_centroids(g):
    """Standard leaf-peeling centroid algorithm: iteratively strip degree-<=1
    nodes until 1 or 2 nodes remain. Every tree has exactly 1 or 2 centroids,
    a purely structural property (independent of node labels)."""
    if g.number_of_nodes() <= 2:
        return list(g.nodes())
    degrees = dict(g.degree())
    leaves = [n for n, d in degrees.items() if d <= 1]
    remaining = g.number_of_nodes()
    adj = {n: set(g.neighbors(n)) for n in g.nodes()}
    while remaining > 2:
        next_leaves = []
        for leaf in leaves:
            remaining -= 1
            for nbr in adj[leaf]:
                adj[nbr].discard(leaf)
                degrees[nbr] -= 1
                if degrees[nbr] == 1:
                    next_leaves.append(nbr)
        leaves = next_leaves
    return leaves


def _ahu_canonical_order(g, root):
    """AHU canonical form of `g` rooted at `root`: (whole-tree label, canonical node
    order). Children ordered by subtree label so isomorphic subtrees always match."""
    parent = {root: None}
    children = {root: []}
    bfs_order = [root]
    frontier = [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in g.neighbors(u):
                if v not in parent:
                    parent[v] = u
                    children[u].append(v)
                    children[v] = []
                    bfs_order.append(v)
                    nxt.append(v)
        frontier = nxt

    label = {}
    for u in reversed(bfs_order):
        child_labels = sorted(label[c] for c in children[u])
        label[u] = "(" + "".join(child_labels) + ")"

    canonical_order = []
    stack = [root]
    while stack:
        u = stack.pop()
        canonical_order.append(u)
        # sorted() ascending, then push in reverse so pop() visits ascending order first
        ordered_children = sorted(children[u], key=lambda c: label[c])
        stack.extend(reversed(ordered_children))

    return label[root], canonical_order


def canonical_tree_perm(g):
    """Canonical node ordering for tree `g`, exactly invariant to input labeling (not
    just invariant-in-distribution): roots at the centroid(s), orders children by AHU
    subtree label. Returns {original_node_label: canonical_index}. See Aho, Hopcroft
    & Ullman (1974)."""
    candidates = _tree_centroids(g)
    best_label, best_order = None, None
    for root in candidates:
        label, order = _ahu_canonical_order(g, root)
        if best_label is None or label < best_label:
            best_label, best_order = label, order
    return {node: idx for idx, node in enumerate(best_order)}


class CanonicalTreeTransform(BaseTransform):
    """Relabels a tree into AHU canonical order (see canonical_tree_perm) before any
    other pre-transform runs, so encode() becomes genuinely permutation-invariant."""
    def __call__(self, data):
        g = to_networkx(data, to_undirected=True)
        perm = canonical_tree_perm(g)  # {old_label: new_index}
        n = data.num_nodes
        new_index = torch.empty(n, dtype=torch.long)
        for old, new in perm.items():
            new_index[old] = new

        inv = torch.empty(n, dtype=torch.long)
        inv[new_index] = torch.arange(n)  # inv[new_idx] = old_idx

        data.x = data.x[inv]
        data.edge_index = new_index[data.edge_index]
        return data


def build_pe_transform(cfg):
    """The positional-encoding pre-transform implied by `cfg.encoder.pe_type`.

    Factored out of utils.instantiate_dataset so that other code paths can rebuild
    the exact same node features from cfg in one place.
    """
    pe_type = cfg.encoder.get("pe_type", "lpe")
    if pe_type == "rrwp":
        return RRWPTransform(ksteps=cfg.encoder.num_vecs)
    if pe_type == "rrwp_full":
        return RRWPFullTransform(ksteps=cfg.encoder.num_vecs)
    return Transform(
        directed=cfg.dataset.directed,
        normalized_laplacian=cfg.encoder.normalized_laplacian,
        normalize_eigenvecs=cfg.encoder.normalize_eigenvecs,
        num_vecs=cfg.encoder.num_vecs,
        large_graph=cfg.dataset.large_graph,
    )
