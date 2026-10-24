import hashlib
import os
import pickle
import time
from typing import Any

import networkx as nx
import numpy as np
import torch
import wandb
from ema_pytorch import EMA
from loguru import logger
from omegaconf import OmegaConf
from torch_geometric.utils import to_dense_batch



def checkpoint_cache_key(checkpoint_path: str | None) -> str | None:
    """Content hash of an AE checkpoint, so copies across runs share the same dim-weight cache."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None
    h = hashlib.md5()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

from evaluation.moses.metrics import get_all_metrics
from evaluation.moses.molecules import graph_to_smiles
from evaluation.synthetic import (
    PlanarSamplingMetrics,
    SBMSamplingMetrics,
    TreeSamplingMetrics,
)
from fm import VParamInterpolant, X0ParamInterpolant, X1ParamInterpolant
from models import (
    AutoencoderKL,
    DecoderTransformerUnDirected,
    DiT,
    EncoderGINVAE,
)


def get_selection_metric(cfg: Any) -> str:
    if "selection_metric" in cfg:
        return cfg.selection_metric
    if cfg.dataset.molecular:
        return "valid"
    if cfg.dataset.directed:
        return "sampling/frac_unic_non_iso_valid"
    return "acc" if cfg.dataset.name in {"planar", "planar_original", "tree", "tree_original", "tree_128"} else "ratio"


def get_selection_mode(cfg: Any) -> str:
    if "selection_mode" in cfg:
        return cfg.selection_mode
    return "max" if get_selection_metric(cfg) in {"valid", "acc", "sampling/frac_unic_non_iso_valid"} else "min"


def should_save_every_eval(cfg: Any) -> bool:
    return bool(cfg.get("save_every_eval", cfg.dataset.directed))


def build_sampling_metrics(
    cfg: Any,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    test: bool,
    cache_path: str | None = None,
) -> Any:
    """cache_path: skip the reference-set precompute and reuse a cached result if present."""
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        logger.info(f"Loaded cached sampling metrics from {cache_path} (skipping reference-set precompute)")
        return cached

    metrics = _build_sampling_metrics(cfg, train_loader, val_loader, test_loader, test)

    if cache_path is not None and metrics is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(metrics, f)
        logger.info(f"Cached sampling metrics to {cache_path} for future runs to reuse")

    return metrics


def _build_sampling_metrics(
    cfg: Any,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    test: bool,
) -> Any:
    if cfg.dataset.molecular:
        return None

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if cfg.dataset.name in {"planar", "planar_original"}:
        return PlanarSamplingMetrics(dataloaders, test=test)
    if cfg.dataset.name == "sbm_modified":
        return SBMSamplingMetrics(dataloaders, test=test)
    if cfg.dataset.name in {"tree_original", "tree_128"}:
        return TreeSamplingMetrics(dataloaders, test=test)

    raise NotImplementedError(f"Unknown FM dataset: {cfg.dataset.name}")


def _build_decoder(cfg: Any, num_edge_types: int) -> Any:
    """Instantiate the undirected decoder. Set Transformer + bilinear edge head
    is the only decoder this release supports (decoder_type=transformer, matching
    the VGAE architecture everywhere it's used)."""
    decoder_type = cfg.get("decoder_type", "transformer")
    if decoder_type != "transformer":
        raise NotImplementedError(f"decoder_type={decoder_type!r} is not supported; only 'transformer' is.")
    return DecoderTransformerUnDirected(
        pe_dim=cfg.encoder.pe_dim,
        num_node_features=cfg.dataset.num_node_types,
        num_edge_features=num_edge_types,
        num_heads=cfg.get("decoder_num_heads", 4),
        num_layers=cfg.get("decoder_num_layers", 4),
        hidden_dim=cfg.get("decoder_hidden_dim", 256),
        dropout=cfg.dropout,
    )


def build_autoencoder(cfg: Any, device_id: Any) -> AutoencoderKL:
    # Encoder reads num_edge_features, decoder reads num_edge_types -- differ only for molecular datasets.
    num_edge_types = cfg.dataset.num_edge_types if not cfg.dataset.molecular else cfg.dataset.num_edge_features
    encoder_type = cfg.get("encoder_type", "vgae")
    if encoder_type != "vgae":
        raise NotImplementedError(f"encoder_type={encoder_type!r} is not supported; only 'vgae' is.")
    decoder = _build_decoder(cfg, cfg.dataset.num_edge_types)
    encoder = EncoderGINVAE(
        num_node_types=cfg.dataset.num_node_types,
        num_node_features=cfg.dataset.num_node_features,
        num_edge_features=num_edge_types,
        global_cfg=cfg.encoder,
        rho_cfg=cfg.encoder.rho,
        dropout=cfg.dropout,
    )
    return AutoencoderKL(encoder, decoder, directed=False).to(device_id)


def _ckpt_load_args(device_id: Any) -> dict:
    """torch.load map_location kwargs; device_id must be spelled "cuda:<idx>", not a bare int."""
    return dict(map_location=f"cuda:{device_id}") if torch.cuda.is_available() else {}


def resolve_scale_factor(
    cfg: Any,
    checkpoint_path: str,
    autoencoder: AutoencoderKL,
    batch: Any,
    device_id: Any,
    extra_batches: list | None = None,
) -> torch.Tensor:
    """Prefers the checkpoint's stored scale_factor over recomputing (noisier);
    extra_batches pools more batches when none is stored."""
    ckpt = torch.load(checkpoint_path, **_ckpt_load_args(device_id), weights_only=False)
    stored = ckpt.get("scale_factor", None)
    if stored is not None:
        logger.info(f"Using scale_factor from checkpoint: {float(stored):.6f}")
        return torch.as_tensor(stored, device=device_id)

    batches = [batch] if not extra_batches else [batch, *extra_batches]
    z_chunks = []
    with torch.no_grad():
        for i, b in enumerate(batches):
            posterior = autoencoder.encode(b)
            z_b = posterior.sample() if cfg.get("sample", True) else posterior.mode()
            z_chunks.append(z_b.detach().flatten().cpu())
            # Free CUDA cache between iterations to avoid fragmentation OOMs across pooled batches.
            if i < len(batches) - 1:
                torch.cuda.empty_cache()
    z = torch.cat(z_chunks).to(device_id)
    scale_factor = 1.0 / z.std()
    logger.warning(
        f"Checkpoint has no stored scale_factor; recomputed from {len(batches)} "
        f"batch(es): {float(scale_factor):.6f}"
    )
    return scale_factor


def predict_x1(param: str, x_t: torch.Tensor, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Derive the predicted clean latent x_1 from the denoiser output, given cfg.param.

    t: (bs, 1); x_t, out: (bs, n, pe_dim)
    """
    t3 = t.unsqueeze(-1)  # (bs, 1, 1)
    if param == "x_0":
        return (x_t - (1 - t3) * out) / t3.clamp(min=1e-4)
    if param == "x_1":
        return out
    return x_t + (1 - t3) * out


def _batch_sensitivity_sum(
    autoencoder: AutoencoderKL,
    z_probe: torch.Tensor,
    batch_index_probe: torch.Tensor,
    q_probe: Any,
    eps: float,
    exact: bool,
) -> tuple[torch.Tensor, int]:
    """One batch's SUM (not mean) of per-graph sensitivity norms, plus graph count, so
    callers can accumulate a correctly weighted mean across batches of different sizes."""
    pe_dim = z_probe.shape[-1]
    sensitivity_sum = torch.zeros(pe_dim, device=z_probe.device)
    num_graphs = int(batch_index_probe.max().item()) + 1

    def _per_graph_sum_norm(diff_e: torch.Tensor, diff_x: torch.Tensor) -> torch.Tensor:
        e_sq_per_graph = diff_e.float().pow(2).sum(dim=tuple(range(1, diff_e.dim())))  # (num_graphs,)
        x_sq_per_node = diff_x.float().pow(2).sum(dim=-1)  # (n_total,)
        x_sq_per_graph = torch.zeros(num_graphs, device=diff_x.device).scatter_add_(
            0, batch_index_probe, x_sq_per_node
        )
        per_graph_norm = torch.sqrt(e_sq_per_graph + x_sq_per_graph)  # (num_graphs,)
        return per_graph_norm.sum()

    if exact:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def _decode_fn(z: torch.Tensor):
            e_hat, x_hat, _, _ = autoencoder.decode(z, batch_index_probe, q_probe)
            return e_hat, x_hat

        prev_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with sdpa_kernel(SDPBackend.MATH):
                for d in range(pe_dim):
                    tangent = torch.zeros_like(z_probe)
                    tangent[:, d] = 1.0
                    with torch.no_grad():
                        _, (jvp_e, jvp_x) = torch.func.jvp(_decode_fn, (z_probe,), (tangent,))
                    sensitivity_sum[d] = _per_graph_sum_norm(jvp_e, jvp_x)
        finally:
            torch.backends.mha.set_fastpath_enabled(prev_fastpath)
    else:
        with torch.no_grad():
            for d in range(pe_dim):
                delta = torch.zeros_like(z_probe)
                delta[:, d] = eps
                e_plus, x_plus, _, _ = autoencoder.decode(z_probe + delta, batch_index_probe, q_probe)
                e_minus, x_minus, _, _ = autoencoder.decode(z_probe - delta, batch_index_probe, q_probe)
                diff_e = (e_plus.float() - e_minus.float()) / (2 * eps)
                diff_x = (x_plus.float() - x_minus.float()) / (2 * eps)
                sensitivity_sum[d] = _per_graph_sum_norm(diff_e, diff_x)

    return sensitivity_sum, num_graphs


def measure_jacobian_dim_weight_dataset(
    autoencoder: AutoencoderKL,
    data_loader: Any,
    device_id: Any,
    q_fn: Any = None,
    min_var: float = 0.01,
    eps: float = 1e-2,
    exact: bool = True,
    max_batches: int | None = None,
    cache_path: str | None = None,
    cache_key: str | None = None,
    mc_samples: int = 1,
) -> torch.Tensor:
    """w_d = Var[mu_d] * E_z[||d(decode(z))/dz_d||], accumulated over the full dataset
    and cached per AE checkpoint."""
    if cache_path is not None and os.path.exists(cache_path):
        cached = torch.load(
            cache_path,
            map_location=(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"),
            weights_only=False,
        )
        if cached.get("cache_key") == cache_key:
            logger.info(f"Loaded cached jacobian dim weight from {cache_path} (skipping ~1hr measurement)")
            return cached["weight"].to(device_id)
        logger.warning(
            f"Cache at {cache_path} doesn't match this checkpoint (cache_key mismatch: "
            f"cached={cached.get('cache_key')!r} vs current={cache_key!r}) -- recomputing and overwriting it"
        )

    if mc_samples < 1:
        raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")

    sum_sensitivity: torch.Tensor | None = None
    total_graphs = 0
    var_chunks: list[torch.Tensor] = []

    for i, batch in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch.to(device_id)
        with torch.no_grad():
            posterior = autoencoder.encode(batch)
            var_sample = posterior.mean.detach()
        var_chunks.append(var_sample.cpu())  # accumulate on CPU -- full dataset is too large for GPU
        q_probe = q_fn(batch) if q_fn is not None else None

        # Sensitivity is always probed at real sampled z; mc_samples>1 averages several draws.
        for _ in range(mc_samples):
            with torch.no_grad():
                z_sample = posterior.sample().detach()
            batch_sum, batch_graphs = _batch_sensitivity_sum(autoencoder, z_sample, batch.batch, q_probe, eps, exact)
            if sum_sensitivity is None:
                sum_sensitivity = batch_sum
            else:
                sum_sensitivity = sum_sensitivity + batch_sum
        total_graphs += batch_graphs * mc_samples

    sensitivity = sum_sensitivity / total_graphs
    dim_var = torch.cat(var_chunks, dim=0).to(device_id).var(dim=0)

    floored_var = dim_var.clamp(min=min_var)
    raw_weight = floored_var * sensitivity
    weight = raw_weight / raw_weight.mean()

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {"weight": weight.cpu(), "cache_key": cache_key, "pe_dim": weight.numel()},
            cache_path,
        )
        logger.info(f"Cached jacobian dim weight to {cache_path} for future runs to reuse")

    return weight


def measure_dim_standardize_stats_dataset(
    data_loader: Any,
    autoencoder: AutoencoderKL,
    device_id: Any,
    max_batches: int | None = None,
    cache_path: str | None = None,
    cache_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opt-in per-dimension (mean_d, std_d) of the encoder's posterior mean, in place of
    main_fm.py's single global scale_factor. Returns (dim_mean, dim_std)."""
    if cache_path is not None and os.path.exists(cache_path):
        cached = torch.load(
            cache_path,
            map_location=(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"),
            weights_only=False,
        )
        if cached.get("cache_key") == cache_key:
            logger.info(f"Loaded cached dim-standardize stats from {cache_path} (skipping full-dataset pass)")
            return cached["dim_mean"].to(device_id), cached["dim_std"].to(device_id)
        logger.warning(
            f"Cache at {cache_path} doesn't match this checkpoint (cache_key mismatch: "
            f"cached={cached.get('cache_key')!r} vs current={cache_key!r}) -- recomputing and overwriting it"
        )

    mu_chunks: list[torch.Tensor] = []
    for i, batch in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch.to(device_id)
        with torch.no_grad():
            posterior = autoencoder.encode(batch)
            mu_chunks.append(posterior.mean.detach().cpu())

    mu_all = torch.cat(mu_chunks, dim=0).to(device_id)
    dim_mean = mu_all.mean(dim=0)
    dim_std = mu_all.std(dim=0).clamp(min=1e-4)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {"dim_mean": dim_mean.cpu(), "dim_std": dim_std.cpu(), "cache_key": cache_key},
            cache_path,
        )
        logger.info(f"Cached dim-standardize stats to {cache_path} for future runs to reuse")

    return dim_mean, dim_std


def build_interpolant(cfg: Any, device_id: Any) -> Any:
    interpolant_kwargs = dict(
        num_timesteps=cfg.num_sampling_steps,
        time_density=cfg.time_density,
        time_density_params=cfg.density_params,
        sampling_time_density=cfg.sampling_time_density,
        sampling_time_density_params=cfg.sampling_density_params,
        conditioning=cfg.sc_prob > 0,
        device=device_id,
    )
    if cfg.param == "x_0":
        return X0ParamInterpolant(**interpolant_kwargs)
    if cfg.param == "x_1":
        return X1ParamInterpolant(**interpolant_kwargs)
    return VParamInterpolant(**interpolant_kwargs)


def build_denoiser(cfg: Any, device_id: Any) -> DiT:
    return DiT(
        num_layers=cfg.denoiser.num_layers,
        num_heads=cfg.denoiser.num_heads,
        in_dim=cfg.encoder.pe_dim,
        embed_dim=cfg.denoiser.embed_dim,
        dropout=cfg.dropout,
        use_pe=cfg.denoiser.use_pe,
        # Opt in with +denoiser.qk_norm=true; off by default for existing configs.
        qk_norm=cfg.denoiser.get("qk_norm", False),
    ).to(device_id)


def build_ema_model(cfg: Any, device_id: Any) -> EMA:
    model = build_denoiser(cfg, device_id)
    return EMA(model, beta=0.9999, update_after_step=1000, update_every=1, allow_different_devices=True)


def sample_n_nodes(node_distribution: Any, cfg: Any, device_id: Any, num_samples: int | None = None) -> torch.Tensor:
    sampled = node_distribution.sample_n(cfg.num_samples if num_samples is None else num_samples, device=device_id)
    if torch.is_tensor(sampled):
        n_nodes = sampled.to(device_id)
    else:
        n_nodes = torch.tensor(sampled, device=device_id)
    return n_nodes.long()


def mst_project_adjacency(edge_logits: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Maximum-spanning-tree projection of one graph's edge logits into a guaranteed
    valid tree (connected, acyclic, n_nodes-1 edges), via Kruskal's algorithm on
    symmetrized, strictly-positive confidence scores."""
    n = n_nodes
    confidence = (edge_logits[:n, :n, 1] - edge_logits[:n, :n, 0]).detach().float().cpu().numpy()
    confidence = (confidence + confidence.T) / 2
    off_diag = confidence[~np.eye(n, dtype=bool)]
    confidence = confidence - off_diag.min() + 1.0
    np.fill_diagonal(confidence, 0.0)
    graph = nx.from_numpy_array(confidence)
    mst = nx.maximum_spanning_tree(graph, weight="weight")
    adj = nx.to_numpy_array(mst, nodelist=range(n))
    return torch.from_numpy((adj > 0).astype(np.int64))


def _eval_chunk_sizes(total: int, device_count: int) -> list[int]:
    """Split an eval batch into device_count chunks (only for device_count > 2) to
    bound peak memory; same total samples either way."""
    num_chunks = device_count if device_count > 2 else 1
    base, rem = divmod(total, num_chunks)
    return [base + (1 if i < rem else 0) for i in range(num_chunks) if base + (1 if i < rem else 0) > 0]


def generate_samples(
    interpolant: Any,
    model: Any,
    autoencoder: AutoencoderKL,
    scale_factor: float,
    node_distribution: Any,
    cfg: Any,
    device_id: Any,
    atom_decoder: Any | None = None,
    q: float | None = None,
    enc_sample_mean: Any | None = None,
    device_count: int = 1,
    latent_transform: Any | None = None,
    dim_mean: Any = 0.0,
    as_mol_tuples: bool = False,
) -> list[Any]:
    """Sample latents with the FM model and decode them into graphs. dim_mean is
    opt-in (paired with a per-dim scale_factor under dim_standardize); 0.0 is a no-op."""
    generated_samples: list[Any] = []
    sampling_start = time.time()
    chunk_sizes = _eval_chunk_sizes(cfg.num_samples, device_count)
    for _ in range(cfg.num_sample_batch):
        for chunk_size in chunk_sizes:
            n_nodes = sample_n_nodes(node_distribution, cfg, device_id, num_samples=chunk_size)  # (bs,)
            batch_size = len(n_nodes)
            n_max = int(torch.max(n_nodes).detach().cpu())
            arange = torch.arange(n_max, device=device_id).unsqueeze(0).expand(batch_size, -1)  # (bs, n)
            sample_mask = arange < n_nodes.unsqueeze(1)  # (bs, n)
            graph_indices = torch.arange(batch_size, device=device_id).repeat_interleave(n_max).reshape(batch_size, n_max)  # (bs, n)
            batch_tensor = graph_indices[sample_mask]  # (n_total,)

            latent_samples = interpolant.sample(
                batch_size=batch_size,
                num_tokens=n_max,
                embed_dim=cfg.encoder.pe_dim,
                model=model,
                token_mask=sample_mask,
                return_traj=False,  # only tokens_traj[-1] is used below — avoid O(num_timesteps) memory
            )
            latent_tokens = latent_samples["tokens_traj"][-1] / scale_factor + dim_mean  # (bs, n, pe_dim)

            # Optional: shift generated latents so their per-dim mean matches the encoder's.
            if enc_sample_mean is not None:
                z_valid = latent_tokens[sample_mask]                         # (n_total, pe_dim)
                gen_mean = z_valid.mean(0)                                   # (pe_dim,)
                shift = enc_sample_mean.to(device_id) - gen_mean            # (pe_dim,)
                latent_tokens = latent_tokens + shift.unsqueeze(0).unsqueeze(0)

            decode_kwargs = {"batch": batch_tensor}
            if cfg.dataset.directed:
                decode_kwargs["q"] = torch.full((batch_size,), fill_value=q, device=device_id)  # (bs,)

            z_sparse = latent_tokens[sample_mask]  # (n_total, pe_dim)
            # Optional inference-time transport mapping generated latents onto encoder statistics.
            if latent_transform is not None:
                z_sparse = latent_transform(z_sparse)

            decoded = autoencoder.decode(z_sparse, **decode_kwargs)

            if cfg.dataset.molecular and as_mol_tuples:
                # (atom_types, edge_types) tuples for evaluation.molecular_digress (DiGress/DeFoG protocol).
                e_hat, x_hat, _, _ = decoded
                e_cls = e_hat.argmax(dim=-1).long()                       # (bs, n, n)
                x_hat, _ = to_dense_batch(x_hat, batch=batch_tensor)
                x_cls = x_hat.argmax(dim=-1).long()                       # (bs, n)
                for i in range(batch_size):
                    ni = int(n_nodes[i])
                    generated_samples.append(
                        (x_cls[i, :ni].cpu(), e_cls[i, :ni, :ni].cpu())
                    )
            elif cfg.dataset.molecular:
                e_hat, x_hat, _, _ = decoded
                x_hat, _ = to_dense_batch(x_hat, batch=batch_tensor)
                generated_samples.extend(
                    graph_to_smiles(
                        e_hat=e_hat,
                        x_hat=x_hat,
                        num_nodes=n_nodes,
                        atom_decoder=atom_decoder,
                    )
                )
            elif cfg.dataset.directed:
                e_hat, x_hat, edge_mask, _ = decoded
                e_hat = e_hat.argmax(dim=-1)
                e_hat = (e_hat * edge_mask).long()
                x_hat, _ = to_dense_batch(x_hat, batch=batch_tensor)
                x_hat = x_hat.argmax(dim=-1).long()
                for i in range(batch_size):
                    generated_samples.append((x_hat[i, : n_nodes[i]], e_hat[i, : n_nodes[i], : n_nodes[i]]))
            else:
                e_hat_raw, _, edge_mask, _ = decoded
                # MST-projection path disabled -- standard independent per-slot argmax instead.
                # if bool(cfg.get("mst_project", False)):
                #     e_hat = torch.zeros(e_hat_raw.shape[:3], dtype=torch.long, device=e_hat_raw.device)
                #     for i in range(batch_size):
                #         ni = int(n_nodes[i])
                #         e_hat[i, :ni, :ni] = mst_project_adjacency(e_hat_raw[i], ni).to(e_hat.device)
                # else:
                e_hat = e_hat_raw.argmax(dim=-1)
                e_hat = (e_hat * edge_mask).long()
                for i in range(batch_size):
                    generated_samples.append(e_hat[i, : n_nodes[i], : n_nodes[i]])

    sampling_end = time.time()
    total_samples = cfg.num_samples * cfg.num_sample_batch
    logger.info(
        f"Sampling {cfg.num_sample_batch} batches took : {sampling_end - sampling_start} - "
        f"Samples per second: {total_samples / (sampling_end - sampling_start)}"
    )
    return generated_samples


def evaluate_samples(
    generated_samples: list[Any],
    cfg: Any,
    device_id: Any,
    step: int,
    sampling_metrics: Any | None = None,
    train_dataset: Any | None = None,
    test_dataset: Any | None = None,
) -> dict[str, float]:
    if sampling_metrics is None and not cfg.dataset.molecular:
        return {}
    if cfg.dataset.molecular:
        try:
            metrics = get_all_metrics(
                gen=generated_samples,
                k=cfg.num_samples * cfg.num_sample_batch,
                device=device_id,
                n_jobs=8,
                test=list(test_dataset.smiles),
                train=list(train_dataset.smiles),
            )
        except ZeroDivisionError:
            logger.info(f"No valid molecule on step {step}")
            metrics = {"valid": -1.0}
    else:
        metrics = sampling_metrics(generated_graphs=generated_samples)

    if cfg.wandb_project is not None:
        wandb.log(metrics, step=step)
    return metrics
