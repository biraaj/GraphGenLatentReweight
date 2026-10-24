"""
K-fold test evaluation for flow matching models.

Runs generation k independent times with different seeds, reports mean ± std
of all metrics — matches standard SOTA evaluation protocol for graph generation
(typically k=5, 1024 samples per fold).

Usage:
    python eval_fm_kfold.py --config-name=tree_original_fm_vgae \
        root=/path/to/outputs num_folds=5 num_samples=256 num_sample_batch=4
"""

import os
import numpy as np
import torch
from typing import Any

import hydra
import wandb
from loguru import logger
from omegaconf import OmegaConf, open_dict

from fm import compute_n_nodes_distr
from fm.fm_helpers import (
    build_autoencoder,
    build_ema_model,
    build_interpolant,
    build_sampling_metrics,
    evaluate_samples,
    generate_samples,
    resolve_scale_factor,
)
from evaluation.molecular_digress import compute_molecular_metrics_full, precompute_fcd_reference
from utils import create_dataloaders, instantiate_dataset, load_checkpoint, setup_everything


def collect_mol_tuples(loader: Any, device_id: Any) -> list[Any]:
    """(atom_types [n] long, edge_types [n,n] long) tuples for every graph in loader,
    in the DiGress/DeFoG convention (0 = no bond, 1.. = bond order). Used as the
    reference set for NSPDK MMD -- see compute_molecular_metrics_full."""
    from torch_geometric.utils import to_dense_adj, to_dense_batch

    tuples = []
    for batch in loader:
        batch = batch.to(device_id)
        n_per_graph = torch.bincount(batch.batch.cpu()).tolist()
        x_idx = batch.x.argmax(dim=-1) if (batch.x.dim() > 1 and batch.x.shape[-1] > 1) else batch.x.reshape(-1).long()
        x_dense, _ = to_dense_batch(x_idx, batch=batch.batch)
        e_dense = to_dense_adj(
            edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr[..., 0] + 1,
        ).long()
        for i, nn in enumerate(n_per_graph):
            tuples.append((x_dense[i, :nn].cpu(), e_dense[i, :nn, :nn].cpu()))
    return tuples


def infer_ae_arch(ckpt_path: str) -> dict:
    """Infer AE architecture hyper-parameters (ds_dim, pe_dim, num_vecs, rni_dim) from
    checkpoint weight shapes; mirrors analysis/fm_analysis.py's infer_ae_arch."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw)
    arch = {}

    if "decoder.in_proj.weight" in state:
        arch["ds_dim"] = state["decoder.in_proj.weight"].shape[1]
    elif "decoder.edge_out_proj.wctx.weight" in state:
        arch["ds_dim"] = state["decoder.edge_out_proj.wctx.weight"].shape[0]

    if "encoder.rho.mlp.2.weight" in state:
        arch["pe_dim"] = state["encoder.rho.mlp.2.weight"].shape[0] // 2
        if "ds_dim" not in arch:
            arch["ds_dim"] = arch["pe_dim"]

    if "encoder.lpe_proj.weight" in state:
        arch["num_vecs"] = state["encoder.lpe_proj.weight"].shape[1]
    elif "encoder.eps" in state:
        arch["num_vecs"] = state["encoder.eps"].shape[1]

    if "encoder.rni_proj.weight" in state:
        arch["rni_dim"] = state["encoder.rni_proj.weight"].shape[1]
    else:
        arch["rni_dim"] = 0

    return arch


@hydra.main(version_base=None, config_path="./configs", config_name="planar_fm_test")
def main(cfg: Any) -> None:
    device, device_id, device_count, master_process, data_dir, ckpt_dir, dtype, tdtype = setup_everything(cfg)

    logger.info(f"Loading datasets from {data_dir}")
    train_dataset, val_dataset, test_dataset = instantiate_dataset(
        name=cfg.dataset.name,
        data_dir=data_dir,
        cfg=cfg,
        master_process=master_process,
    )
    node_distribution = compute_n_nodes_distr(
        train_n_nodes=train_dataset.num_nodes,
        val_n_nodes=val_dataset.num_nodes,
        test_n_nodes=test_dataset.num_nodes,
    )
    logger.info("Dataset loaded")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
        device_count=device_count,
        master_process=master_process,
    )

    # test=True: reference graphs are the held-out test set; cached per dataset, not per checkpoint.
    sampling_metrics = (
        build_sampling_metrics(
            cfg, train_loader, val_loader, test_loader, test=True,
            cache_path=os.path.join(data_dir, "sampling_metrics_test_cache.pkl"),
        )
        if master_process else None
    )

    # ── infer architecture from AE checkpoint (before building the model) ──────
    ae_ckpt_path = f"{ckpt_dir}/{cfg.ae_checkpoint_file}.pt"
    if master_process:
        logger.info(f"[arch] reading architecture from checkpoint: {ae_ckpt_path}")
    ae_arch = infer_ae_arch(ae_ckpt_path)
    with open_dict(cfg.encoder):
        for key, val in ae_arch.items():
            old = OmegaConf.select(cfg.encoder, key, default=None)
            if old != val:
                if master_process:
                    logger.info(f"[arch]   encoder.{key}: {old} → {val}  (from checkpoint)")
                setattr(cfg.encoder, key, val)
            elif master_process:
                logger.info(f"[arch]   encoder.{key} = {val}  (matches config)")

    # ── load models ──────────────────────────────────────────────────────────
    autoencoder = build_autoencoder(cfg, device_id)
    autoencoder.eval()
    load_checkpoint(f"{ckpt_dir}/{cfg.ae_checkpoint_file}.pt", autoencoder, device_id)
    for param in autoencoder.parameters():
        param.requires_grad = False

    interpolant = build_interpolant(cfg, device_id)
    ema_model = build_ema_model(cfg, device_id)
    load_checkpoint(f"{ckpt_dir}/{cfg.checkpoint}.pt", ema_model, device_id, ema=True)
    ema_model.eval()

    # Compute scale factor from the first training batch; pool extra batches for a
    # less noisy recompute when the checkpoint has no stored value.
    train_iter = iter(train_loader)
    batch = next(train_iter).to(device_id)
    extra_batches = [next(train_iter).to(device_id) for _ in range(3)]
    fm_ckpt_path = f"{ckpt_dir}/{cfg.checkpoint}.pt"
    scale_factor = resolve_scale_factor(cfg, fm_ckpt_path, autoencoder, batch, device_id, extra_batches=extra_batches)
    q = batch.q.squeeze()[0].item() if cfg.dataset.directed else None

    # Precompute per-dim mean of encoder samples over full training set for
    # optional post-hoc mean correction (+mean_correct=true on command line).
    enc_sample_mean = None
    if cfg.get("mean_correct", False):
        logger.info("Computing encoder sample mean over training set for mean correction ...")
        all_z = []
        with torch.no_grad():
            for b in train_loader:
                z_b = autoencoder.encode(b.to(device_id)).sample().detach()
                all_z.append(z_b)
        enc_sample_mean = torch.cat(all_z, dim=0).mean(0)  # (pe_dim,)
        logger.info(f"  enc_sample_mean norm = {enc_sample_mean.norm().item():.4f}")

    # Molecular datasets: DiGress/DeFoG-style metrics instead of the generic
    # sampling_metrics path. No natural "fold" notion -- num_folds=1 with a large
    # sample count is the usual case, though num_folds>1 still works.
    mol_ctx = None
    if cfg.dataset.molecular:
        mol_atom_decoder = getattr(train_dataset, "atom_decoder", None)
        mol_train_smiles = set(getattr(train_dataset, "smiles", []) or [])
        mol_ref_smiles = list(getattr(test_dataset, "smiles", []) or [])
        nspdk = bool(cfg.get("nspdk", True))
        logger.info(f"[mol] train SMILES: {len(mol_train_smiles)}  test/ref SMILES: {len(mol_ref_smiles)}")
        fcd_cache = os.path.join(data_dir, f"fcd_ref_{cfg.dataset.name}.pkl")
        mol_fcd_ref = precompute_fcd_reference(mol_ref_smiles, fcd_cache, device=device_id)
        mol_ref_tuples = collect_mol_tuples(test_loader, device_id) if nspdk else []
        mol_ctx = {
            "atom_decoder": mol_atom_decoder,
            "train_smiles": mol_train_smiles,
            "ref_smiles": mol_ref_smiles,
            "ref_tuples": mol_ref_tuples,
            "fcd_ref_stats": mol_fcd_ref,
            "nspdk": nspdk,
            "dataset_name": cfg.dataset.name,
        }

    # ── k-fold generation ────────────────────────────────────────────────────
    num_folds = cfg.get("num_folds", 5)
    base_seed = cfg.get("seed", 0)
    samples_per_fold = cfg.num_samples * cfg.num_sample_batch

    logger.info(
        f"K-fold eval: {num_folds} folds × {samples_per_fold} samples "
        f"= {num_folds * samples_per_fold} total generated graphs"
    )

    fold_metrics: list[dict] = []
    for fold in range(num_folds):
        seed = base_seed + fold
        torch.manual_seed(seed)
        np.random.seed(seed)

        logger.info(f"Fold {fold + 1}/{num_folds}  (seed={seed})")
        generated = generate_samples(
            interpolant=interpolant,
            model=ema_model,
            autoencoder=autoencoder,
            scale_factor=scale_factor,
            node_distribution=node_distribution,
            cfg=cfg,
            device_id=device_id,
            atom_decoder=getattr(train_dataset, "atom_decoder", None),
            q=q,
            enc_sample_mean=enc_sample_mean,
            device_count=device_count,
            as_mol_tuples=cfg.dataset.molecular,
        )
        if mol_ctx is not None:
            metrics = compute_molecular_metrics_full(
                generated, mol_ctx["atom_decoder"],
                train_smiles=mol_ctx["train_smiles"], ref_smiles=mol_ctx["ref_smiles"],
                ref_tuples=mol_ctx["ref_tuples"], fcd_ref_stats=mol_ctx["fcd_ref_stats"],
                device=device_id, nspdk=mol_ctx["nspdk"], dataset_name=mol_ctx["dataset_name"],
            )
            if cfg.wandb_project is not None:
                wandb.log(metrics, step=fold)
        else:
            metrics = evaluate_samples(
                generated_samples=generated,
                cfg=cfg,
                device_id=device_id,
                step=fold,
                sampling_metrics=sampling_metrics,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )
        fold_metrics.append(metrics)
        logger.info(f"  fold {fold + 1} → {metrics}")

    # ── aggregate ─────────────────────────────────────────────────────────────
    numeric_keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    mean_m = {k: float(np.mean([m[k] for m in fold_metrics])) for k in numeric_keys}
    std_m  = {k: float(np.std( [m[k] for m in fold_metrics])) for k in numeric_keys}

    # ── wandb summary ─────────────────────────────────────────────────────────
    if cfg.wandb_project and master_process:
        summary = {}
        for k in numeric_keys:
            summary[f"kfold/{k}_mean"] = mean_m[k]
            summary[f"kfold/{k}_std"]  = std_m[k]
        wandb.log(summary, step=num_folds)
        # Also push to the run summary panel so mean/std appear at the top of the run
        wandb.run.summary.update(summary)

    # ── print table ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"K-fold test results — {num_folds} folds × {samples_per_fold} samples/fold"
        f"  (total {num_folds * samples_per_fold})"
    )
    logger.info(f"{'Metric':<30}  {'Mean':>10}  {'Std':>8}")
    logger.info("-" * 54)
    for k in numeric_keys:
        logger.info(f"{k:<30}  {mean_m[k]:>10.4f}  {std_m[k]:>8.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
