import os
import random
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from torch.distributed import destroy_process_group
from torch_geometric.utils import to_dense_batch

from fm import compute_n_nodes_distr
from fm.fm_helpers import (
    build_autoencoder,
    build_denoiser,
    build_interpolant,
    build_sampling_metrics,
    evaluate_samples,
    generate_samples,
    get_selection_metric,
    get_selection_mode,
    checkpoint_cache_key,
    measure_dim_standardize_stats_dataset,
    measure_jacobian_dim_weight_dataset,
    predict_x1,
    should_save_every_eval,
)
from utils import (
    create_dataloaders,
    instantiate_dataset,
    load_checkpoint,
    save_checkpoint,
    setup_everything,
    setup_training,
)


def _scale_ckpt_fields(scale_factor: Any, dim_mean: Any) -> dict:
    """Checkpoint fields for scale_factor: scalar -> {"scale_factor"}, per-dim tensor -> {"dim_std", "dim_mean"}."""
    sf = torch.as_tensor(scale_factor)
    if sf.numel() == 1:
        return {"scale_factor": float(sf)}
    return {"dim_std": (1.0 / sf).detach().cpu(), "dim_mean": torch.as_tensor(dim_mean).detach().cpu()}


@hydra.main(version_base=None, config_path="./configs", config_name="planar_fm_train")
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

    sampling_metrics = None
    if master_process:
        sampling_metrics = build_sampling_metrics(
            cfg, train_loader, val_loader, test_loader, test=False,
            # test=False: reference set is val_graphs; keep "val" in the cache filename.
            cache_path=os.path.join(data_dir, "sampling_metrics_val_cache.pkl"),
        )
    if device_count > 1:
        # Master-only precompute above can outlast NCCL's timeout; barrier keeps other ranks in sync.
        torch.distributed.barrier()

    # ── Autoencoder setup ─────────────────────────────────────────────────────
    autoencoder = build_autoencoder(cfg, device_id)
    # strict_ae_load=false loads only matching keys, e.g. to swap in a different decoder_type.
    strict_ae_load = bool(cfg.get("strict_ae_load", True))
    ae_checkpoint_path = f"{ckpt_dir}/{cfg.ae_checkpoint_file}.pt"
    load_checkpoint(ae_checkpoint_path, autoencoder, device_id, strict=strict_ae_load)
    # Content hash, not path, so the dim-weight cache is reused across runs/checkpoints.
    ae_checkpoint_cache_key = checkpoint_cache_key(ae_checkpoint_path)

    # The autoencoder is always frozen: FM trains only the denoiser against its fixed latent.
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    # Weight on the primary FM criterion (default 1.0).
    fm_mse_weight = float(cfg.get("fm_mse_weight", 1.0))

    # Per-dimension FM loss reweighting by measured decoder sensitivity (jacobian, var_source=mu);
    # off by default. Measured once over the full training set, cached per AE checkpoint.
    dim_reweight = bool(cfg.get("dim_reweight", False))
    dim_reweight_min_var = float(cfg.get("dim_reweight_min_var", 0.01))
    if dim_reweight:
        logger.info(
            f"Per-dimension loss reweighting enabled (jacobian, var_source=mu), "
            f"dim_reweight_min_var={dim_reweight_min_var}"
        )

    # dim_standardize: opt-in per-dimension (mean, std) standardization instead of one
    # global scale_factor. Off by default.
    dim_standardize = bool(cfg.get("dim_standardize", False))
    if dim_standardize:
        logger.info(
            "Per-dimension latent standardization enabled — replacing the single global "
            "scale_factor with per-dimension (mean_d, std_d), measured over the full "
            "training set and cached per AE checkpoint, see below"
        )

    interpolant = build_interpolant(cfg, device_id)
    model = build_denoiser(cfg, device_id)

    model, ema_model, optimizer, lr_scheduler, step, checkpoint_file, best_val = setup_training(
        cfg=cfg,
        model=model,
        master_process=master_process,
        device=device,
        device_id=device_id,
        device_count=device_count,
        dtype=dtype,
        ckpt_dir=ckpt_dir,
    )

    # On resume, restore fine-tuned AE weights and the persisted scale_factor/dim_weight.
    resume_scale_factor: float | None = None
    resume_dim_weight: torch.Tensor | None = None
    resume_dim_mean: torch.Tensor | None = None
    resume_dim_std: torch.Tensor | None = None
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        load_args = dict(map_location=f"cuda:{device_id}") if torch.cuda.is_available() else {}
        # weights_only=False: checkpoints contain numpy scalars PyTorch 2.6's default can't load.
        resume_ckpt = torch.load(checkpoint_file, weights_only=False, **load_args)
        if "scale_factor" in resume_ckpt:
            resume_scale_factor = float(resume_ckpt["scale_factor"])
            logger.info(f"Resumed scale_factor from checkpoint: {resume_scale_factor:.6f}")
        else:
            logger.warning(
                f"{checkpoint_file} has no stored scale_factor (predates persistence) -- "
                "will recompute fresh from this run's first batch, which may not exactly "
                "match what the resumed weights were trained under."
            )
        if "dim_weight" in resume_ckpt:
            resume_dim_weight = resume_ckpt["dim_weight"]
            logger.info("Resumed per-dimension loss weight from checkpoint")
        if "dim_std" in resume_ckpt:
            # dim_mean defaults to 0.0 for older checkpoints missing the key.
            resume_dim_mean = resume_ckpt.get("dim_mean", 0.0)
            resume_dim_std = resume_ckpt["dim_std"]
            logger.info("Resumed per-dimension latent standardization stats from checkpoint")

    best_value = best_val[0]
    selection_metric = get_selection_metric(cfg)
    selection_mode = get_selection_mode(cfg)
    save_every_eval = should_save_every_eval(cfg)
    # Optional ceiling on the "ratio" metric a new best checkpoint must also satisfy.
    selection_ratio_max = cfg.get("selection_ratio_max", None)
    if selection_ratio_max is not None:
        logger.info(f"Checkpoint selection gated on ratio <= {selection_ratio_max} (in addition to {selection_metric})")

    logger.info(f"Starting/resuming training for {int(cfg.num_steps) - (step - 1)} steps 🚀")

    if master_process:
        start_time = time.time()
    if device_count > 1:
        train_loader.sampler.set_epoch(epoch := 0)

    first_batch = True
    q: float | None = None
    scale_factor = 1.0
    dim_mean = 0.0
    while step <= cfg.num_steps:
        model.train()
        loss_window: list[float] = []
        norm_loss_window: list[float] = []
        for batch in train_loader:
            batch = batch.to(device_id)
            if first_batch:
                posterior = autoencoder.encode(batch)
                z_init = posterior.sample().detach() if cfg.get("sample", True) else posterior.mode().detach()
                # Prefer the resumed checkpoint's scale_factor over recomputing from this batch.
                scale_factor = (
                    torch.as_tensor(resume_scale_factor, device=device_id)
                    if resume_scale_factor is not None
                    else 1.0 / z_init.flatten().std()
                )
                if cfg.dataset.directed:
                    q = batch.q.squeeze()[0].item()
                if dim_reweight:
                    # Prefer the resumed checkpoint's dim_weight when present.
                    if resume_dim_weight is not None:
                        dim_weight = resume_dim_weight.to(device_id)
                        logger.info("Using resumed per-dimension loss weight")
                    else:
                        # Measured once over the full training set; costs pe_dim decoder passes per batch.
                        _jac_hash_prefix = (ae_checkpoint_cache_key or "nokey")[:8]
                        dim_weight = measure_jacobian_dim_weight_dataset(
                            autoencoder=autoencoder,
                            data_loader=train_loader,
                            device_id=device_id,
                            q_fn=(lambda b: b.q.squeeze()[0].item()) if cfg.dataset.directed else None,
                            min_var=dim_reweight_min_var,
                            eps=float(cfg.get("dim_reweight_jacobian_eps", 1e-2)),
                            exact=bool(cfg.get("dim_reweight_jacobian_exact", True)),
                            max_batches=cfg.get("dim_reweight_jacobian_max_batches", None),
                            cache_path=os.path.join(data_dir, f"jacobian_dim_weight_mu_cache_{_jac_hash_prefix}.pt"),
                            cache_key=ae_checkpoint_cache_key,
                        )
                        n_active = int((dim_weight > dim_weight.mean()).sum().item())
                        logger.info(
                            f"Per-dimension loss weight measured (jacobian, full dataset, "
                            f"var_source=mu): {n_active}/{dim_weight.numel()} dims above the mean weight, "
                            f"(var > {dim_reweight_min_var}), "
                            f"weight range [{dim_weight.min().item():.4f}, {dim_weight.max().item():.4f}]"
                        )
                    # dim_reweight_floor: minimum weight as a fraction of the mean; 0.0 (default) is a no-op.
                    dim_reweight_floor = float(cfg.get("dim_reweight_floor", 0.0))
                    if dim_reweight_floor > 0.0:
                        pre_min, pre_max = dim_weight.min().item(), dim_weight.max().item()
                        dim_weight = dim_weight.clamp(min=dim_reweight_floor)
                        dim_weight = dim_weight / dim_weight.mean()
                        logger.info(
                            f"dim_reweight_floor={dim_reweight_floor}: floored dim_weight "
                            f"(was [{pre_min:.4f}, {pre_max:.4f}]) and renormalized to mean 1, "
                            f"new range [{dim_weight.min().item():.4f}, {dim_weight.max().item():.4f}]"
                        )
                    interpolant.dim_weight = dim_weight

                if dim_standardize:
                    if resume_dim_std is not None:
                        dim_mean = torch.as_tensor(resume_dim_mean, device=device_id)
                        scale_factor = 1.0 / resume_dim_std.to(device_id)
                        logger.info("Using resumed per-dimension latent standardization stats")
                    else:
                        dstd_hash_prefix = (ae_checkpoint_cache_key or "nokey")[:8]
                        dim_mean, dim_std = measure_dim_standardize_stats_dataset(
                            data_loader=train_loader,
                            autoencoder=autoencoder,
                            device_id=device_id,
                            cache_path=os.path.join(data_dir, f"dim_standardize_cache_{dstd_hash_prefix}.pt"),
                            cache_key=ae_checkpoint_cache_key,
                        )
                        scale_factor = 1.0 / dim_std
                        logger.info(
                            "Per-dimension latent standardization measured (full dataset): "
                            f"mean range [{dim_mean.min().item():.4f}, {dim_mean.max().item():.4f}], "
                            f"std range [{dim_std.min().item():.4f}, {dim_std.max().item():.4f}]"
                        )
                    if dim_reweight:
                        # dim_weight was measured on the raw latent; rescale by std_j^2 to
                        # cancel dim_standardize's effect on it.
                        dim_std_for_correction = 1.0 / scale_factor
                        interpolant.dim_weight = interpolant.dim_weight * dim_std_for_correction.pow(2)
                        logger.info(
                            "dim_reweight + dim_standardize both enabled -- rescaled "
                            "dim_weight by dim_std^2 so reweighting's effective per-"
                            "dimension emphasis is unchanged from the dim_standardize=false case"
                        )
                first_batch = False

            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                posterior = autoencoder.encode(batch)
                z = (posterior.sample() if cfg.get("sample", True) else posterior.mode()).detach()

                x_1 = scale_factor * (z - dim_mean)  # (n_total, pe_dim); dim_mean=0.0 no-op unless dim_standardize
                x_1, mask = to_dense_batch(x_1, batch.batch)  # (bs, n, pe_dim), (bs, n)
                x_t, t, x_0 = interpolant.corrupt_batch(x_1, mask)

                if random.random() < cfg.sc_prob:
                    with torch.no_grad():
                        x_sc = model(x_t, t, mask)
                else:
                    x_sc = None

                out = model(x_t, t, mask, x_sc)  # (bs, n, pe_dim)

                norm_loss_weight = cfg.get("norm_loss_weight", 0.0)
                # Shared clean-latent estimate for the norm loss below.
                x_1_pred = predict_x1(cfg.param, x_t, t, out) if norm_loss_weight > 0.0 else None

                if cfg.param == "x_0":
                    loss = fm_mse_weight * interpolant.criterion(x_0, t, out, mask)
                elif cfg.param == "x_1":
                    loss = fm_mse_weight * interpolant.criterion(x_1, t, out, mask)
                else:
                    loss = fm_mse_weight * interpolant.criterion(x_1 - x_0, t, out, mask)

                # ── Norm loss: match per-graph mean ‖z‖₂ of predicted vs encoder ── off by default.
                norm_loss_val = 0.0
                if norm_loss_weight > 0.0:
                    # Per-graph mean ‖z‖₂: avg node L2 norm within each graph
                    n_per_graph = mask.float().sum(dim=1).clamp(min=1.0)  # (bs,)
                    norm_pred = (x_1_pred.norm(dim=-1) * mask).sum(dim=1) / n_per_graph
                    norm_tgt  = (x_1.norm(dim=-1)      * mask).sum(dim=1) / n_per_graph
                    norm_loss = F.mse_loss(norm_pred, norm_tgt.detach())
                    norm_loss_val = float(norm_loss.detach().cpu())
                    loss = loss + norm_loss_weight * norm_loss

            loss.backward()
            if cfg.gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            if master_process:
                ema_model.update()
                loss_window.append(float(loss.detach().cpu()))
                if norm_loss_weight > 0.0:
                    norm_loss_window.append(norm_loss_val)

            if step % int(cfg.log_after) == 0 and master_process:
                avg_loss = sum(loss_window) / len(loss_window)
                log_msg = f"Step [{step}/{int(cfg.num_steps)}] | loss: {avg_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}"
                log_dict = dict(train_loss=avg_loss, lr=optimizer.param_groups[0]["lr"], time=time.time() - start_time)
                if norm_loss_weight > 0.0 and norm_loss_window:
                    avg_norm_loss = sum(norm_loss_window) / len(norm_loss_window)
                    log_msg += f" | norm_loss: {avg_norm_loss:.4f}"
                    log_dict["norm_loss"] = avg_norm_loss
                    norm_loss_window = []
                logger.info(log_msg)
                if cfg.wandb_project is not None:
                    wandb.log(log_dict, step=step)
                loss_window = []
                start_time = time.time()

            if step % int(cfg.val_after) == 0 and step >= int(cfg.val_after):
                logger.info(f"Completed epoch on [{device_id}]")
                if device_count > 1:
                    epoch += 1
                    train_loader.sampler.set_epoch(epoch)
                if master_process:
                    logger.info("Evaluating model")
                    ema_model.eval()
                    generated_samples = generate_samples(
                        interpolant=interpolant,
                        model=ema_model,
                        autoencoder=autoencoder,
                        scale_factor=scale_factor,
                        dim_mean=dim_mean,
                        node_distribution=node_distribution,
                        cfg=cfg,
                        device_id=device_id,
                        atom_decoder=getattr(train_dataset, "atom_decoder", None),
                        q=q,
                        device_count=device_count,
                    )
                    metrics = evaluate_samples(
                        generated_samples=generated_samples,
                        cfg=cfg,
                        device_id=device_id,
                        step=step,
                        sampling_metrics=sampling_metrics,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                    )
                    current_value = metrics[selection_metric]
                    is_new_best = best_value is None or (
                        current_value > best_value if selection_mode == "max" else current_value < best_value
                    )
                    if is_new_best and selection_ratio_max is not None:
                        eval_ratio = metrics.get("ratio", float("inf"))
                        if eval_ratio > selection_ratio_max:
                            is_new_best = False
                            logger.info(
                                f"Step {step}: {selection_metric}={current_value:.4f} would be a new "
                                f"best, but ratio={eval_ratio:.2f} > selection_ratio_max="
                                f"{selection_ratio_max} -- not saving as best"
                            )
                    if is_new_best:
                        best_value = current_value
                        if cfg.checkpoint is not None and not save_every_eval:
                            extra_ckpt = dict(best_val=best_value, **_scale_ckpt_fields(scale_factor, dim_mean))
                            if dim_reweight:
                                extra_ckpt["dim_weight"] = interpolant.dim_weight.detach().cpu()
                            save_checkpoint(
                                checkpoint_file,
                                step,
                                ema_model,
                                optimizer,
                                ema=True,
                                **extra_ckpt,
                            )
                    if cfg.checkpoint is not None and save_every_eval:
                        checkpoint_path = Path(checkpoint_file)
                        save_path = str(
                            checkpoint_path.with_name(
                                f"{checkpoint_path.stem}{f'step_{step}'}{checkpoint_path.suffix}"
                            )
                        )
                        extra_ckpt = dict(best_val=best_value, **_scale_ckpt_fields(scale_factor, dim_mean))
                        if dim_reweight:
                            extra_ckpt["dim_weight"] = interpolant.dim_weight.detach().cpu()
                        save_checkpoint(
                            save_path,
                            step,
                            ema_model,
                            optimizer,
                            ema=True,
                            **extra_ckpt,
                        )
                    ema_model.train()
                if device_count > 1:
                    # Master-only eval above can outlast NCCL's timeout; barrier keeps other ranks in sync.
                    torch.distributed.barrier()

            step += 1


if __name__ == "__main__":
    main()
