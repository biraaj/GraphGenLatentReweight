import hydra
import torch
import wandb

torch.autograd.set_detect_anomaly(True)
from loguru import logger

from evaluation.autoencoder import evaluate_autoencoder
from fm import compute_n_nodes_distr
from fm.fm_helpers import build_autoencoder
from utils import (
    setup_everything,
    create_dataloaders,
    instantiate_dataset,
    load_checkpoint
)
from torch.distributed import destroy_process_group


def _fmt(v) -> str:
    if torch.is_tensor(v):
        v = v.item()
    return f"{v:.4f}"


@hydra.main(version_base=None, config_path="./configs", config_name="debug")
def main(cfg):
    device, device_id, device_count, master_process, data_dir, ckpt_dir, dtype, tdtype = setup_everything(cfg)

    logger.info(f"Loading datasets from {data_dir}")
    train_dataset, val_dataset, test_dataset = instantiate_dataset(
        name=cfg.dataset.name,
        data_dir=data_dir,
        cfg=cfg,
        master_process=master_process
    )
    logger.info("Dataset loaded")
    _ = compute_n_nodes_distr(
        train_n_nodes=train_dataset.num_nodes,
        val_n_nodes=val_dataset.num_nodes,
        test_n_nodes=test_dataset.num_nodes
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
        device_count=device_count,
        master_process=master_process
    )

    logger.info(f"Test set size: {len(test_dataset)} graphs")

    model = build_autoencoder(cfg, device_id)
    model.eval()
    load_checkpoint(f"{ckpt_dir}/{cfg.checkpoint}.pt", model, device_id)
    for param in model.parameters():
        param.requires_grad = False

    # Open a separate wandb run so test results don't overwrite the training run
    if master_process and cfg.wandb_project is not None:
        test_run_name = f"{cfg.wandb_name}-ae-test" if cfg.wandb_name else None
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.get("wandb_entity", None),
            name=test_run_name,
            config=dict(cfg),
        )

    if master_process:
        logger.info("Evaluating on TEST set…")

    sample_acc, edge_metrics, node_metrics, acc_by_nodes = evaluate_autoencoder(
        model,
        test_loader,
        device_id,
        cfg.dataset.num_node_types,
        cfg.dataset.num_edge_types,
        cfg.dataset.num_nodes,
        pad_edges_to_max_num_nodes=False,
        disable_metric_sync=True,
    )

    if master_process:
        # ── Build log payload ─────────────────────────────────────────────────
        log_payload = {
            "test/sample_acc": float(sample_acc.cpu()),
            "test/edge_acc":   float(edge_metrics["accuracy"].mean().cpu()),
            "test/edge_prec":  float(edge_metrics["precision"].mean().cpu()),
            "test/edge_rec":   float(edge_metrics["recall"].mean().cpu()),
            "test/edge_f1":    float(edge_metrics["f1"].mean().cpu()),
        }
        if cfg.dataset.num_node_types > 1 and node_metrics.get("accuracy") is not None:
            log_payload.update({
                "test/node_acc":  float(node_metrics["accuracy"].mean().cpu()),
                "test/node_prec": float(node_metrics["precision"].mean().cpu()),
                "test/node_rec":  float(node_metrics["recall"].mean().cpu()),
                "test/node_f1":   float(node_metrics["f1"].mean().cpu()),
            })
        for i, (acc, prec, rec, f1) in enumerate(zip(
            edge_metrics["per_label"]["accuracy"],
            edge_metrics["per_label"]["precision"],
            edge_metrics["per_label"]["recall"],
            edge_metrics["per_label"]["f1"],
        )):
            log_payload[f"test/edge_label{i}_acc"]  = float(acc.cpu())
            log_payload[f"test/edge_label{i}_prec"] = float(prec.cpu())
            log_payload[f"test/edge_label{i}_rec"]  = float(rec.cpu())
            log_payload[f"test/edge_label{i}_f1"]   = float(f1.cpu())

        # ── Print table ───────────────────────────────────────────────────────
        logger.info("=" * 54)
        logger.info(f"  AE TEST RESULTS  —  {cfg.get('wandb_name', cfg.checkpoint)}")
        logger.info("=" * 54)
        logger.info(f"  {'sample_acc':<22s}  {_fmt(sample_acc)}")
        logger.info(f"  {'edge_acc':<22s}  {_fmt(edge_metrics['accuracy'].mean())}")
        logger.info(f"  {'edge_precision':<22s}  {_fmt(edge_metrics['precision'].mean())}")
        logger.info(f"  {'edge_recall':<22s}  {_fmt(edge_metrics['recall'].mean())}")
        logger.info(f"  {'edge_f1':<22s}  {_fmt(edge_metrics['f1'].mean())}")
        if cfg.dataset.num_node_types > 1 and node_metrics.get("accuracy") is not None:
            logger.info(f"  {'node_acc':<22s}  {_fmt(node_metrics['accuracy'].mean())}")
            logger.info(f"  {'node_f1':<22s}  {_fmt(node_metrics['f1'].mean())}")
        logger.info("-" * 54)
        if len(acc_by_nodes["num_nodes"]) > 0:
            logger.info("  Sample accuracy by node count:")
            for n, a, c in zip(
                acc_by_nodes["num_nodes"].tolist(),
                acc_by_nodes["sample_accuracy"].tolist(),
                acc_by_nodes["num_samples"].tolist(),
            ):
                logger.info(f"    nodes={n:4d}  acc={a:.4f}  n={c}")
        logger.info("=" * 54)

        # ── Wandb log ─────────────────────────────────────────────────────────
        if cfg.wandb_project is not None:
            wandb.log(log_payload)
            wandb.run.summary.update(log_payload)
            wandb.finish()

    if device_count > 1:
        destroy_process_group()


if __name__ == "__main__":
    main()
