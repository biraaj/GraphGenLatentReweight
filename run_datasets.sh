#!/bin/bash
# run_datasets.sh -- train AE + FM (baseline) + FM (dimreweight) for every
# dataset in this repo, or just one.
#
# Usage:
#   bash run_datasets.sh                # run every dataset, sequentially
#   bash run_datasets.sh <dataset>      # run just one dataset
#   bash run_datasets.sh --list         # list available dataset names
#
# Output lands in outputs/<dataset>/ (override with OUT=<dir>).
# Every dataset except guacamol trains on a single GPU -- set GPU=<index>
# to pick which one (default 0). On a multi-GPU machine this MUST be set:
# plain `python main_ae.py`/`main_fm.py` only works single-GPU if exactly one
# GPU is visible, since the codebase auto-detects distributed training from
# GPU count and (without torchrun's env vars) fails outright if several GPUs
# are visible at once.
# guacamol's batch_size=2048 needs more than one GPU's memory, so it trains
# via `torchrun --nproc_per_node=$GUACAMOL_GPUS` (default 4) -- set
# GUACAMOL_GPUS=1 and add `batch_size=512` (or similar) yourself if you only
# have one GPU available.
# W&B logging is off by default (the configs ship with placeholder project/entity
# names) -- set WANDB_PROJECT (and optionally WANDB_ENTITY) to turn it on.
set -e

OUT=${OUT:-outputs}
GPU=${GPU:-0}
GUACAMOL_GPUS=${GUACAMOL_GPUS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-null}
WANDB_ENTITY=${WANDB_ENTITY:-null}
WANDB_ARGS="wandb_project=$WANDB_PROJECT wandb_entity=$WANDB_ENTITY"

DATASETS="planar planar_original tree_128 tree_original sbm_modified moses guacamol"

run_dataset() {
  local name=$1
  local root="$OUT/$name"
  echo "=== $name -> $root ==="

  case "$name" in
    planar)
      CUDA_VISIBLE_DEVICES=$GPU python main_ae.py --config-name=planar_ae_vgae root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=planar_fm_vgae root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=planar_fm_vgae_dimreweight root="$root" $WANDB_ARGS
      ;;
    planar_original)
      CUDA_VISIBLE_DEVICES=$GPU python main_ae.py --config-name=planar_original_ae_vgae root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=planar_original_fm_vgae root="$root" $WANDB_ARGS
      ;;
    tree_128)
      CUDA_VISIBLE_DEVICES=$GPU python main_ae.py --config-name=tree_ae_vgae_128 root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=tree_fm_vgae_128 root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=tree_fm_vgae_128_dimreweight root="$root" $WANDB_ARGS
      ;;
    tree_original)
      CUDA_VISIBLE_DEVICES=$GPU python main_ae.py --config-name=tree_original_ae_vgae root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=tree_original_fm_vgae root="$root" $WANDB_ARGS
      ;;
    sbm_modified)
      CUDA_VISIBLE_DEVICES=$GPU python main_ae.py --config-name=sbm_modified_ae_vgae root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=sbm_modified_fm_vgae root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=sbm_modified_fm_vgae_dimreweight root="$root" $WANDB_ARGS
      ;;
    moses)
      CUDA_VISIBLE_DEVICES=$GPU python main_ae.py --config-name=moses_ae_vgae root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=moses_fm_vgae root="$root" $WANDB_ARGS
      CUDA_VISIBLE_DEVICES=$GPU python main_fm.py --config-name=moses_fm_vgae_dimreweight root="$root" $WANDB_ARGS
      ;;
    guacamol)
      CUDA_VISIBLE_DEVICES=$GPU python main_ae.py --config-name=guacamol_ae_vgae root="$root" $WANDB_ARGS
      torchrun --nproc_per_node="$GUACAMOL_GPUS" main_fm.py --config-name=guacamol_fm_vgae root="$root" $WANDB_ARGS
      torchrun --nproc_per_node="$GUACAMOL_GPUS" main_fm.py --config-name=guacamol_fm_vgae_dimreweight root="$root" $WANDB_ARGS
      ;;
    *)
      echo "Unknown dataset: $name" >&2
      echo "Available: $DATASETS" >&2
      exit 1
      ;;
  esac
}

case "$1" in
  --list)
    echo "$DATASETS"
    ;;
  -h|--help)
    sed -n '2,22p' "$0"
    ;;
  "")
    for name in $DATASETS; do
      run_dataset "$name"
    done
    ;;
  *)
    run_dataset "$1"
    ;;
esac
