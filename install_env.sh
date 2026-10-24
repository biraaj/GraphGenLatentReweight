#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Removing existing ljrw environment (if any)..."
conda env remove -n ljrw --yes 2>/dev/null || true

echo "==> Creating conda environment from environment.yml..."
conda env create -f "$SCRIPT_DIR/environment.yml" --environment-spec environment.yml

echo "==> Installing torch from the CUDA 13.0 wheel index..."
conda run -n ljrw pip install --no-build-isolation --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0

echo "==> Installing remaining pip packages from PyPI..."
conda run -n ljrw pip install --no-build-isolation -r "$SCRIPT_DIR/requirements.txt"

echo "==> Installing torch-scatter from PyG wheels..."
conda run -n ljrw pip install --force-reinstall torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.11.0+cu130.html

echo "==> Done. Activate with: conda activate ljrw"
echo ""
echo "NOTE: Add this to your shell rc (e.g. ~/.bashrc) or prefix your commands:"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
