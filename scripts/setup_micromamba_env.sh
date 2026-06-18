#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# PanCirc-Fungi: micromamba environment setup script
# ═════════════════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   micromamba installed (see https://mamba.readthedocs.io/ for install)
#
# Usage:
#   bash scripts/setup_micromamba_env.sh
#
# This script:
#   1. Creates a micromamba environment named "pancirc-fungi"
#   2. Installs conda-based dependencies (numpy, pandas, biopython, etc.)
#   3. Installs PyTorch with CUDA 12.x via pip (not available on conda channels)
#   4. Installs pip-only packages (pyfaidx, logomaker)
#   5. Runs a quick validation check
# ═════════════════════════════════════════════════════════════════════════════

set -euo pipefail

ENV_NAME="${1:-pancirc-fungi}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_YML="${SCRIPT_DIR}/env_micromamba.yml"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     PanCirc-Fungi — micromamba Environment Setup                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Environment name : ${ENV_NAME}"
echo "  YAML config      : ${ENV_YML}"
echo ""

# ── 1. Check micromamba ──────────────────────────────────────────────────────
if ! command -v micromamba &> /dev/null; then
    echo "❌ micromamba not found in PATH."
    echo ""
    echo "  Install micromamba with:"
    echo "    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest |"
    echo "      tar -xvj bin/micromamba"
    echo "    ./bin/micromamba shell init -s bash -p ~/micromamba"
    echo "    source ~/.bashrc"
    echo ""
    exit 1
fi

MICROMAMBA_VERSION=$(micromamba --version 2>/dev/null || echo "unknown")
echo "  ✓ micromamba ${MICROMAMBA_VERSION}"

# ── 2. Check GPU / CUDA ──────────────────────────────────────────────────────
HAS_CUDA=false
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    if [ -n "$GPU_NAME" ] && [ "$GPU_NAME" != "No devices were found" ]; then
        echo "  ✓ GPU: ${GPU_NAME}"
        CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        echo "  ✓ NVIDIA driver ${CUDA_DRIVER} detected"
        HAS_CUDA=true
    else
        echo "  ⚠  nvidia-smi found but no GPU devices available (login node)."
        echo "     GPU training will work on compute nodes via sbatch."
        HAS_CUDA=false
    fi
else
    echo "  ⚠  NVIDIA driver not found — PyTorch will use CPU."
    echo "     Submit to a GPU node for GPU training."
fi

echo ""

# ── 3. Create environment ────────────────────────────────────────────────────
if micromamba env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "  ⚠  Environment '${ENV_NAME}' already exists."
    echo "     Run the following to recreate:"
    echo "       micromamba env remove -n ${ENV_NAME}"
    echo "       bash $0"
    echo ""
    read -r -p "  Overwrite existing environment? [y/N] " REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "  Aborted."
        exit 0
    fi
    echo "  Removing existing environment '${ENV_NAME}'..."
    micromamba env remove -n "${ENV_NAME}" -y
fi

echo "  Creating environment '${ENV_NAME}' (conda packages)..."
echo "  (This may take 5-15 minutes depending on network and packages)"
echo ""

time micromamba env create -n "${ENV_NAME}" -f "${ENV_YML}" -y

echo ""
echo "  ✓ Conda packages installed!"
echo ""

# ── 4. Activate environment ──────────────────────────────────────────────────
eval "$(micromamba shell hook --shell=bash)"
micromamba activate "${ENV_NAME}"
echo "  ✓ Environment activated"

# ── 5. Install PyTorch with CUDA 12.x via pip ────────────────────────────────
echo ""
echo "  Installing PyTorch with CUDA 12.x support (via pip)..."
echo "  (This may take 2-5 minutes)"

if $HAS_CUDA; then
    pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
        --quiet --no-input 2>&1 | tail -5
else
    # CPU-only fallback
    pip install torch>=2.4.0 --quiet --no-input 2>&1 | tail -5
fi

echo "  ✓ PyTorch installed"

echo ""
echo "  Installing remaining pip packages (pyfaidx, logomaker)..."
pip install --quiet --no-input pyfaidx>=0.7.2 logomaker>=0.8 2>&1 | tail -3
echo "  ✓ Pip packages installed"

# ── 6. Verify key packages ───────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Validating installation...                                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

FAILURES=0

check_module() {
    local module="$1"
    local import_name="${2:-$1}"
    if python -c "import ${import_name}" 2>/dev/null; then
        echo "  ✓ ${module}"
    else
        echo "  ✗ ${module}  — FAILED (import '${import_name}')"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "  Deep Learning:"
python -c "
import torch
print(f'  ✓ PyTorch {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ✓ CUDA version (torch): {torch.version.cuda}')
"

echo ""
echo "  Data processing:"
check_module "numpy"
check_module "pandas"
check_module "sklearn" "sklearn"
check_module "scipy"

echo ""
echo "  Sequence / genomics:"
check_module "Bio" "Bio"
check_module "pyfaidx"

echo ""
echo "  Visualization:"
check_module "matplotlib"
check_module "seaborn"

echo ""
echo "  Utilities:"
check_module "yaml"
check_module "tqdm"
check_module "h5py"

echo ""
echo "  Interpretability:"
check_module "captum" || true

echo ""
echo "  Training tools:"
check_module "accelerate"
check_module "tensorboard" "tensorboard"

echo ""
if [ "${FAILURES}" -eq 0 ]; then
    echo "  ✅ All core dependencies installed successfully!"
else
    echo "  ⚠  ${FAILURES} module(s) failed to import. Check output above."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Environment ready!                                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Activate:"
echo "    micromamba activate ${ENV_NAME}"
echo ""
echo "  Deactivate:"
echo "    micromamba deactivate"
echo ""
echo "  Remove (if needed):"
echo "    micromamba env remove -n ${ENV_NAME}"
echo ""

# ── SLURM activation snippet ─────────────────────────────────────────────────
echo "───────────────────────────────────────────────────────────────────"
echo "  For SLURM job scripts, add these lines:"
echo "───────────────────────────────────────────────────────────────────"
echo ""
echo '    source "$(micromamba info --base)/etc/profile.d/micromamba.sh"'
echo "    micromamba activate ${ENV_NAME}"
echo ""
echo "  ✓ Done."
