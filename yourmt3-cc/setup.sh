#!/bin/bash
# setup.sh — one-shot YourMT3+ installation for Compute Canada
#
# Run this from your $SCRATCH directory:
#   cd $SCRATCH
#   bash yourmt3-cc/setup.sh
#
# After it completes, the repo will be at $SCRATCH/YourMT3
# with a ready-to-use .venv inside it.

set -e

INSTALL_DIR="${SCRATCH}/YourMT3"
REPO_URL="https://huggingface.co/spaces/mimbres/YourMT3"

echo "============================================"
echo "  YourMT3+ Setup for Compute Canada"
echo "  Target: ${INSTALL_DIR}"
echo "============================================"

# ── 1. Modules ────────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 gcc/12.3 python/3.11
echo "[1/5] Modules loaded."

# ── 2. Clone (includes model checkpoints via git LFS) ─────────────────────────
if [ -d "${INSTALL_DIR}/.git" ]; then
    echo "[2/5] Repo already exists — pulling latest + LFS objects."
    cd "${INSTALL_DIR}"
    git pull
    git lfs pull
else
    echo "[2/5] Cloning from HuggingFace (this downloads ~2 GB of checkpoints)..."
    git lfs install --skip-repo 2>/dev/null || true
    git clone "${REPO_URL}" "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
    git lfs pull
fi

# ── 3. Virtual environment ────────────────────────────────────────────────────
echo "[3/5] Creating Python virtual environment..."
python -m venv "${INSTALL_DIR}/.venv"
source "${INSTALL_DIR}/.venv/bin/activate"
pip install --upgrade pip --quiet

# ── 4. Dependencies ───────────────────────────────────────────────────────────
echo "[4/5] Installing dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements_cc.txt" --quiet
echo "Dependencies installed."

# ── 5. Copy custom scripts ───────────────────────────────────────────────────
echo "[5/5] Copying custom scripts..."
cp "${SCRIPT_DIR}/model_helper.py"    "${INSTALL_DIR}/model_helper.py"
cp "${SCRIPT_DIR}/transcribe_stems.py" "${INSTALL_DIR}/transcribe_stems.py"
cp "${SCRIPT_DIR}/inference.py"        "${INSTALL_DIR}/inference.py"
cp "${SCRIPT_DIR}/config.json.example" "${INSTALL_DIR}/config.json.example"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To activate the environment:"
echo "    module load StdEnv/2023 gcc/12.3 python/3.11"
echo "    source ${INSTALL_DIR}/.venv/bin/activate"
echo ""
echo "  See README.md for usage examples."
echo "============================================"
