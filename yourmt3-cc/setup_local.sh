#!/bin/bash
# One-shot YourMT3+ install for local GPU machines (not Compute Canada).
#
# Usage:
#   bash yourmt3-cc/setup_local.sh
#   source YourMT3/.venv/bin/activate
#
# Requires: git, git-lfs, python3, ~2 GB for checkpoints

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${YOURMT3_DIR:-${REPO_ROOT}/YourMT3}"
REPO_URL="https://huggingface.co/spaces/mimbres/YourMT3"

echo "Installing YourMT3+ to ${INSTALL_DIR}"

if [ -d "${INSTALL_DIR}/.git" ]; then
    echo "Repo exists — pulling latest + LFS..."
    cd "${INSTALL_DIR}"
    git pull
    git lfs pull
else
    git lfs install --skip-repo 2>/dev/null || true
    git clone "${REPO_URL}" "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
    git lfs pull
fi

python3 -m venv "${INSTALL_DIR}/.venv"
source "${INSTALL_DIR}/.venv/bin/activate"
pip install --upgrade pip --quiet
pip install -r "${REPO_ROOT}/yourmt3-cc/requirements_cc.txt" --quiet

SCRIPT_DIR="${REPO_ROOT}/yourmt3-cc"
cp "${SCRIPT_DIR}/model_helper.py"    "${INSTALL_DIR}/model_helper.py"
cp "${SCRIPT_DIR}/transcribe_stems.py" "${INSTALL_DIR}/transcribe_stems.py"
cp "${SCRIPT_DIR}/inference.py"        "${INSTALL_DIR}/inference.py"
cp "${SCRIPT_DIR}/config.json.example" "${INSTALL_DIR}/config.json.example"

echo ""
echo "Done. Activate with:"
echo "  source ${INSTALL_DIR}/.venv/bin/activate"
echo ""
echo "Then run AMT for DEAM:"
echo "  python yourmt3-cc/generate_config.py --dataset deam --output /tmp/deam_amt.json"
echo "  cd ${INSTALL_DIR} && python inference.py --config_json /tmp/deam_amt.json --device cuda"
