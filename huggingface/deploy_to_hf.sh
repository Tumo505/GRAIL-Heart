#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# deploy_to_hf.sh — Deploy GRAIL-Heart to Hugging Face Spaces
#
# Usage:
#   chmod +x huggingface/deploy_to_hf.sh
#   ./huggingface/deploy_to_hf.sh <hf_username>
#
# Example:
#   ./huggingface/deploy_to_hf.sh Tumo505
#
# Prerequisites:
#   - pip install huggingface_hub
#   - huggingface-cli login  (authenticate first)
#   - Run: python huggingface/prepare_demo_data.py  (generate assets)
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

HF_USER="${1:?Usage: $0 <hf_username>}"
SPACE_NAME="grail-heart-demo"
SPACE_ID="${HF_USER}/${SPACE_NAME}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMP_DIR="/tmp/hf-grail-heart-deploy"

echo "=============================================="
echo "Deploying GRAIL-Heart to HF Space: ${SPACE_ID}"
echo "=============================================="

# 1. Verify assets exist
echo ""
echo "[1/5] Checking assets …"

required_files=(
    "${SCRIPT_DIR}/app.py"
    "${SCRIPT_DIR}/README.md"
    "${SCRIPT_DIR}/requirements.txt"
    "${SCRIPT_DIR}/.gitattributes"
    "${SCRIPT_DIR}/model/best.pt"
    "${SCRIPT_DIR}/demo_data/lr_database_cache.csv"
)

for f in "${required_files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  ERROR: Missing file: $f"
        echo "  Run first: python huggingface/prepare_demo_data.py"
        exit 1
    fi
done
echo "  All required files present."

# 2. Create the HF Space (idempotent)
echo ""
echo "[2/5] Creating HF Space …"
huggingface-cli repo create "${SPACE_NAME}" \
    --type space \
    --space-sdk streamlit \
    2>/dev/null || echo "  Space already exists (OK)"

# 3. Clone into temp dir
echo ""
echo "[3/5] Cloning Space repo …"
rm -rf "${TEMP_DIR}"
git clone "https://huggingface.co/spaces/${SPACE_ID}" "${TEMP_DIR}"
cd "${TEMP_DIR}"

# 4. Copy files
echo ""
echo "[4/5] Copying files …"
git lfs install

# Top-level files
cp "${SCRIPT_DIR}/app.py" .
cp "${SCRIPT_DIR}/README.md" .
cp "${SCRIPT_DIR}/requirements.txt" .
cp "${SCRIPT_DIR}/.gitattributes" .

# Model
mkdir -p model
cp "${SCRIPT_DIR}/model/best.pt" model/

# Demo data
mkdir -p demo_data/tables
cp "${SCRIPT_DIR}/demo_data/"*.csv demo_data/ 2>/dev/null || true
cp "${SCRIPT_DIR}/demo_data/"*.h5ad demo_data/ 2>/dev/null || true
cp "${SCRIPT_DIR}/demo_data/tables/"*.csv demo_data/tables/ 2>/dev/null || true

# Source modules
if [ -d "${SCRIPT_DIR}/src" ]; then
    cp -r "${SCRIPT_DIR}/src" .
fi

echo "  Files copied."
ls -la

# 5. Commit and push
echo ""
echo "[5/5] Pushing to Hugging Face …"
git add .
git commit -m "Deploy GRAIL-Heart Streamlit app with model and demo data"
git push

echo ""
echo "=============================================="
echo "Deployment complete!"
echo "View at: https://huggingface.co/spaces/${SPACE_ID}"
echo "=============================================="
