#!/bin/bash
#SBATCH --job-name=ymt3-batch
#SBATCH --output=logs/batch_%A_%a.out
#SBATCH --error=logs/batch_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3:00:00
#SBATCH --account=def-pasquier

# ── Usage ─────────────────────────────────────────────────────────────────────
# Transcribes full audio mixes → multi-instrument MIDI (one MIDI per input).
# Edit CONFIG_JSON and OUTPUT_DIR below, then:
#   mkdir -p logs
#   sbatch slurm/inference_batch.sh
#
# The config JSON must list all input audio paths under "data".
# See config.json.example for the format.
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_JSON="${SCRATCH}/YourMT3/config.json.example"   # <-- point to your config
OUTPUT_DIR="${SCRATCH}/output/batch_midi"

# ─────────────────────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 gcc/12.3 python/3.11

cd "${SCRATCH}/YourMT3"
source .venv/bin/activate

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "SLURM array task ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

python inference.py \
    --config_json "${CONFIG_JSON}" \
    --output "${OUTPUT_DIR}" \
    --device cuda \
    --gpu_index "${SLURM_ARRAY_TASK_ID}" \
    --num_gpus "${SLURM_ARRAY_TASK_COUNT}"
