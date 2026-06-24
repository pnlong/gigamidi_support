#!/bin/bash
#SBATCH --job-name=ymt3-stems
#SBATCH --output=logs/stems_%A_%a.out
#SBATCH --error=logs/stems_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2:00:00
#SBATCH --account=def-pasquier

# ── Usage ─────────────────────────────────────────────────────────────────────
# Transcribes per-instrument audio stems → one MIDI per stem.
# Edit STEMS_JSON and OUTPUT_DIR below, then:
#   mkdir -p logs
#   sbatch slurm/transcribe_stems.sh
#
# The --array=0-3 runs 4 parallel workers (one MIG slice each).
# Each worker handles a shard of the stem list automatically.
# Change --array=0-0 for a single-GPU run.
# ─────────────────────────────────────────────────────────────────────────────

EXP_ID="ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100"
PRESET="yptf_single"   # dedicated single-instrument checkpoint; use mc13_full_plus_256 for the MoE multi-track model
STEMS_JSON="${SCRATCH}/data/stems_manifest.json"
OUTPUT_DIR="${SCRATCH}/output/stems_midi"

# ─────────────────────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 gcc/12.3 python/3.11

cd "${SCRATCH}/YourMT3"
source .venv/bin/activate

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "SLURM array task ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

python transcribe_stems.py \
    --stems_json "${STEMS_JSON}" \
    --exp_id "${EXP_ID}" \
    --preset "${PRESET}" \
    --output_dir "${OUTPUT_DIR}" \
    --stats \
    --stats_out "${OUTPUT_DIR}/stats_${SLURM_ARRAY_TASK_ID}.json" \
    --gpu_index "${SLURM_ARRAY_TASK_ID}" \
    --num_gpus "${SLURM_ARRAY_TASK_COUNT}"
