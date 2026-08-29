#!/usr/bin/env bash
#SBATCH --job-name=miq-grpo
#SBATCH --account=CHANGE_ME_BOOSTER
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --signal=USR1@600
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
source "${MIQ_PROJECT_ROOT}/scripts/activate_leonardo.sh" training
export MIQ_DATA_BUNDLE="$(<"${MIQ_PIPELINE_STATE}/dataset_bundle_path.txt")"

CONFIGS=(count_grpo.yaml index_grpo.yaml constraint_grpo.yaml)
EXPERIMENTS=(qwen25_05b_miq_count_grpo_v1 qwen25_05b_miq_index_grpo_v1 qwen25_05b_miq_constraint_grpo_v1)
RESUME_VARIABLES=(MIQ_RESUME_COUNT_CHECKPOINT MIQ_RESUME_INDEX_CHECKPOINT MIQ_RESUME_CONSTRAINT_CHECKPOINT)
CONFIG="${MIQ_PROJECT_ROOT}/configs/experiments/${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
EXPERIMENT="${EXPERIMENTS[${SLURM_ARRAY_TASK_ID}]}"
RESUME_VARIABLE="${RESUME_VARIABLES[${SLURM_ARRAY_TASK_ID}]}"

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID}"
module list
python --version
nvidia-smi

RESUME_ARGS=()
if [[ -n "${!RESUME_VARIABLE:-}" ]]; then
  RESUME_ARGS=(--resume-from-checkpoint "${!RESUME_VARIABLE}")
fi
srun python -m miq_grpo.train --config "${CONFIG}" "${RESUME_ARGS[@]}"

FINAL_MODEL="${MIQ_RUN_ROOT}/${EXPERIMENT}/final_model"
if [[ ! -f "${MIQ_RUN_ROOT}/${EXPERIMENT}/training_complete.json" ]]; then
  echo "Training stopped after checkpointing and needs an explicit resume." >&2
  exit 99
fi
srun python -m miq_grpo.preflight checkpoint \
  --path "${FINAL_MODEL}" \
  --reference-path "${MIQ_MODEL_PATH}"
