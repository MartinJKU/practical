#!/usr/bin/env bash
#SBATCH --job-name=miq-official-eval
#SBATCH --account=CHANGE_ME_BOOSTER
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
source "${MIQ_PROJECT_ROOT}/scripts/activate_leonardo.sh" evaluation

MODEL_KEYS=(baseline count_grpo index_grpo constraint_grpo)
MODEL_KEY="${MODEL_KEYS[${SLURM_ARRAY_TASK_ID}]}"

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID} model=${MODEL_KEY}"
module list
python --version
nvidia-smi

srun python -m miq_grpo.preflight environment --require-cuda
srun python -m miq_grpo.official_eval \
  --config "${MIQ_PROJECT_ROOT}/configs/evaluation.yaml" \
  --model-key "${MODEL_KEY}"
