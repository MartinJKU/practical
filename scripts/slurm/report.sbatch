#!/usr/bin/env bash
#SBATCH --job-name=miq-report
#SBATCH --account=CHANGE_ME_DCGP
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
source "${MIQ_PROJECT_ROOT}/scripts/activate_leonardo.sh" evaluation

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID}"
module list
python --version

srun python -m miq_grpo.reporting \
  --config "${MIQ_PROJECT_ROOT}/configs/evaluation.yaml" \
  --output-root "${MIQ_REPORT_ROOT}"

