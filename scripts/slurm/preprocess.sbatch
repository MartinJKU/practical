#!/usr/bin/env bash
#SBATCH --job-name=miq-preprocess
#SBATCH --account=CHANGE_ME_DCGP
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
source "${MIQ_PROJECT_ROOT}/scripts/activate_leonardo.sh" training

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
module list
python --version

export MIQ_STAGED_TRAIN_POOL="${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets/train_pool/dataset"
export MIQ_STAGED_TRAIN_POOL_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets/train_pool/manifest.json"

srun python -m miq_grpo.dataset_builder \
  --config "${MIQ_PROJECT_ROOT}/configs/preprocessing.yaml" \
  --write-bundle-pointer "${MIQ_PIPELINE_STATE}/dataset_bundle_path.txt"

