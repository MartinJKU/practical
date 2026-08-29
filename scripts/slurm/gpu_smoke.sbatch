#!/usr/bin/env bash
#SBATCH --job-name=miq-gpu-smoke
#SBATCH --account=CHANGE_ME_BOOSTER
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
source "${MIQ_PROJECT_ROOT}/scripts/activate_leonardo.sh" training
export MIQ_DATA_BUNDLE="$(<"${MIQ_PIPELINE_STATE}/dataset_bundle_path.txt")"

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
module list
python --version
nvidia-smi

srun python -m miq_grpo.preflight environment --require-cuda
srun python -m miq_grpo.preflight rewards

TASKS=(count index constraint)
CONFIGS=(count_grpo.yaml index_grpo.yaml constraint_grpo.yaml)
EXPERIMENTS=(
  qwen25_05b_miq_count_grpo_v1
  qwen25_05b_miq_index_grpo_v1
  qwen25_05b_miq_constraint_grpo_v1
)

for index in "${!TASKS[@]}"; do
  task="${TASKS[${index}]}"
  config="${MIQ_PROJECT_ROOT}/configs/experiments/${CONFIGS[${index}]}"
  run_dir="${MIQ_RUN_ROOT}/smoke/${EXPERIMENTS[${index}]}-smoke-${SLURM_JOB_ID}"

  srun python -m miq_grpo.preflight dataset \
    --bundle "${MIQ_DATA_BUNDLE}" \
    --task "${task}"
  srun python -m miq_grpo.train \
    --config "${config}" \
    --smoke \
    --smoke-max-steps 2 \
    --smoke-stop-after-step 1
  resume_checkpoint="${run_dir}/checkpoint-1"
  if [[ ! -d "${resume_checkpoint}" ]]; then
    echo "Forced smoke interruption did not create ${resume_checkpoint}" >&2
    exit 2
  fi
  srun python -m miq_grpo.train \
    --config "${config}" \
    --smoke \
    --smoke-max-steps 2 \
    --resume-from-checkpoint "${resume_checkpoint}"
  srun python -m miq_grpo.preflight checkpoint \
    --path "${run_dir}/final_model" \
    --reference-path "${MIQ_MODEL_PATH}"
  srun python -m miq_grpo.preflight smoke-metrics \
    --run-dir "${run_dir}" \
    --task "${task}"
done

# Exercise the isolated, offline official-evaluation environment without running
# a benchmark completion, verifier, result processor, or metric computation.
source "${MIQ_PROJECT_ROOT}/scripts/activate_leonardo.sh" evaluation
srun python -m miq_grpo.preflight evaluation-infrastructure \
  --evaluator-repo "${MIQ_EVAL_REPO}" \
  --assets-manifest "${MIQ_EVAL_ASSETS_MANIFEST}" \
  --baseline-path "${MIQ_MODEL_PATH}"
