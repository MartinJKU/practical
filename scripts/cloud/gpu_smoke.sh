#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
miq_cloud_configure_run
miq_cloud_require_initialized
miq_cloud_lock gpu-smoke
miq_cloud_begin_attempt gpu-smoke
# shellcheck disable=SC1091
source "${MIQ_PROJECT_ROOT}/scripts/activate_portable.sh" training
miq_cloud_load_model_info
miq_cloud_load_dataset
miq_cloud_gpu_preflight
python -m miq_grpo.preflight rewards

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
  experiment="${EXPERIMENTS[${index}]}"
  # train.py already supports a stable scheduler-job component for smoke paths.
  # This synthetic value is only a compatibility run tag; no scheduler is used.
  smoke_tag="cloud-smoke-${MIQ_CLOUD_RUN_ID}-${task}"
  run_dir="${MIQ_RUN_ROOT}/smoke/${experiment}-smoke-${smoke_tag}"
  checkpoint="${run_dir}/checkpoint-1"

  python -m miq_grpo.preflight dataset --bundle "${MIQ_DATA_BUNDLE}" --task "${task}"
  if [[ ! -e "${run_dir}" ]]; then
    echo "Starting forced-interruption smoke for ${task}."
    SLURM_JOB_ID="cloud-smoke-${MIQ_CLOUD_RUN_ID}-${task}" python -m miq_grpo.train \
      --config "${config}" \
      --smoke \
      --smoke-max-steps 2 \
      --smoke-stop-after-step 1
  fi

  if [[ ! -f "${run_dir}/training_complete.json" ]]; then
    [[ -d "${checkpoint}" ]] \
      || miq_cloud_die "forced smoke interruption did not create ${checkpoint}"
    validated_checkpoint="$(miq_cloud_checkpoint "${run_dir}" "${checkpoint}")"
    [[ "${validated_checkpoint}" == "${checkpoint}" ]] \
      || miq_cloud_die "validated smoke checkpoint path changed unexpectedly"
    echo "Resuming forced-interruption smoke for ${task} from ${checkpoint}."
    SLURM_JOB_ID="cloud-smoke-${MIQ_CLOUD_RUN_ID}-${task}" python -m miq_grpo.train \
      --config "${config}" \
      --smoke \
      --smoke-max-steps 2 \
      --resume-from-checkpoint "${checkpoint}"
  fi

  [[ -f "${run_dir}/training_complete.json" && -d "${run_dir}/final_model" ]] \
    || miq_cloud_die "smoke training did not complete for ${task}"
  python -m miq_grpo.preflight checkpoint \
    --path "${run_dir}/final_model" \
    --reference-path "${MIQ_MODEL_PATH}"
  python -m miq_grpo.preflight smoke-metrics --run-dir "${run_dir}" --task "${task}"
done

# Validate the isolated official-evaluation installation without consuming any
# official benchmark completion, verifier result, or metric.
# shellcheck disable=SC1091
source "${MIQ_PROJECT_ROOT}/scripts/activate_portable.sh" evaluation
miq_cloud_load_model_info
python -m miq_grpo.preflight evaluation-infrastructure \
  --evaluator-repo "${MIQ_EVAL_REPO}" \
  --assets-manifest "${MIQ_EVAL_ASSETS_MANIFEST}" \
  --baseline-path "${MIQ_MODEL_PATH}"

if [[ ! -e "${MIQ_PIPELINE_STATE}/gates/gpu_smoke_complete.txt" ]]; then
  miq_cloud_write_once "${MIQ_PIPELINE_STATE}/gates/gpu_smoke_complete.txt" \
    "completed_at=$(miq_cloud_utc)" \
    "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
fi
echo "All-task GPU smoke and forced checkpoint/resume gate passed."
