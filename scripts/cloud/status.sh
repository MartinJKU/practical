#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
miq_cloud_configure_run
miq_cloud_require_initialized

echo "run_id=${MIQ_CLOUD_RUN_ID}"
echo "suite_root=${MIQ_CLOUD_SUITE_ROOT}"
if [[ -f "${MIQ_PIPELINE_STATE}/dataset_bundle_path.txt" ]]; then
  echo "preprocess=complete"
else
  echo "preprocess=pending"
fi
if [[ -f "${MIQ_PIPELINE_STATE}/gates/gpu_smoke_complete.txt" ]]; then
  echo "gpu_smoke=complete"
else
  echo "gpu_smoke=pending"
fi

for task in count index constraint; do
  case "${task}" in
    count) experiment="qwen25_05b_miq_count_grpo_v1" ;;
    index) experiment="qwen25_05b_miq_index_grpo_v1" ;;
    constraint) experiment="qwen25_05b_miq_constraint_grpo_v1" ;;
  esac
  run_dir="${MIQ_RUN_ROOT}/${experiment}"
  if [[ -f "${run_dir}/training_complete.json" ]]; then
    state="complete"
  elif [[ -d "${run_dir}" ]]; then
    state="checkpointed_or_incomplete"
  else
    state="pending"
  fi
  echo "train_${task}=${state}"
done

for model_key in baseline count_grpo index_grpo constraint_grpo; do
  count=0
  model_root="${MIQ_EVAL_RESULTS_ROOT}/${model_key}"
  if [[ -d "${model_root}" ]]; then
    count="$(find "${model_root}" -mindepth 2 -maxdepth 2 -type f -name _COMPLETE | wc -l)"
  fi
  if [[ "${count}" -eq 1 ]]; then
    state="complete"
  elif [[ "${count}" -gt 1 ]]; then
    state="invalid_multiple_complete_runs"
  elif [[ -d "${model_root}" ]]; then
    state="incomplete"
  else
    state="pending"
  fi
  echo "evaluate_${model_key}=${state}"
done

if [[ -f "${MIQ_REPORT_ROOT}/_COMPLETE" ]]; then
  echo "report=complete"
elif [[ -e "${MIQ_REPORT_ROOT}" ]]; then
  echo "report=incomplete"
else
  echo "report=pending"
fi
echo "logs=${MIQ_CLOUD_LOG_ROOT}"
echo "attempt_status=${MIQ_PIPELINE_STATE}/attempts"
