#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/cloud/evaluate.sh baseline|count_grpo|index_grpo|constraint_grpo" >&2
  exit 2
}

[[ "$#" -eq 1 ]] || usage
MODEL_KEY="$1"
case "${MODEL_KEY}" in
  baseline|count_grpo|index_grpo|constraint_grpo) ;;
  *) usage ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
miq_cloud_configure_run
miq_cloud_require_initialized
miq_cloud_lock "evaluate-${MODEL_KEY}"
miq_cloud_begin_attempt "evaluate-${MODEL_KEY}"
# shellcheck disable=SC1091
source "${MIQ_PROJECT_ROOT}/scripts/activate_portable.sh" evaluation
miq_cloud_load_model_info

# Keep the official benchmark test-only: no model is evaluated until all three
# production runs have been finalized independently of benchmark outcomes.
for task in count index constraint; do
  case "${task}" in
    count) experiment="qwen25_05b_miq_count_grpo_v1" ;;
    index) experiment="qwen25_05b_miq_index_grpo_v1" ;;
    constraint) experiment="qwen25_05b_miq_constraint_grpo_v1" ;;
  esac
  [[ -f "${MIQ_RUN_ROOT}/${experiment}/training_complete.json" ]] \
    || miq_cloud_die "all three production models must finish before official evaluation"
done

MODEL_RESULTS_ROOT="${MIQ_EVAL_RESULTS_ROOT}/${MODEL_KEY}"
mapfile -t COMPLETED_RUNS < <(
  if [[ -d "${MODEL_RESULTS_ROOT}" ]]; then
    find "${MODEL_RESULTS_ROOT}" -mindepth 2 -maxdepth 2 -type f -name _COMPLETE -print | sort
  fi
)
if [[ "${#COMPLETED_RUNS[@]}" -gt 1 ]]; then
  miq_cloud_die "multiple complete official runs exist for ${MODEL_KEY}; refuse benchmark selection"
elif [[ "${#COMPLETED_RUNS[@]}" -eq 1 ]]; then
  echo "Official ${MODEL_KEY} evaluation already complete; immutable result was not changed."
  exit 0
fi

miq_cloud_gpu_preflight
python -m miq_grpo.official_eval \
  --config "${MIQ_PROJECT_ROOT}/configs/evaluation.yaml" \
  --model-key "${MODEL_KEY}"
mapfile -t COMPLETED_RUNS < <(
  find "${MODEL_RESULTS_ROOT}" -mindepth 2 -maxdepth 2 -type f -name _COMPLETE -print | sort
)
[[ "${#COMPLETED_RUNS[@]}" -eq 1 ]] \
  || miq_cloud_die "official evaluation did not publish exactly one complete ${MODEL_KEY} run"
miq_cloud_write_once "${MIQ_PIPELINE_STATE}/gates/evaluate_${MODEL_KEY}_complete.txt" \
  "completed_at=$(miq_cloud_utc)" \
  "complete_marker=${COMPLETED_RUNS[0]}"
echo "Official full-benchmark ${MODEL_KEY} evaluation is complete."
