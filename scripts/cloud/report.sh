#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
miq_cloud_configure_run
miq_cloud_require_initialized
miq_cloud_lock report
miq_cloud_begin_attempt report
# shellcheck disable=SC1091
source "${MIQ_PROJECT_ROOT}/scripts/activate_portable.sh" evaluation
miq_cloud_load_model_info

if [[ -f "${MIQ_REPORT_ROOT}/_COMPLETE" ]]; then
  echo "Report is already complete; immutable report was not changed."
  exit 0
fi
[[ ! -e "${MIQ_REPORT_ROOT}" ]] \
  || miq_cloud_die "report root exists without _COMPLETE; choose a fresh run ID for a clean rebuild"
for model_key in baseline count_grpo index_grpo constraint_grpo; do
  model_root="${MIQ_EVAL_RESULTS_ROOT}/${model_key}"
  completed_count=0
  if [[ -d "${model_root}" ]]; then
    completed_count="$(find "${model_root}" -mindepth 2 -maxdepth 2 -type f -name _COMPLETE | wc -l)"
  fi
  [[ "${completed_count}" -eq 1 ]] \
    || miq_cloud_die "expected exactly one complete official run for ${model_key}, found ${completed_count}"
done

python -m miq_grpo.reporting \
  --config "${MIQ_PROJECT_ROOT}/configs/evaluation.yaml" \
  --output-root "${MIQ_REPORT_ROOT}"
[[ -f "${MIQ_REPORT_ROOT}/_COMPLETE" && -f "${MIQ_REPORT_ROOT}/report_manifest.json" ]] \
  || miq_cloud_die "reporting did not publish a complete report"
miq_cloud_write_once "${MIQ_PIPELINE_STATE}/gates/report_complete.txt" \
  "completed_at=$(miq_cloud_utc)" \
  "report_root=${MIQ_REPORT_ROOT}"
echo "Report tables and plots ready: ${MIQ_REPORT_ROOT}"
