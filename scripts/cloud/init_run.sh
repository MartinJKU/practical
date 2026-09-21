#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
miq_cloud_configure_run

[[ -x "${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/python" ]] \
  || miq_cloud_die "sealed bundle is missing; run stage_bundle.sh first"
PYTHONNOUSERSITE=1 "${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/python" -I \
  -m miq_grpo.offline_bundle verify \
  --bundle-root "${MIQ_OFFLINE_BUNDLE_ROOT}" \
  --project-root "${MIQ_PROJECT_ROOT}"

# Reserve exactly one experiments/<run-id> namespace. A partial initialization
# remains reserved after failure so its outputs can never be mistaken for a
# fresh reportable run.
mkdir -p "$(dirname "${MIQ_CLOUD_SUITE_ROOT}")"
mkdir "${MIQ_CLOUD_SUITE_ROOT}" \
  || miq_cloud_die "run namespace already exists; choose a fresh MIQ_CLOUD_RUN_ID"
mkdir -p "${MIQ_PIPELINE_STATE}/attempts" "${MIQ_PIPELINE_STATE}/locks" "${MIQ_CLOUD_LOG_ROOT}"
miq_cloud_lock init-run
miq_cloud_begin_attempt init-run

MANIFEST_HASH="$("${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/python" - \
  "${MIQ_OFFLINE_BUNDLE_ROOT}/offline_bundle_manifest.json" <<'PY'
import hashlib
import sys

with open(sys.argv[1], "rb") as handle:
    print(hashlib.sha256(handle.read()).hexdigest())
PY
)"
miq_cloud_write_once "${MIQ_PIPELINE_STATE}/run_initialized.txt" \
  "initialized_at=$(miq_cloud_utc)" \
  "run_id=${MIQ_CLOUD_RUN_ID}" \
  "suite_root=${MIQ_CLOUD_SUITE_ROOT}" \
  "project_root=${MIQ_PROJECT_ROOT}" \
  "offline_bundle_root=${MIQ_OFFLINE_BUNDLE_ROOT}" \
  "offline_bundle_manifest_sha256=${MANIFEST_HASH}" \
  "data_root=${MIQ_DATA_ROOT}" \
  "run_root=${MIQ_RUN_ROOT}" \
  "evaluation_root=${MIQ_EVAL_RESULTS_ROOT}" \
  "report_root=${MIQ_REPORT_ROOT}"
echo "Initialized immutable cloud run: ${MIQ_CLOUD_SUITE_ROOT}"
