#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
miq_cloud_configure_base

mkdir -p "${MIQ_CLOUD_ROOT}/offline_bundles"
STAGE_STATE="${MIQ_OFFLINE_BUNDLE_ROOT}.cloud_state"
STAGE_LOGS="${MIQ_OFFLINE_BUNDLE_ROOT}.cloud_logs"
if [[ -e "${MIQ_OFFLINE_BUNDLE_ROOT}" || -e "${STAGE_STATE}" || -e "${STAGE_LOGS}" ]]; then
  miq_cloud_die "bundle or staging records already exist; choose a fresh MIQ_OFFLINE_BUNDLE_ROOT"
fi
mkdir -p "${STAGE_STATE}/attempts" "${STAGE_LOGS}"
export MIQ_PIPELINE_STATE="${STAGE_STATE}"
export MIQ_CLOUD_LOG_ROOT="${STAGE_LOGS}"
miq_cloud_lock stage-bundle
miq_cloud_begin_attempt stage-bundle

echo "Staging a new sealed bundle at ${MIQ_OFFLINE_BUNDLE_ROOT}"
echo "This phase requires internet access and Python 3.11. The final path must not be moved."
bash "${MIQ_PROJECT_ROOT}/scripts/stage_offline_bundle.sh"

PYTHONNOUSERSITE=1 "${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/python" -I \
  -m miq_grpo.offline_bundle verify \
  --bundle-root "${MIQ_OFFLINE_BUNDLE_ROOT}" \
  --project-root "${MIQ_PROJECT_ROOT}"
miq_cloud_write_once "${STAGE_STATE}/bundle_ready.txt" \
  "completed_at=$(miq_cloud_utc)" \
  "project_root=${MIQ_PROJECT_ROOT}" \
  "offline_bundle_root=${MIQ_OFFLINE_BUNDLE_ROOT}"
echo "Cloud bundle ready: ${MIQ_OFFLINE_BUNDLE_ROOT}"
