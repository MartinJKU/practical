#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
miq_cloud_configure_run
miq_cloud_require_initialized
miq_cloud_lock preprocess
miq_cloud_begin_attempt preprocess
# shellcheck disable=SC1091
source "${MIQ_PROJECT_ROOT}/scripts/activate_portable.sh" training
miq_cloud_load_model_info

POINTER="${MIQ_PIPELINE_STATE}/dataset_bundle_path.txt"
if [[ -f "${POINTER}" ]]; then
  echo "Frozen dataset pointer already exists; validating instead of regenerating."
  miq_cloud_load_dataset
  for task in count index constraint; do
    python -m miq_grpo.preflight dataset --bundle "${MIQ_DATA_BUNDLE}" --task "${task}"
  done
  exit 0
fi
if [[ -e "${MIQ_DATA_ROOT}" ]]; then
  miq_cloud_die "data root exists without a published pointer; choose a fresh MIQ_CLOUD_RUN_ID"
fi

export MIQ_STAGED_TRAIN_POOL="${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets/train_pool/dataset"
export MIQ_STAGED_TRAIN_POOL_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets/train_pool/manifest.json"
python -m miq_grpo.dataset_builder \
  --config "${MIQ_PROJECT_ROOT}/configs/preprocessing.yaml" \
  --write-bundle-pointer "${POINTER}"
miq_cloud_load_dataset
for task in count index constraint; do
  python -m miq_grpo.preflight dataset --bundle "${MIQ_DATA_BUNDLE}" --task "${task}"
done
miq_cloud_write_once "${MIQ_PIPELINE_STATE}/gates/preprocess_complete.txt" \
  "completed_at=$(miq_cloud_utc)" \
  "dataset_bundle=${MIQ_DATA_BUNDLE}"
echo "Frozen training dataset ready: ${MIQ_DATA_BUNDLE}"
