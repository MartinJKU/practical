#!/usr/bin/env bash

# Source this file from a direct-process Linux runner. It intentionally has no
# scheduler, module-system, or provider assumptions.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "usage: source scripts/activate_portable.sh training|evaluation" >&2
  exit 2
fi

set -euo pipefail

PHASE="${1:?usage: source activate_portable.sh training|evaluation}"
: "${MIQ_OFFLINE_BUNDLE_ROOT:?MIQ_OFFLINE_BUNDLE_ROOT is required}"
: "${MIQ_PROJECT_ROOT:?MIQ_PROJECT_ROOT is required}"

if [[ "${PHASE}" != "training" && "${PHASE}" != "evaluation" ]]; then
  echo "Unknown phase: ${PHASE}" >&2
  return 2
fi

BUNDLE_PYTHON="${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/python"
if [[ ! -x "${BUNDLE_PYTHON}" ]]; then
  echo "Offline bundle lacks its Python environment: ${BUNDLE_PYTHON}" >&2
  return 2
fi
if ! PYTHONNOUSERSITE=1 "${BUNDLE_PYTHON}" -I -m miq_grpo.offline_bundle verify \
  --bundle-root "${MIQ_OFFLINE_BUNDLE_ROOT}" \
  --project-root "${MIQ_PROJECT_ROOT}"; then
  echo "Offline bundle failed immutable-readiness validation." >&2
  return 2
fi

source "${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MIQ_OFFLINE_BUNDLE_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/offline_bundle_manifest.json"
export MIQ_ENV_LOCK="${MIQ_OFFLINE_BUNDLE_ROOT}/requirements.lock"
export MIQ_WHEELHOUSE_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse.sha256"
export MIQ_SOURCE_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/source_manifest.json"
export MIQ_BASE_MODEL_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets/assets_manifest.json"

if [[ "${PHASE}" == "training" ]]; then
  export HF_HOME="${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets/hf_cache"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export MIQ_TRAIN_ASSETS_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets/assets_manifest.json"
  unset MIQ_EVAL_ASSETS_MANIFEST MIQ_EVAL_REPO
else
  export HF_HOME="${MIQ_OFFLINE_BUNDLE_ROOT}/evaluation_assets/hf_cache"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export MIQ_EVAL_ASSETS_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/evaluation_assets/assets_manifest.json"
  export MIQ_EVAL_REPO="${MIQ_OFFLINE_BUNDLE_ROOT}/vendor/moleculariq-eval"
  unset MIQ_TRAIN_ASSETS_MANIFEST
fi
