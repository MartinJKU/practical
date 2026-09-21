#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/cloud/train.sh count|index|constraint [--resume auto|CHECKPOINT | --fresh]" >&2
  exit 2
}

[[ "$#" -ge 1 ]] || usage
TASK="$1"
shift
RESUME_MODE="auto"
if [[ "$#" -gt 0 ]]; then
  case "$1" in
    --fresh)
      RESUME_MODE="fresh"
      shift
      ;;
    --resume)
      [[ "$#" -ge 2 ]] || usage
      RESUME_MODE="$2"
      shift 2
      ;;
    *) usage ;;
  esac
fi
[[ "$#" -eq 0 ]] || usage

case "${TASK}" in
  count)
    CONFIG_NAME="count_grpo.yaml"
    EXPERIMENT="qwen25_05b_miq_count_grpo_v1"
    ;;
  index)
    CONFIG_NAME="index_grpo.yaml"
    EXPERIMENT="qwen25_05b_miq_index_grpo_v1"
    ;;
  constraint)
    CONFIG_NAME="constraint_grpo.yaml"
    EXPERIMENT="qwen25_05b_miq_constraint_grpo_v1"
    ;;
  *) usage ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
miq_cloud_configure_run
miq_cloud_require_initialized
miq_cloud_lock "train-${TASK}"
miq_cloud_begin_attempt "train-${TASK}"
# shellcheck disable=SC1091
source "${MIQ_PROJECT_ROOT}/scripts/activate_portable.sh" training
miq_cloud_load_model_info
miq_cloud_load_dataset
[[ -f "${MIQ_PIPELINE_STATE}/gates/gpu_smoke_complete.txt" ]] \
  || miq_cloud_die "GPU smoke gate is missing; run gpu_smoke.sh before production training"
miq_cloud_gpu_preflight
python -m miq_grpo.preflight dataset --bundle "${MIQ_DATA_BUNDLE}" --task "${TASK}"

CONFIG="${MIQ_PROJECT_ROOT}/configs/experiments/${CONFIG_NAME}"
RUN_DIR="${MIQ_RUN_ROOT}/${EXPERIMENT}"
FINAL_MODEL="${RUN_DIR}/final_model"
if [[ -f "${RUN_DIR}/training_complete.json" ]]; then
  [[ -d "${FINAL_MODEL}" ]] || miq_cloud_die "completion marker exists but final model is missing"
  python -m miq_grpo.preflight checkpoint \
    --path "${FINAL_MODEL}" \
    --reference-path "${MIQ_MODEL_PATH}"
  echo "${TASK} training is already complete; immutable outputs were not changed."
  exit 0
fi

RESUME_ARGS=()
if [[ -d "${RUN_DIR}" ]]; then
  [[ "${RESUME_MODE}" != "fresh" ]] \
    || miq_cloud_die "run directory exists; --fresh cannot overwrite it"
  RESUME_REQUEST="${RESUME_MODE}"
  [[ "${RESUME_REQUEST}" != "auto" ]] || RESUME_REQUEST="auto"
  RESUME_CHECKPOINT="$(miq_cloud_checkpoint "${RUN_DIR}" "${RESUME_REQUEST}")"
  RESUME_ARGS=(--resume-from-checkpoint "${RESUME_CHECKPOINT}")
  echo "Validated resume checkpoint: ${RESUME_CHECKPOINT}"
else
  if [[ "${RESUME_MODE}" != "auto" && "${RESUME_MODE}" != "fresh" ]]; then
    miq_cloud_die "an exact checkpoint was requested but the experiment run does not exist"
  fi
  echo "Starting fresh immutable ${TASK} production run."
fi

miq_cloud_run_training \
  python -m miq_grpo.train --config "${CONFIG}" "${RESUME_ARGS[@]}"
if [[ ! -f "${RUN_DIR}/training_complete.json" ]]; then
  RESUME_CHECKPOINT="$(miq_cloud_checkpoint "${RUN_DIR}" auto)"
  printf '%s\n' \
    "interrupted_at=$(miq_cloud_utc)" \
    "resume_checkpoint=${RESUME_CHECKPOINT}" \
    > "${MIQ_CLOUD_ATTEMPT_DIR}/needs_resume.txt"
  echo "Training checkpointed before completion. Rerun the same command to resume." >&2
  exit 75
fi
python -m miq_grpo.preflight checkpoint \
  --path "${FINAL_MODEL}" \
  --reference-path "${MIQ_MODEL_PATH}"
MARKER="${MIQ_PIPELINE_STATE}/gates/train_${TASK}_complete.txt"
if [[ ! -e "${MARKER}" ]]; then
  miq_cloud_write_once "${MARKER}" \
    "completed_at=$(miq_cloud_utc)" \
    "experiment=${EXPERIMENT}" \
    "final_model=${FINAL_MODEL}"
fi
echo "Production ${TASK} model ready: ${FINAL_MODEL}"
