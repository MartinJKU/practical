#!/usr/bin/env bash
set -euo pipefail

: "${MIQ_PROJECT_ROOT:?Set MIQ_PROJECT_ROOT to this new project directory on Leonardo}"
: "${MIQ_OFFLINE_BUNDLE_ROOT:?Set MIQ_OFFLINE_BUNDLE_ROOT to the staged immutable bundle}"
: "${MIQ_DCGP_ACCOUNT:?Set MIQ_DCGP_ACCOUNT (often the project account with _0 suffix)}"
: "${MIQ_BOOSTER_ACCOUNT:?Set MIQ_BOOSTER_ACCOUNT}"
: "${MIQ_DATA_ROOT:?Set MIQ_DATA_ROOT on persistent shared storage}"
: "${MIQ_RUN_ROOT:?Set MIQ_RUN_ROOT on persistent shared storage}"
: "${MIQ_EVAL_RESULTS_ROOT:?Set MIQ_EVAL_RESULTS_ROOT on persistent shared storage}"
: "${MIQ_REPORT_ROOT:?Set MIQ_REPORT_ROOT to a new report directory}"

export MIQ_PIPELINE_STATE="${MIQ_PIPELINE_STATE:-${MIQ_RUN_ROOT}/pipeline_state}"
cd "${MIQ_PROJECT_ROOT}"

for output_root in \
  "${MIQ_DATA_ROOT}" \
  "${MIQ_RUN_ROOT}" \
  "${MIQ_EVAL_RESULTS_ROOT}" \
  "${MIQ_REPORT_ROOT}"; do
  if [[ -e "${output_root}" ]]; then
    echo "Refusing an existing pipeline output path: ${output_root}" >&2
    exit 2
  fi
done

ASSET_PYTHON="${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/python"
if [[ ! -x "${ASSET_PYTHON}" ]]; then
  echo "Offline bundle lacks its Python environment: ${ASSET_PYTHON}" >&2
  exit 2
fi
PYTHONNOUSERSITE=1 "${ASSET_PYTHON}" -I -m miq_grpo.offline_bundle verify \
  --bundle-root "${MIQ_OFFLINE_BUNDLE_ROOT}" \
  --project-root "${MIQ_PROJECT_ROOT}"

export MIQ_OFFLINE_BUNDLE_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/offline_bundle_manifest.json"
export MIQ_ENV_LOCK="${MIQ_OFFLINE_BUNDLE_ROOT}/requirements.lock"
export MIQ_WHEELHOUSE_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse.sha256"
export MIQ_SOURCE_MANIFEST="${MIQ_OFFLINE_BUNDLE_ROOT}/source_manifest.json"
mkdir -p "${MIQ_PROJECT_ROOT}/logs/slurm" "${MIQ_PIPELINE_STATE}"
SUBMISSION_MARKER="${MIQ_PIPELINE_STATE}/submission_started.txt"
if [[ -e "${SUBMISSION_MARKER}" ]]; then
  echo "Refusing to submit a second pipeline into ${MIQ_PIPELINE_STATE}" >&2
  exit 2
fi
printf 'started_at=%s\n' "$(date --iso-8601=seconds)" > "${SUBMISSION_MARKER}"

record_failed_submission() {
  local exit_status=$?
  if [[ "${exit_status}" -ne 0 ]]; then
    printf 'failed_at=%s\nexit_status=%s\n' \
      "$(date --iso-8601=seconds)" "${exit_status}" \
      > "${MIQ_PIPELINE_STATE}/submission_failed.txt"
  fi
  return "${exit_status}"
}
trap record_failed_submission EXIT

submit_job() {
  local label="$1"
  shift
  local raw_submission
  local job_id
  if ! raw_submission="$(sbatch --parsable "$@")"; then
    echo "SLURM rejected the ${label} submission" >&2
    return 1
  fi
  job_id="${raw_submission%%;*}"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "SLURM returned a non-numeric ${label} job ID: ${raw_submission}" >&2
    return 2
  fi
  printf '%s\n' "${raw_submission}" > "${MIQ_PIPELINE_STATE}/${label}_submission_raw.txt"
  printf '%s=%s\n' "${label}" "${job_id}" >> "${MIQ_PIPELINE_STATE}/submission_progress.txt"
  printf '%s\n' "${job_id}"
}

readarray -t MODEL_INFO < <("${ASSET_PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["model_path"]); print(d["resolved_model_revision"])' "${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets/assets_manifest.json")
export MIQ_MODEL_PATH="${MODEL_INFO[0]}"
export MIQ_MODEL_REVISION="${MODEL_INFO[1]}"

COMMON_EXPORT="ALL,MIQ_PROJECT_ROOT,MIQ_OFFLINE_BUNDLE_ROOT,MIQ_OFFLINE_BUNDLE_MANIFEST,MIQ_ENV_LOCK,MIQ_WHEELHOUSE_MANIFEST,MIQ_SOURCE_MANIFEST,MIQ_DATA_ROOT,MIQ_RUN_ROOT,MIQ_EVAL_RESULTS_ROOT,MIQ_REPORT_ROOT,MIQ_PIPELINE_STATE,MIQ_MODEL_PATH,MIQ_MODEL_REVISION"
PREPROCESS_JOB="$(
  submit_job preprocess \
    --account "${MIQ_DCGP_ACCOUNT}" \
    --export "${COMMON_EXPORT}" \
    "${MIQ_PROJECT_ROOT}/scripts/slurm/preprocess.sbatch"
)"
SMOKE_JOB="$(
  submit_job smoke \
    --account "${MIQ_BOOSTER_ACCOUNT}" \
    --dependency "afterok:${PREPROCESS_JOB}" \
    --export "${COMMON_EXPORT}" \
    "${MIQ_PROJECT_ROOT}/scripts/slurm/gpu_smoke.sbatch"
)"
TRAIN_JOB="$(
  submit_job train \
    --account "${MIQ_BOOSTER_ACCOUNT}" \
    --dependency "afterok:${SMOKE_JOB}" \
    --export "${COMMON_EXPORT}" \
    "${MIQ_PROJECT_ROOT}/scripts/slurm/train_array.sbatch"
)"
EVAL_JOB="$(
  submit_job eval \
    --account "${MIQ_BOOSTER_ACCOUNT}" \
    --dependency "afterok:${TRAIN_JOB}" \
    --export "${COMMON_EXPORT}" \
    "${MIQ_PROJECT_ROOT}/scripts/slurm/official_eval_array.sbatch"
)"
REPORT_JOB="$(
  submit_job report \
    --account "${MIQ_DCGP_ACCOUNT}" \
    --dependency "afterok:${EVAL_JOB}" \
    --export "${COMMON_EXPORT}" \
    "${MIQ_PROJECT_ROOT}/scripts/slurm/report.sbatch"
)"

printf 'preprocess=%s\nsmoke=%s\ntrain=%s\neval=%s\nreport=%s\n' \
  "${PREPROCESS_JOB}" "${SMOKE_JOB}" "${TRAIN_JOB}" "${EVAL_JOB}" "${REPORT_JOB}" \
  > "${MIQ_PIPELINE_STATE}/submitted_jobs.txt"
trap - EXIT
echo "preprocess=${PREPROCESS_JOB} smoke=${SMOKE_JOB} train=${TRAIN_JOB} eval=${EVAL_JOB} report=${REPORT_JOB}"
