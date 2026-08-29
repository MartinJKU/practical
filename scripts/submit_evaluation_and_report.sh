#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "Recovery finalization refused: $*" >&2
  exit 2
}

: "${MIQ_PROJECT_ROOT:?Set MIQ_PROJECT_ROOT to this project directory on Leonardo}"
: "${MIQ_OFFLINE_BUNDLE_ROOT:?Set MIQ_OFFLINE_BUNDLE_ROOT to the staged immutable bundle}"
: "${MIQ_DCGP_ACCOUNT:?Set MIQ_DCGP_ACCOUNT}"
: "${MIQ_BOOSTER_ACCOUNT:?Set MIQ_BOOSTER_ACCOUNT}"
: "${MIQ_DATA_ROOT:?Set MIQ_DATA_ROOT to the persistent frozen-dataset root}"
: "${MIQ_RUN_ROOT:?Set MIQ_RUN_ROOT to the persistent training-run root}"
: "${MIQ_EVAL_RESULTS_ROOT:?Set MIQ_EVAL_RESULTS_ROOT to a new evaluation-results path}"
: "${MIQ_REPORT_ROOT:?Set MIQ_REPORT_ROOT to a new report path}"

export MIQ_PIPELINE_STATE="${MIQ_PIPELINE_STATE:-${MIQ_RUN_ROOT}/pipeline_state}"
PROJECT_ROOT="$(cd "${MIQ_PROJECT_ROOT}" && pwd)"
PIPELINE_STATE="${MIQ_PIPELINE_STATE}"
OFFLINE_ROOT="${MIQ_OFFLINE_BUNDLE_ROOT}"
ASSET_PYTHON="${OFFLINE_ROOT}/venv/bin/python"
TRAIN_ASSETS_MANIFEST="${OFFLINE_ROOT}/training_assets/assets_manifest.json"
EVAL_ASSETS_MANIFEST="${OFFLINE_ROOT}/evaluation_assets/assets_manifest.json"
EVAL_REPO="${OFFLINE_ROOT}/vendor/moleculariq-eval"
ORIGINAL_JOBS="${PIPELINE_STATE}/submitted_jobs.txt"
BUNDLE_POINTER="${PIPELINE_STATE}/dataset_bundle_path.txt"
SUBMISSION_ROOT="${PIPELINE_STATE}/evaluation_and_report_submission"

[[ -f "${PROJECT_ROOT}/pyproject.toml" ]] || fail "MIQ_PROJECT_ROOT is not this project"
[[ -f "${PROJECT_ROOT}/configs/evaluation.yaml" ]] || fail "evaluation config is missing"
[[ -f "${PROJECT_ROOT}/scripts/slurm/official_eval_array.sbatch" ]] || fail "evaluation SLURM script is missing"
[[ -f "${PROJECT_ROOT}/scripts/slurm/report.sbatch" ]] || fail "report SLURM script is missing"
[[ -d "${PIPELINE_STATE}" ]] || fail "pipeline state directory does not exist: ${PIPELINE_STATE}"
[[ -f "${PIPELINE_STATE}/submission_started.txt" ]] || fail "original submission marker is missing"
[[ -f "${ORIGINAL_JOBS}" ]] || fail "original job-ID record is missing"
[[ -f "${BUNDLE_POINTER}" ]] || fail "frozen-dataset bundle pointer is missing"
[[ -x "${ASSET_PYTHON}" ]] || fail "staged Python environment is missing"
[[ -f "${OFFLINE_ROOT}/requirements.lock" ]] || fail "offline environment lock is missing"
[[ -f "${OFFLINE_ROOT}/wheelhouse.sha256" ]] || fail "wheelhouse hash inventory is missing"
[[ -f "${TRAIN_ASSETS_MANIFEST}" && -f "${OFFLINE_ROOT}/training_assets/_READY" ]] \
  || fail "training assets are incomplete"
[[ -f "${EVAL_ASSETS_MANIFEST}" && -f "${OFFLINE_ROOT}/evaluation_assets/_READY" ]] \
  || fail "evaluation assets are incomplete"
[[ -d "${EVAL_REPO}/.git" ]] || fail "pinned official evaluator checkout is missing"
[[ ! -e "${MIQ_EVAL_RESULTS_ROOT}" ]] \
  || fail "evaluation-results path already exists; a fresh official evaluation is required"
[[ ! -e "${MIQ_REPORT_ROOT}" ]] || fail "report path already exists"
[[ ! -e "${SUBMISSION_ROOT}" ]] \
  || fail "a recovery finalization submission is already recorded at ${SUBMISSION_ROOT}"

for command_name in git sbatch squeue; do
  command -v "${command_name}" >/dev/null 2>&1 || fail "required command is unavailable: ${command_name}"
done

PYTHONNOUSERSITE=1 "${ASSET_PYTHON}" -I -m miq_grpo.offline_bundle verify \
  --bundle-root "${OFFLINE_ROOT}" \
  --project-root "${PROJECT_ROOT}" \
  >/dev/null || fail "sealed offline bundle verification failed"

EXPECTED_EVAL_COMMIT="425ecaaa8faf65aa43aa60ec0f584b7b7f060063"
[[ "$(git -C "${EVAL_REPO}" rev-parse HEAD)" == "${EXPECTED_EVAL_COMMIT}" ]] \
  || fail "official evaluator checkout is at the wrong commit"
[[ -z "$(git -C "${EVAL_REPO}" status --porcelain)" ]] \
  || fail "official evaluator checkout has uncommitted changes"

DATA_BUNDLE="$(<"${BUNDLE_POINTER}")"
[[ -n "${DATA_BUNDLE}" && -d "${DATA_BUNDLE}" ]] || fail "frozen-dataset bundle does not exist"
[[ -f "${DATA_BUNDLE}/_READY" && -f "${DATA_BUNDLE}/bundle_manifest.json" ]] \
  || fail "frozen-dataset bundle is incomplete"

VALIDATION_OUTPUT="$(
  "${ASSET_PYTHON}" - \
    "${MIQ_DATA_ROOT}" \
    "${DATA_BUNDLE}" \
    "${MIQ_RUN_ROOT}" \
    "${TRAIN_ASSETS_MANIFEST}" \
    "${EVAL_ASSETS_MANIFEST}" <<'PY'
import json
import re
import sys
from pathlib import Path

data_root, bundle_path, run_root, training_manifest_path, evaluation_manifest_path = map(
    lambda value: Path(value).expanduser().resolve(), sys.argv[1:]
)

if data_root not in bundle_path.parents:
    raise SystemExit(f"dataset bundle is outside MIQ_DATA_ROOT: {bundle_path}")

with training_manifest_path.open(encoding="utf-8") as handle:
    training_assets = json.load(handle)
with evaluation_manifest_path.open(encoding="utf-8") as handle:
    evaluation_assets = json.load(handle)
with (bundle_path / "bundle_manifest.json").open(encoding="utf-8") as handle:
    bundle = json.load(handle)

if training_assets.get("model_id") != "Qwen/Qwen2.5-0.5B-Instruct":
    raise SystemExit("training-assets manifest names the wrong base model")
if training_assets.get("official_benchmark_staged") is not False:
    raise SystemExit("training assets violate the benchmark boundary")
model_path = Path(training_assets.get("model_path", "")).expanduser().resolve()
model_revision = training_assets.get("resolved_model_revision")
model_tree_hash = training_assets.get("model_identity", {}).get("tree_sha256")
if not model_path.is_dir() or not isinstance(model_revision, str) or not model_revision:
    raise SystemExit("training-assets manifest lacks a usable model snapshot")
if not isinstance(model_tree_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", model_tree_hash):
    raise SystemExit("training-assets manifest lacks the base-model tree hash")

if (
    evaluation_assets.get("dataset_id") != "ml-jku/moleculariq-v0.0"
    or evaluation_assets.get("evaluation_only") is not True
    or evaluation_assets.get("training_pool_present") is not False
    or evaluation_assets.get("test_rows") != 5111
):
    raise SystemExit("evaluation-assets manifest is not the complete held-out benchmark")

expected_runs = (
    ("qwen25_05b_miq_count_grpo_v1", "count"),
    ("qwen25_05b_miq_index_grpo_v1", "index"),
    ("qwen25_05b_miq_constraint_grpo_v1", "constraint"),
)
artifacts = bundle.get("artifacts", {})
for experiment_id, task_family in expected_runs:
    completion_path = run_root / experiment_id / "training_complete.json"
    if not completion_path.is_file():
        raise SystemExit(f"missing completed production run: {completion_path}")
    with completion_path.open(encoding="utf-8") as handle:
        completion = json.load(handle)
    expected_final_model = (run_root / experiment_id / "final_model").resolve()
    bundle_entry = artifacts.get(task_family, {})
    checks = {
        "experiment_id": completion.get("experiment_id") == experiment_id,
        "task_family": completion.get("task_family") == task_family,
        "non_smoke": completion.get("smoke_run") is False,
        "benchmark_not_evaluated": completion.get("official_benchmark_evaluated") is False,
        "base_model_id": completion.get("base_model_id") == training_assets.get("model_id"),
        "base_model_revision": completion.get("base_model_revision") == model_revision,
        "base_model_hash": completion.get("base_model_tree_sha256") == model_tree_hash,
        "dataset_artifact": completion.get("dataset_artifact_id") == bundle_entry.get("artifact_id"),
        "dataset_hash": completion.get("dataset_records_sha256") == bundle_entry.get("records_sha256"),
        "final_model_path": Path(completion.get("final_model", "")).expanduser().resolve()
        == expected_final_model,
        "final_model_exists": expected_final_model.is_dir(),
        "final_model_hash": bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(completion.get("final_model_identity", {}).get("tree_sha256", "")),
            )
        ),
        "positive_global_step": type(completion.get("global_step")) is int
        and completion["global_step"] > 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"invalid {experiment_id} completion manifest: {', '.join(failed)}")

print(model_path)
print(model_revision)
PY
)"
readarray -t MODEL_INFO <<<"${VALIDATION_OUTPUT}"
[[ "${#MODEL_INFO[@]}" -eq 2 ]] || fail "asset/run validation returned malformed model information"
export MIQ_MODEL_PATH="${MODEL_INFO[0]}"
export MIQ_MODEL_REVISION="${MODEL_INFO[1]}"

readarray -t OLD_DOWNSTREAM_JOBS < <(
  "${ASSET_PYTHON}" - "${ORIGINAL_JOBS}" <<'PY'
import sys
from pathlib import Path

entries = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key, separator, value = line.partition("=")
    if separator:
        entries[key] = value.split(";", 1)[0]
for key in ("eval", "report"):
    value = entries.get(key, "")
    if not value.isdigit():
        raise SystemExit(f"original job record lacks a numeric {key} job ID")
    print(value)
PY
)
[[ "${#OLD_DOWNSTREAM_JOBS[@]}" -eq 2 ]] || fail "original downstream job record is malformed"
OLD_EVAL_JOB="${OLD_DOWNSTREAM_JOBS[0]}"
OLD_REPORT_JOB="${OLD_DOWNSTREAM_JOBS[1]}"
ACTIVE_OLD_JOBS="$(
  squeue --noheader --jobs "${OLD_EVAL_JOB},${OLD_REPORT_JOB}" --format='%i'
)" || fail "could not query the original downstream SLURM jobs"
if [[ -n "${ACTIVE_OLD_JOBS//[[:space:]]/}" ]]; then
  fail "stale downstream jobs are still active (${ACTIVE_OLD_JOBS//$'\n'/, }); cancel ${OLD_EVAL_JOB} and ${OLD_REPORT_JOB} first"
fi

ORIGINAL_JOBS_SHA256="$(
  "${ASSET_PYTHON}" -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "${ORIGINAL_JOBS}"
)"

if ! mkdir "${SUBMISSION_ROOT}"; then
  fail "another finalization attempt reserved ${SUBMISSION_ROOT}"
fi
printf '%s\n' \
  "validated_at=$(date --iso-8601=seconds)" \
  "original_submitted_jobs_sha256=${ORIGINAL_JOBS_SHA256}" \
  "superseded_eval_job=${OLD_EVAL_JOB}" \
  "superseded_report_job=${OLD_REPORT_JOB}" \
  "model_revision=${MIQ_MODEL_REVISION}" \
  "dataset_bundle=${DATA_BUNDLE}" \
  "evaluation_results_root=${MIQ_EVAL_RESULTS_ROOT}" \
  "report_root=${MIQ_REPORT_ROOT}" \
  > "${SUBMISSION_ROOT}/validation.txt"

mkdir -p "${PROJECT_ROOT}/logs/slurm"
cd "${PROJECT_ROOT}"
COMMON_EXPORT="ALL,MIQ_PROJECT_ROOT,MIQ_OFFLINE_BUNDLE_ROOT,MIQ_DATA_ROOT,MIQ_RUN_ROOT,MIQ_EVAL_RESULTS_ROOT,MIQ_REPORT_ROOT,MIQ_PIPELINE_STATE,MIQ_MODEL_PATH,MIQ_MODEL_REVISION"

if ! EVAL_SUBMISSION="$(
  sbatch --parsable \
    --account "${MIQ_BOOSTER_ACCOUNT}" \
    --array=0-3 \
    --export "${COMMON_EXPORT}" \
    "${PROJECT_ROOT}/scripts/slurm/official_eval_array.sbatch"
)"; then
  printf 'failed_at=%s\n' "$(date --iso-8601=seconds)" \
    > "${SUBMISSION_ROOT}/evaluation_submission_failed.txt"
  exit 1
fi
printf '%s\n' "${EVAL_SUBMISSION}" > "${SUBMISSION_ROOT}/evaluation_submission_raw.txt"
EVAL_JOB="${EVAL_SUBMISSION%%;*}"
if [[ ! "${EVAL_JOB}" =~ ^[0-9]+$ ]]; then
  printf 'failed_at=%s\nreason=non_numeric_job_id\n' "$(date --iso-8601=seconds)" \
    > "${SUBMISSION_ROOT}/evaluation_submission_failed.txt"
  exit 1
fi
printf '%s\n' "${EVAL_JOB}" > "${SUBMISSION_ROOT}/evaluation_job_id.txt"

if ! REPORT_SUBMISSION="$(
  sbatch --parsable \
    --account "${MIQ_DCGP_ACCOUNT}" \
    --dependency "afterok:${EVAL_JOB}" \
    --export "${COMMON_EXPORT}" \
    "${PROJECT_ROOT}/scripts/slurm/report.sbatch"
)"; then
  printf 'failed_at=%s\nevaluation_job=%s\n' "$(date --iso-8601=seconds)" "${EVAL_JOB}" \
    > "${SUBMISSION_ROOT}/report_submission_failed.txt"
  exit 1
fi
printf '%s\n' "${REPORT_SUBMISSION}" > "${SUBMISSION_ROOT}/report_submission_raw.txt"
REPORT_JOB="${REPORT_SUBMISSION%%;*}"
if [[ ! "${REPORT_JOB}" =~ ^[0-9]+$ ]]; then
  printf 'failed_at=%s\nreason=non_numeric_job_id\nevaluation_job=%s\n' \
    "$(date --iso-8601=seconds)" "${EVAL_JOB}" \
    > "${SUBMISSION_ROOT}/report_submission_failed.txt"
  exit 1
fi
printf '%s\n' "${REPORT_JOB}" > "${SUBMISSION_ROOT}/report_job_id.txt"
printf 'submitted_at=%s\nevaluation=%s\nreport=%s\n' \
  "$(date --iso-8601=seconds)" "${EVAL_JOB}" "${REPORT_JOB}" \
  > "${SUBMISSION_ROOT}/submission_complete.txt"

echo "recovery evaluation=${EVAL_JOB} report=${REPORT_JOB} state=${SUBMISSION_ROOT}"
