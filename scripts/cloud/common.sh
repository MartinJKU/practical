#!/usr/bin/env bash

# Shared helpers for provider-neutral, direct-process execution on one x86-64
# Linux host/pod. This file is sourced; phase scripts set their own strict mode.

MIQ_CLOUD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MIQ_CLOUD_DEFAULT_PROJECT_ROOT="$(cd "${MIQ_CLOUD_SCRIPT_DIR}/../.." && pwd -P)"

miq_cloud_die() {
  echo "ERROR: $*" >&2
  exit 2
}

miq_cloud_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

miq_cloud_abs() {
  realpath -m -- "$1"
}

miq_cloud_configure_base() {
  command -v realpath >/dev/null 2>&1 || miq_cloud_die "GNU realpath is required"
  command -v flock >/dev/null 2>&1 || miq_cloud_die "util-linux flock is required"
  command -v setsid >/dev/null 2>&1 || miq_cloud_die "util-linux setsid is required"
  command -v tee >/dev/null 2>&1 || miq_cloud_die "tee is required"

  export MIQ_PROJECT_ROOT="$(miq_cloud_abs "${MIQ_PROJECT_ROOT:-${MIQ_CLOUD_DEFAULT_PROJECT_ROOT}}")"
  [[ -f "${MIQ_PROJECT_ROOT}/pyproject.toml" ]] \
    || miq_cloud_die "MIQ_PROJECT_ROOT is not the MolecularIQ project: ${MIQ_PROJECT_ROOT}"
  [[ -d "${MIQ_PROJECT_ROOT}/src/miq_grpo" ]] \
    || miq_cloud_die "MIQ_PROJECT_ROOT lacks src/miq_grpo: ${MIQ_PROJECT_ROOT}"

  : "${MIQ_CLOUD_ROOT:?Set MIQ_CLOUD_ROOT to a persistent shared-volume directory}"
  export MIQ_CLOUD_ROOT="$(miq_cloud_abs "${MIQ_CLOUD_ROOT}")"
  [[ "${MIQ_CLOUD_ROOT}" != "/" ]] || miq_cloud_die "MIQ_CLOUD_ROOT must not be /"
  export MIQ_OFFLINE_BUNDLE_ROOT="$(miq_cloud_abs \
    "${MIQ_OFFLINE_BUNDLE_ROOT:-${MIQ_CLOUD_ROOT}/offline_bundles/offline_bundle_v1}")"
  [[ "${MIQ_OFFLINE_BUNDLE_ROOT}" != "${MIQ_CLOUD_ROOT}" ]] \
    || miq_cloud_die "the offline bundle must not equal MIQ_CLOUD_ROOT"
  export MIQ_EXECUTION_PLATFORM="${MIQ_EXECUTION_PLATFORM:-cloud_direct}"
  export MIQ_CLOUD_PROVIDER="${MIQ_CLOUD_PROVIDER:-generic_cloud}"
}

miq_cloud_configure_run() {
  miq_cloud_configure_base
  : "${MIQ_CLOUD_RUN_ID:?Set MIQ_CLOUD_RUN_ID to a new immutable run name}"
  [[ "${MIQ_CLOUD_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
    || miq_cloud_die "MIQ_CLOUD_RUN_ID must be 1-128 safe filename characters"

  export MIQ_CLOUD_SUITE_ROOT="${MIQ_CLOUD_ROOT}/experiments/${MIQ_CLOUD_RUN_ID}"
  export MIQ_DATA_ROOT="${MIQ_CLOUD_SUITE_ROOT}/data"
  export MIQ_RUN_ROOT="${MIQ_CLOUD_SUITE_ROOT}/runs"
  export MIQ_EVAL_RESULTS_ROOT="${MIQ_CLOUD_SUITE_ROOT}/evaluation"
  export MIQ_REPORT_ROOT="${MIQ_CLOUD_SUITE_ROOT}/report"
  export MIQ_PIPELINE_STATE="${MIQ_CLOUD_SUITE_ROOT}/state"
  export MIQ_CLOUD_LOG_ROOT="${MIQ_CLOUD_SUITE_ROOT}/logs"
}

miq_cloud_require_initialized() {
  [[ -f "${MIQ_PIPELINE_STATE}/run_initialized.txt" ]] \
    || miq_cloud_die "run is not initialized; run scripts/cloud/init_run.sh first"
  [[ -d "${MIQ_CLOUD_LOG_ROOT}" && -d "${MIQ_PIPELINE_STATE}/attempts" ]] \
    || miq_cloud_die "initialized run is missing log/state directories"
  grep -Fxq "run_id=${MIQ_CLOUD_RUN_ID}" "${MIQ_PIPELINE_STATE}/run_initialized.txt" \
    || miq_cloud_die "run marker does not match MIQ_CLOUD_RUN_ID"
  grep -Fxq "project_root=${MIQ_PROJECT_ROOT}" "${MIQ_PIPELINE_STATE}/run_initialized.txt" \
    || miq_cloud_die "run marker is bound to a different project root"
  grep -Fxq "offline_bundle_root=${MIQ_OFFLINE_BUNDLE_ROOT}" \
    "${MIQ_PIPELINE_STATE}/run_initialized.txt" \
    || miq_cloud_die "run marker is bound to a different offline bundle"
}

miq_cloud_load_model_info() {
  local model_info
  model_info="$(python - "${MIQ_BASE_MODEL_MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
print(manifest["model_path"])
print(manifest["resolved_model_revision"])
PY
)"
  mapfile -t MIQ_CLOUD_MODEL_INFO <<<"${model_info}"
  [[ "${#MIQ_CLOUD_MODEL_INFO[@]}" -eq 2 ]] \
    || miq_cloud_die "base-model manifest returned malformed model information"
  export MIQ_MODEL_PATH="${MIQ_CLOUD_MODEL_INFO[0]}"
  export MIQ_MODEL_REVISION="${MIQ_CLOUD_MODEL_INFO[1]}"
}

miq_cloud_activate() {
  local phase="$1"
  # shellcheck disable=SC1091
  source "${MIQ_PROJECT_ROOT}/scripts/activate_portable.sh" "${phase}"
  miq_cloud_load_model_info
}

miq_cloud_load_dataset() {
  local pointer="${MIQ_PIPELINE_STATE}/dataset_bundle_path.txt"
  [[ -f "${pointer}" ]] || miq_cloud_die "frozen dataset pointer is missing; run preprocess.sh"
  [[ "$(wc -l < "${pointer}")" -eq 1 ]] || miq_cloud_die "dataset pointer must contain one line"
  export MIQ_DATA_BUNDLE
  MIQ_DATA_BUNDLE="$(miq_cloud_abs "$(<"${pointer}")")"
  [[ "${MIQ_DATA_BUNDLE}" == "${MIQ_DATA_ROOT}/"* ]] \
    || miq_cloud_die "dataset bundle is outside this run's immutable data root"
  [[ -f "${MIQ_DATA_BUNDLE}/_READY" && -f "${MIQ_DATA_BUNDLE}/bundle_manifest.json" ]] \
    || miq_cloud_die "dataset bundle is incomplete: ${MIQ_DATA_BUNDLE}"
}

miq_cloud_lock() {
  local name="$1"
  mkdir -p "${MIQ_PIPELINE_STATE}/locks"
  exec {MIQ_CLOUD_LOCK_FD}>"${MIQ_PIPELINE_STATE}/locks/${name}.lock"
  flock -n "${MIQ_CLOUD_LOCK_FD}" \
    || miq_cloud_die "phase '${name}' is already running in another process/pod"
}

miq_cloud_create_attempt_dir() {
  local attempt_dir="$1"
  mkdir -p "$(dirname "${attempt_dir}")"
  mkdir "${attempt_dir}" \
    || miq_cloud_die "attempt identifier collision: ${attempt_dir}"
}

miq_cloud_begin_attempt() {
  local phase="$1"
  local host
  host="$(hostname 2>/dev/null || echo unknown-host)"
  host="${host//[^A-Za-z0-9_.-]/_}"
  MIQ_CLOUD_ATTEMPT_ID="$(date -u '+%Y%m%dT%H%M%SZ')-${host}-$$"
  MIQ_CLOUD_ATTEMPT_DIR="${MIQ_PIPELINE_STATE}/attempts/${phase}/${MIQ_CLOUD_ATTEMPT_ID}"
  MIQ_CLOUD_ATTEMPT_LOG="${MIQ_CLOUD_LOG_ROOT}/${phase}/${MIQ_CLOUD_ATTEMPT_ID}.log"
  export MIQ_EXECUTION_ID="${MIQ_CLOUD_ATTEMPT_ID}"
  mkdir -p "$(dirname "${MIQ_CLOUD_ATTEMPT_LOG}")"
  miq_cloud_create_attempt_dir "${MIQ_CLOUD_ATTEMPT_DIR}"
  printf '%s\n' \
    "phase=${phase}" \
    "attempt_id=${MIQ_CLOUD_ATTEMPT_ID}" \
    "started_at=$(miq_cloud_utc)" \
    "host=${host}" \
    "pid=$$" \
    "run_id=${MIQ_CLOUD_RUN_ID:-none}" \
    "project_root=${MIQ_PROJECT_ROOT}" \
    "offline_bundle_root=${MIQ_OFFLINE_BUNDLE_ROOT}" \
    "execution_platform=${MIQ_EXECUTION_PLATFORM}" \
    "cloud_provider=${MIQ_CLOUD_PROVIDER}" \
    "execution_id=${MIQ_EXECUTION_ID}" \
    > "${MIQ_CLOUD_ATTEMPT_DIR}/started.txt"
  exec > >(tee -a "${MIQ_CLOUD_ATTEMPT_LOG}") 2>&1
  echo "phase=${phase} attempt=${MIQ_CLOUD_ATTEMPT_ID} started_at=$(miq_cloud_utc)"
  trap 'miq_cloud_finish_attempt "$?"' EXIT
}

miq_cloud_finish_attempt() {
  local status="$1"
  trap - EXIT
  if [[ "${status}" -eq 0 ]]; then
    printf 'completed_at=%s\nexit_status=0\nlog=%s\n' \
      "$(miq_cloud_utc)" "${MIQ_CLOUD_ATTEMPT_LOG}" \
      > "${MIQ_CLOUD_ATTEMPT_DIR}/succeeded.txt"
    echo "phase complete at $(miq_cloud_utc)"
  else
    printf 'failed_at=%s\nexit_status=%s\nlog=%s\n' \
      "$(miq_cloud_utc)" "${status}" "${MIQ_CLOUD_ATTEMPT_LOG}" \
      > "${MIQ_CLOUD_ATTEMPT_DIR}/failed.txt"
    echo "phase failed with exit status ${status} at $(miq_cloud_utc)" >&2
  fi
  exit "${status}"
}

miq_cloud_gpu_preflight() {
  echo "host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  command -v nvidia-smi >/dev/null 2>&1 || miq_cloud_die "nvidia-smi is unavailable"
  nvidia-smi
  python -m miq_grpo.preflight environment --require-cuda
}

miq_cloud_run_training() {
  # Run training as a managed child so a Pod/container TERM reaches the Python
  # handler. The handler requests a checkpoint and clean stop; keep waiting
  # after an interrupted wait so the shell never abandons that checkpoint.
  local child_pid child_status wait_status signal_received=""
  # Isolate the child from terminal Ctrl-C. The wrapper receives INT and
  # forwards TERM, which train.py treats as a checkpoint request.
  setsid "$@" &
  child_pid=$!
  trap 'signal_received=TERM; kill -TERM "${child_pid}" 2>/dev/null || true' TERM
  trap 'signal_received=INT; kill -TERM "${child_pid}" 2>/dev/null || true' INT
  while kill -0 "${child_pid}" 2>/dev/null; do
    set +e
    wait "${child_pid}"
    wait_status=$?
    set -e
    if ! kill -0 "${child_pid}" 2>/dev/null; then
      child_status="${wait_status}"
      break
    fi
  done
  if [[ -z "${child_status:-}" ]]; then
    set +e
    wait "${child_pid}"
    child_status=$?
    set -e
  fi
  trap - TERM INT
  if [[ -n "${signal_received}" ]]; then
    echo "Forwarded ${signal_received} as TERM; training child exited with ${child_status}."
  fi
  return "${child_status}"
}

miq_cloud_checkpoint() {
  local run_dir="$1"
  local requested="${2:-auto}"
  python - "${run_dir}" "${requested}" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).expanduser().resolve()
requested = sys.argv[2]
if not run_dir.is_dir():
    raise SystemExit(f"run directory does not exist: {run_dir}")

if requested == "auto":
    numbered = []
    for path in run_dir.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir():
            numbered.append((int(match.group(1)), path.resolve()))
    if not numbered:
        raise SystemExit(f"no checkpoint-N directory exists in {run_dir}")
    step, checkpoint = max(numbered)
else:
    checkpoint = Path(requested).expanduser().resolve()
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if not match or checkpoint.parent != run_dir:
        raise SystemExit("exact resume path must be checkpoint-N directly inside this experiment run")
    step = int(match.group(1))

state_path = checkpoint / "trainer_state.json"
if not state_path.is_file():
    raise SystemExit(f"checkpoint lacks trainer_state.json: {checkpoint}")
with state_path.open(encoding="utf-8") as handle:
    state = json.load(handle)
if state.get("global_step") != step:
    raise SystemExit(
        f"checkpoint directory step {step} differs from trainer_state global_step "
        f"{state.get('global_step')}"
    )
for filename in ("optimizer.pt", "scheduler.pt"):
    if not (checkpoint / filename).is_file():
        raise SystemExit(f"checkpoint lacks resumable state file {filename}: {checkpoint}")
if not any(
    (checkpoint / filename).is_file()
    for filename in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
):
    raise SystemExit(f"checkpoint lacks full model weights: {checkpoint}")
print(checkpoint)
PY
}

miq_cloud_write_once() {
  local path="$1"
  shift
  mkdir -p "$(dirname "${path}")"
  (set -o noclobber; printf '%s\n' "$@" > "${path}") \
    || miq_cloud_die "refusing to overwrite marker: ${path}"
}
