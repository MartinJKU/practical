#!/usr/bin/env bash
set -euo pipefail

# Run once on a connected Leonardo login node. Downloads are login-node work;
# all chemistry preprocessing, training, and evaluation remain SLURM jobs.
PROJECT_ROOT="${MIQ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MIQ_PROJECT_ROOT="${PROJECT_ROOT}"
: "${MIQ_OFFLINE_BUNDLE_ROOT:?Set MIQ_OFFLINE_BUNDLE_ROOT to a new persistent path under WORK or SCRATCH}"

if [[ -n "${MIQ_BOOTSTRAP_PYTHON:-}" ]]; then
  BOOTSTRAP_PYTHON="${MIQ_BOOTSTRAP_PYTHON}"
elif command -v python3.11 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3.11)"
else
  BOOTSTRAP_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${BOOTSTRAP_PYTHON}" || ! -x "${BOOTSTRAP_PYTHON}" ]]; then
  echo "A Python 3.11 interpreter is required for connected-node staging." >&2
  exit 2
fi
BOOTSTRAP_VERSION="$(
  "${BOOTSTRAP_PYTHON}" -c \
    'import sys; print(".".join(map(str, sys.version_info[:3])))'
)"
if [[ ! "${BOOTSTRAP_VERSION}" =~ ^3\.11\. ]]; then
  cat >&2 <<EOF
Python 3.11 is required, but ${BOOTSTRAP_PYTHON} is Python ${BOOTSTRAP_VERSION}.
On Leonardo, load the deep-learning profile and an available Python 3.11 module first, then run:
  export MIQ_BOOTSTRAP_PYTHON="\$(command -v python)"
Use 'module spider python/3.11' to inspect the currently available module name.
EOF
  exit 2
fi
if ! "${BOOTSTRAP_PYTHON}" -c 'import venv' >/dev/null 2>&1; then
  echo "${BOOTSTRAP_PYTHON} does not provide the Python venv module." >&2
  exit 2
fi
echo "Using bootstrap Python ${BOOTSTRAP_VERSION}: ${BOOTSTRAP_PYTHON}"

if [[ -e "${MIQ_OFFLINE_BUNDLE_ROOT}" ]]; then
  echo "Refusing to overwrite ${MIQ_OFFLINE_BUNDLE_ROOT}" >&2
  exit 2
fi

mkdir -p "${MIQ_OFFLINE_BUNDLE_ROOT}/vendor" "${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse"
git clone https://github.com/ml-jku/moleculariq-core.git "${MIQ_OFFLINE_BUNDLE_ROOT}/vendor/moleculariq-core"
git -C "${MIQ_OFFLINE_BUNDLE_ROOT}/vendor/moleculariq-core" checkout a1b89635371c3cd942e44ebeec63ec3665e7743d
git clone https://github.com/ml-jku/moleculariq-eval.git "${MIQ_OFFLINE_BUNDLE_ROOT}/vendor/moleculariq-eval"
git -C "${MIQ_OFFLINE_BUNDLE_ROOT}/vendor/moleculariq-eval" checkout 425ecaaa8faf65aa43aa60ec0f584b7b7f060063

"${BOOTSTRAP_PYTHON}" -m venv "${MIQ_OFFLINE_BUNDLE_ROOT}/venv"
PYTHON_BIN="${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/python"
"${PYTHON_BIN}" -m pip wheel \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --wheel-dir "${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse" \
  -r "${PROJECT_ROOT}/requirements/leonardo-pypi.in"
"${PYTHON_BIN}" -m pip wheel \
  --wheel-dir "${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse" \
  --no-deps "${MIQ_OFFLINE_BUNDLE_ROOT}/vendor/moleculariq-core"
"${PYTHON_BIN}" -m pip wheel \
  --wheel-dir "${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse" \
  --find-links "${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --constraint "${PROJECT_ROOT}/requirements/leonardo-pypi.in" \
  "${MIQ_OFFLINE_BUNDLE_ROOT}/vendor/moleculariq-eval[hf]"
"${PYTHON_BIN}" -m pip wheel \
  --wheel-dir "${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse" \
  --no-deps "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pip install --no-index \
  --find-links "${MIQ_OFFLINE_BUNDLE_ROOT}/wheelhouse" \
  -r "${PROJECT_ROOT}/requirements/leonardo-pypi.in" \
  moleculariq-core "lm-eval[hf]" moleculariq-single-task-grpo
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -m pip freeze --all > "${MIQ_OFFLINE_BUNDLE_ROOT}/requirements.lock"
"${PYTHON_BIN}" -I -m miq_grpo.offline_bundle hash-wheelhouse \
  --bundle-root "${MIQ_OFFLINE_BUNDLE_ROOT}"

"${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/miq-stage-training-assets" \
  --output-root "${MIQ_OFFLINE_BUNDLE_ROOT}/training_assets" \
  --model-revision main
"${MIQ_OFFLINE_BUNDLE_ROOT}/venv/bin/miq-stage-evaluation-assets" \
  --output-root "${MIQ_OFFLINE_BUNDLE_ROOT}/evaluation_assets"

"${PYTHON_BIN}" -I -m miq_grpo.offline_bundle create \
  --bundle-root "${MIQ_OFFLINE_BUNDLE_ROOT}" \
  --project-root "${PROJECT_ROOT}"

echo "Offline bundle ready: ${MIQ_OFFLINE_BUNDLE_ROOT}"
