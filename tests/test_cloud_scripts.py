from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLOUD_ROOT = PROJECT_ROOT / "scripts" / "cloud"
ACTIVATE = PROJECT_ROOT / "scripts" / "activate_portable.sh"

EXPECTED_CLOUD_SCRIPTS = {
    "common.sh",
    "evaluate.sh",
    "gpu_smoke.sh",
    "init_run.sh",
    "preprocess.sh",
    "report.sh",
    "stage_bundle.sh",
    "status.sh",
    "train.sh",
}


def _text(name: str) -> str:
    return (CLOUD_ROOT / name).read_text(encoding="utf-8")


def _without_comment_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _run_selector(script: Path, value: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), value],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "MIQ_PROJECT_ROOT": str(PROJECT_ROOT),
            "MIQ_CLOUD_ROOT": "/definitely-not-a-real-cloud-root",
            "MIQ_CLOUD_RUN_ID": "selector-test",
            "MIQ_OFFLINE_BUNDLE_ROOT": "/definitely-not-a-real-offline-bundle",
        },
    )


def test_cloud_script_inventory_and_shell_syntax() -> None:
    assert CLOUD_ROOT.is_dir()
    assert {path.name for path in CLOUD_ROOT.glob("*.sh")} == EXPECTED_CLOUD_SCRIPTS
    scripts = sorted(CLOUD_ROOT.glob("*.sh")) + [ACTIVATE]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_cloud_layer_has_no_slurm_command_or_module_requirement() -> None:
    scripts = sorted(CLOUD_ROOT.glob("*.sh")) + [ACTIVATE]
    scheduler_command = re.compile(
        r"(?<![A-Za-z0-9_])(sbatch|srun|squeue|scancel|sacct)(?![A-Za-z0-9_])"
    )
    required_slurm_variable = re.compile(r"\$\{SLURM_[A-Z0-9_]+:\?")
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        executable_text = _without_comment_lines(text)
        assert "#SBATCH" not in text
        assert scheduler_command.search(executable_text) is None, script
        assert re.search(r"(?:^|[;&|\s])module\s+(?:load|purge)", executable_text) is None
        assert required_slurm_variable.search(executable_text) is None, script


def test_portable_activation_rejects_unknown_phase_before_bundle_access() -> None:
    completed = subprocess.run(
        ["bash", "-c", 'source "$1" not-a-phase', "bash", str(ACTIVATE)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "MIQ_PROJECT_ROOT": str(PROJECT_ROOT),
            "MIQ_OFFLINE_BUNDLE_ROOT": "/definitely-not-a-real-offline-bundle",
        },
    )
    assert completed.returncode != 0
    assert "phase" in completed.stderr.lower()


def test_task_and_model_selectors_fail_closed() -> None:
    train = _run_selector(CLOUD_ROOT / "train.sh", "not-a-task")
    evaluate = _run_selector(CLOUD_ROOT / "evaluate.sh", "not-a-model")
    assert train.returncode != 0
    assert evaluate.returncode != 0
    assert "count" in train.stderr and "index" in train.stderr and "constraint" in train.stderr
    assert "baseline" in evaluate.stderr
    assert "count_grpo" in evaluate.stderr
    assert "index_grpo" in evaluate.stderr
    assert "constraint_grpo" in evaluate.stderr


def test_run_and_bundle_roots_are_reserved_once() -> None:
    init_text = _text("init_run.sh")
    stage_text = _text("stage_bundle.sh")
    common_text = _text("common.sh")

    assert "MIQ_CLOUD_RUN_ID" in init_text
    assert 'MIQ_CLOUD_SUITE_ROOT="${MIQ_CLOUD_ROOT}/experiments/${MIQ_CLOUD_RUN_ID}"' in common_text
    assert re.search(r"mkdir\s+\"?\$\{?[^}\" ]*(SUITE|EXPERIMENT)[^}\" ]*\}?\"?", init_text)
    assert "mkdir -p \"${MIQ_SUITE_ROOT}" not in init_text
    assert re.search(r"(already exists|refus|fresh|new)", init_text, re.IGNORECASE)

    assert "MIQ_OFFLINE_BUNDLE_ROOT" in stage_text
    assert re.search(r"(already exists|overwrite|refus|fresh|new)", stage_text, re.IGNORECASE)
    assert "stage_offline_bundle.sh" in stage_text
    fresh_bundle_guard = 'if [[ -e "${MIQ_OFFLINE_BUNDLE_ROOT}"'
    assert fresh_bundle_guard in stage_text
    assert stage_text.index(fresh_bundle_guard) < stage_text.index("stage_offline_bundle.sh")
    assert '-e "${STAGE_STATE}"' in stage_text
    assert '-e "${STAGE_LOGS}"' in stage_text


def test_parallel_phases_use_locks_scoped_below_the_suite() -> None:
    common = _text("common.sh")
    train = _text("train.sh")
    evaluate = _text("evaluate.sh")

    assert "flock" in common
    assert "MIQ_CLOUD_RUN_ID" in common
    assert "MIQ_PIPELINE_STATE" in common
    assert "MIQ_CLOUD_ATTEMPT_ID" in common
    assert "-$$" in common
    assert "miq_cloud_create_attempt_dir" in common
    assert re.search(r"lock[^\n]*TASK|TASK[^\n]*lock", train, re.IGNORECASE)
    assert re.search(r"lock[^\n]*MODEL|MODEL[^\n]*lock", evaluate, re.IGNORECASE)
    assert "MIQ_EVAL_RESULTS_ROOT" in evaluate


def test_attempt_creation_makes_the_phase_parent_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
source "$1"
miq_cloud_create_attempt_dir "$2/state/attempts/stage-bundle/attempt-1"
""",
            "bash",
            str(CLOUD_ROOT / "common.sh"),
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    attempts = list((tmp_path / "state" / "attempts" / "stage-bundle").iterdir())
    assert len(attempts) == 1
    assert attempts[0].is_dir()


def test_production_training_forwards_termination_to_checkpoint_handler() -> None:
    common = _text("common.sh")
    train = _text("train.sh")

    assert "miq_cloud_run_training()" in common
    assert 'setsid "$@" &' in common
    assert 'kill -TERM "${child_pid}"' in common
    assert "miq_cloud_run_training" in train
    completion_check = 'if [[ ! -f "${RUN_DIR}/training_complete.json" ]]'
    assert train.index("miq_cloud_run_training") < train.index(completion_check)


def test_official_evaluation_wrapper_exposes_exactly_four_full_benchmark_models() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "evaluation.yaml").read_text(encoding="utf-8")
    )
    assert set(config["models"]) == {
        "baseline",
        "count_grpo",
        "index_grpo",
        "constraint_grpo",
    }
    assert config["official_evaluator"]["task"] == "moleculariq_pass_at_k"
    assert config["official_evaluator"]["expected_repeats"] == 3
    assert config["official_evaluator"]["full_benchmark"] is True

    evaluate = _text("evaluate.sh")
    for model_key in config["models"]:
        assert model_key in evaluate
    assert '--model-key "${MODEL_KEY}"' in evaluate
    assert re.search(r"--(limit|samples|subset)(=|\s)", evaluate) is None


def test_smoke_forces_checkpoint_then_resumes_the_same_task_run() -> None:
    smoke = _text("gpu_smoke.sh")
    assert all(task in smoke for task in ("count", "index", "constraint"))
    assert "--smoke-max-steps 2" in smoke
    assert "--smoke-stop-after-step 1" in smoke
    assert "checkpoint-1" in smoke
    assert "--resume-from-checkpoint" in smoke
    assert "smoke-metrics" in smoke
    assert 'smoke_tag="cloud-smoke-${MIQ_CLOUD_RUN_ID}-${task}"' in smoke
    stable_tag = 'SLURM_JOB_ID="cloud-smoke-${MIQ_CLOUD_RUN_ID}-${task}"'
    assert smoke.count(stable_tag) == 2
    assert smoke.index("--smoke-stop-after-step 1") < smoke.index("--resume-from-checkpoint")


def test_training_processes_do_not_receive_official_benchmark_assets() -> None:
    activation = ACTIVATE.read_text(encoding="utf-8")
    training_branch = activation.split('elif [[ "${PHASE}" == "evaluation" ]]', 1)[0]
    assert "unset MIQ_EVAL_ASSETS_MANIFEST MIQ_EVAL_REPO" in training_branch

    for name in ("preprocess.sh", "train.sh"):
        text = _text(name)
        assert 'activate_portable.sh" training' in text
        assert 'activate_portable.sh" evaluation' not in text
        assert "MIQ_EVAL_ASSETS_MANIFEST" not in text
        assert "MIQ_EVAL_REPO" not in text
        assert "moleculariq-v0.0" not in text

    smoke = _text("gpu_smoke.sh")
    smoke_training = smoke.split('activate_portable.sh" evaluation', 1)[0]
    assert 'activate_portable.sh" training' in smoke_training
    assert "MIQ_EVAL_ASSETS_MANIFEST" not in smoke_training
    assert "MIQ_EVAL_REPO" not in smoke_training
    assert "moleculariq-v0.0" not in smoke_training


def test_every_execution_uses_the_fresh_sealed_bundle_contract() -> None:
    activation = ACTIVATE.read_text(encoding="utf-8")
    assert "miq_grpo.offline_bundle verify" in activation
    assert '--bundle-root "${MIQ_OFFLINE_BUNDLE_ROOT}"' in activation
    assert '--project-root "${MIQ_PROJECT_ROOT}"' in activation

    for name in ("preprocess.sh", "gpu_smoke.sh", "train.sh", "evaluate.sh", "report.sh"):
        text = _text(name)
        assert "activate_portable.sh" in text, name
