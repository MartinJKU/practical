from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_shell_and_slurm_scripts_parse() -> None:
    scripts = sorted((PROJECT_ROOT / "scripts").glob("*.sh")) + sorted(
        (PROJECT_ROOT / "scripts" / "slurm").glob("*.sbatch")
    )
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_training_activation_does_not_export_evaluation_manifest() -> None:
    text = (PROJECT_ROOT / "scripts" / "activate_leonardo.sh").read_text(encoding="utf-8")
    training_branch = text.split('elif [[ "${PHASE}" == "evaluation" ]]', 1)[0]
    assert "unset MIQ_EVAL_ASSETS_MANIFEST MIQ_EVAL_REPO" in training_branch
