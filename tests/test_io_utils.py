from __future__ import annotations

from pathlib import Path

from miq_grpo import io_utils


def test_runtime_provenance_whitelists_cloud_metadata_without_secrets(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "tracked.txt").write_text("source", encoding="utf-8")
    monkeypatch.setenv("MIQ_CLOUD_PROVIDER", "runpod")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-for-test")
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-be-recorded")
    monkeypatch.setattr(
        io_utils,
        "_accelerator_provenance",
        lambda: {"cuda_visible_devices": "0", "nvidia_smi": {}, "torch_cuda": {}},
    )

    result = io_utils.runtime_provenance(tmp_path)

    assert result["execution_environment"]["MIQ_CLOUD_PROVIDER"] == "runpod"
    assert result["execution_environment"]["RUNPOD_POD_ID"] == "pod-for-test"
    assert "RUNPOD_API_KEY" not in result["execution_environment"]
    assert "must-not-be-recorded" not in str(result)
    assert result["accelerator"]["cuda_visible_devices"] == "0"


def test_accelerator_provenance_handles_missing_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(io_utils.shutil, "which", lambda command: None)

    result = io_utils._accelerator_provenance()

    assert result["nvidia_smi"] == {"available": False, "gpus": []}
    assert isinstance(result["torch_cuda"], dict)
