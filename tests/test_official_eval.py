from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from miq_grpo.io_utils import records_sha256
from miq_grpo.official_eval import (
    _assert_no_limit,
    _git_checkout_identity,
    _verify_sample_documents,
)


def test_official_eval_rejects_nested_sample_limits() -> None:
    _assert_no_limit({"limit": None, "nested": {"samples": []}})
    with pytest.raises(ValueError, match="forbids"):
        _assert_no_limit({"nested": [{"limit": 10}]})


def test_evaluated_documents_must_match_staged_snapshot(tmp_path: Path) -> None:
    documents = [{"question": "q0"}, {"question": "q1"}]
    samples = tmp_path / "samples.jsonl"
    with samples.open("w", encoding="utf-8") as handle:
        for doc_id, doc in enumerate(documents):
            handle.write(json.dumps({"doc_id": doc_id, "doc": doc}) + "\n")
    assets = {
        "test_rows": 2,
        "test_records_sha256": records_sha256(documents),
    }

    assert _verify_sample_documents([samples], assets)["num_documents"] == 2

    with samples.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"doc_id": 0, "doc": documents[0]}) + "\n")
    with pytest.raises(RuntimeError, match="duplicate"):
        _verify_sample_documents([samples], assets)


def test_official_evaluator_checkout_must_be_clean(tmp_path: Path) -> None:
    repo = tmp_path / "evaluator"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    tracked = repo / "task.yaml"
    tracked.write_text("task: moleculariq\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "task.yaml"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )

    identity = _git_checkout_identity(repo)
    assert identity["clean"] is True
    assert len(identity["source_tree_sha256"]) == 64

    tracked.write_text("task: changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        _git_checkout_identity(repo)
