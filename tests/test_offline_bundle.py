from __future__ import annotations

from pathlib import Path

import pytest

import miq_grpo.offline_bundle as offline_bundle
from miq_grpo.io_utils import offline_bundle_file_identities, sha256_file
from miq_grpo.offline_bundle import (
    OfflineBundleError,
    _package_source_binding,
    _verify_wheelhouse,
    write_wheelhouse_hashes,
)


def test_wheelhouse_inventory_detects_mutation_and_unlisted_files(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "package-1.0-py3-none-any.whl"
    wheel.write_bytes(b"original wheel bytes")

    manifest = write_wheelhouse_hashes(tmp_path)
    identity = _verify_wheelhouse(tmp_path)

    assert identity["num_files"] == 1
    assert identity["manifest"]["sha256"] == sha256_file(manifest)

    wheel.write_bytes(b"mutated")
    with pytest.raises(OfflineBundleError, match="hash mismatch"):
        _verify_wheelhouse(tmp_path)

    wheel.write_bytes(b"original wheel bytes")
    (wheelhouse / "unlisted.whl").write_bytes(b"extra")
    with pytest.raises(OfflineBundleError, match="manifest/file mismatch"):
        _verify_wheelhouse(tmp_path)


def test_compute_artifacts_bind_all_sealed_environment_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {
        "MIQ_OFFLINE_BUNDLE_MANIFEST": "offline_bundle_manifest.json",
        "MIQ_ENV_LOCK": "requirements.lock",
        "MIQ_SOURCE_MANIFEST": "source_manifest.json",
        "MIQ_WHEELHOUSE_MANIFEST": "wheelhouse.sha256",
    }
    for key, filename in environment.items():
        path = tmp_path / filename
        path.write_text(f"{key}\n", encoding="utf-8")
        monkeypatch.setenv(key, str(path))

    identities = offline_bundle_file_identities()

    assert set(identities) == {
        "offline_bundle_manifest",
        "requirements_lock",
        "source_manifest",
        "wheelhouse_hash_manifest",
    }
    assert all(len(entry["sha256"]) == 64 for entry in identities.values())


def test_installed_package_must_match_its_sealed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    installed = bundle / "venv" / "site-packages" / "example_package"
    source = tmp_path / "source" / "example_package"
    installed.mkdir(parents=True)
    source.mkdir(parents=True)
    (installed / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(offline_bundle, "_installed_package_root", lambda name: installed)

    binding = _package_source_binding(
        bundle,
        module_name="example_package",
        source_root=source,
    )
    assert binding["num_files"] == 1
    assert len(binding["tree_sha256"]) == 64

    (installed / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(OfflineBundleError, match="differs from its sealed source"):
        _package_source_binding(
            bundle,
            module_name="example_package",
            source_root=source,
        )
