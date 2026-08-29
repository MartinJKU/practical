"""Create and verify the immutable top-level Leonardo offline-bundle seal."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .constants import (
    MODEL_ID,
    MOLECULARIQ_CORE_COMMIT,
    MOLECULARIQ_EVAL_COMMIT,
    OFFICIAL_BENCHMARK_ID,
    TRAIN_POOL_ID,
)
from .io_utils import (
    directory_identity,
    read_json,
    sha256_file,
    source_tree_sha256,
    utc_now,
    write_json_exclusive,
)

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "offline_bundle_manifest.json"
SOURCE_MANIFEST_NAME = "source_manifest.json"
READY_MARKER_NAME = "_READY"
WHEELHOUSE_MANIFEST_NAME = "wheelhouse.sha256"
ENVIRONMENT_LOCK_NAME = "requirements.lock"

EXPECTED_VENDOR_COMMITS = {
    "moleculariq-core": MOLECULARIQ_CORE_COMMIT,
    "moleculariq-eval": MOLECULARIQ_EVAL_COMMIT,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class OfflineBundleError(RuntimeError):
    """Raised when an offline bundle is incomplete, inconsistent, or mutated."""


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OfflineBundleError(f"required manifest is missing: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise OfflineBundleError(f"manifest root must be a mapping: {path}")
    return value


def _inside(root: Path, value: str | Path, *, context: str) -> Path:
    candidate = Path(value).expanduser().resolve()
    if candidate != root and root not in candidate.parents:
        raise OfflineBundleError(f"{context} escapes the offline bundle: {candidate}")
    return candidate


def _file_identity(root: Path, path: Path) -> dict[str, Any]:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise OfflineBundleError(f"file escapes the offline bundle: {absolute}") from exc
    if not absolute.is_file():
        raise OfflineBundleError(f"required file is missing: {absolute}")
    return {
        "path": relative.as_posix(),
        "size_bytes": absolute.stat().st_size,
        "sha256": sha256_file(absolute),
    }


def _git_output(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OfflineBundleError(f"cannot inspect vendored Git repository: {repo}") from exc


def _vendor_identity(bundle_root: Path, name: str, expected_commit: str) -> dict[str, Any]:
    repo = bundle_root / "vendor" / name
    if not repo.is_dir():
        raise OfflineBundleError(f"vendored repository is missing: {repo}")
    commit = _git_output(repo, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise OfflineBundleError(f"{name} is at {commit}, expected {expected_commit}")
    status = _git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise OfflineBundleError(f"vendored repository is dirty: {name}")
    tree = _git_output(repo, "rev-parse", "HEAD^{tree}")
    return {
        "path": repo.relative_to(bundle_root).as_posix(),
        "commit": commit,
        "git_tree": tree,
        "clean": True,
    }


def _installed_package_root(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if len(locations) != 1:
        raise OfflineBundleError(f"cannot resolve one installed package root for {module_name}")
    return Path(locations[0]).resolve()


def _package_source_binding(
    bundle_root: Path,
    *,
    module_name: str,
    source_root: Path,
) -> dict[str, Any]:
    installed_root = _inside(
        bundle_root,
        _installed_package_root(module_name),
        context=f"installed {module_name} package",
    )
    source_root = source_root.resolve()
    installed_identity = directory_identity(installed_root)
    source_identity = directory_identity(source_root)
    if installed_identity != source_identity:
        raise OfflineBundleError(f"installed {module_name} package differs from its sealed source tree")
    return {
        "installed_path": installed_root.relative_to(bundle_root).as_posix(),
        "source_path": str(source_root),
        "tree_sha256": installed_identity["tree_sha256"],
        "num_files": installed_identity["num_files"],
    }


def _source_identity(bundle_root: Path, project_root: Path) -> dict[str, Any]:
    if not project_root.is_dir():
        raise OfflineBundleError(f"project source tree is missing: {project_root}")
    vendors = {
        name: _vendor_identity(bundle_root, name, expected)
        for name, expected in sorted(EXPECTED_VENDOR_COMMITS.items())
    }
    installed_packages = {
        "lm_eval": _package_source_binding(
            bundle_root,
            module_name="lm_eval",
            source_root=bundle_root / "vendor" / "moleculariq-eval" / "lm_eval",
        ),
        "miq_grpo": _package_source_binding(
            bundle_root,
            module_name="miq_grpo",
            source_root=project_root / "src" / "miq_grpo",
        ),
        "moleculariq_core": _package_source_binding(
            bundle_root,
            module_name="moleculariq_core",
            source_root=bundle_root / "vendor" / "moleculariq-core" / "src" / "moleculariq_core",
        ),
    }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": "moleculariq_offline_source_identity",
        "project_root": str(project_root),
        "project_source_tree_sha256": source_tree_sha256(project_root),
        "installed_packages": installed_packages,
        "vendors": vendors,
    }


def _asset_identity(bundle_root: Path) -> dict[str, Any]:
    training_root = bundle_root / "training_assets"
    evaluation_root = bundle_root / "evaluation_assets"
    for asset_root in (training_root, evaluation_root):
        if not (asset_root / READY_MARKER_NAME).is_file():
            raise OfflineBundleError(f"asset ready marker is missing: {asset_root}")

    training_manifest_path = training_root / "assets_manifest.json"
    evaluation_manifest_path = evaluation_root / "assets_manifest.json"
    training_manifest = _mapping(training_manifest_path)
    evaluation_manifest = _mapping(evaluation_manifest_path)

    if (
        training_manifest.get("artifact_type") != "moleculariq_grpo_training_assets"
        or training_manifest.get("model_id") != MODEL_ID
        or training_manifest.get("official_benchmark_staged") is not False
    ):
        raise OfflineBundleError("training-assets manifest violates the offline training boundary")
    model_revision = training_manifest.get("resolved_model_revision")
    if not isinstance(model_revision, str) or not _REVISION_RE.fullmatch(model_revision):
        raise OfflineBundleError("training-assets manifest lacks a resolved model revision")
    model_path = _inside(
        bundle_root,
        training_manifest.get("model_path", ""),
        context="staged model path",
    )
    actual_model_identity = directory_identity(model_path)
    if training_manifest.get("model_identity") != actual_model_identity:
        raise OfflineBundleError("staged base model differs from its asset manifest")

    pool_manifest_path = _inside(
        bundle_root,
        training_manifest.get("training_pool_manifest", ""),
        context="training-pool manifest path",
    )
    pool_root = pool_manifest_path.parent
    if not (pool_root / READY_MARKER_NAME).is_file():
        raise OfflineBundleError("training-pool ready marker is missing")
    pool_manifest = _mapping(pool_manifest_path)
    if (
        pool_manifest.get("artifact_type") != "moleculariq_training_pool_snapshot"
        or pool_manifest.get("source_dataset") != TRAIN_POOL_ID
        or pool_manifest.get("pool_name") != "train"
        or pool_manifest.get("benchmark_data_present") is not False
        or not isinstance(pool_manifest.get("records_sha256"), str)
    ):
        raise OfflineBundleError("training-pool manifest is incomplete or unsafe")
    pool_dataset = _inside(
        bundle_root,
        training_manifest.get("training_pool_path", ""),
        context="training-pool dataset path",
    )

    if (
        evaluation_manifest.get("artifact_type") != "moleculariq_official_evaluation_assets"
        or evaluation_manifest.get("dataset_id") != OFFICIAL_BENCHMARK_ID
        or evaluation_manifest.get("evaluation_only") is not True
        or evaluation_manifest.get("training_pool_present") is not False
        or int(evaluation_manifest.get("test_rows", -1)) != 5111
        or not isinstance(evaluation_manifest.get("test_records_sha256"), str)
    ):
        raise OfflineBundleError("evaluation-assets manifest is incomplete or violates the held-out boundary")
    benchmark_revision = evaluation_manifest.get("dataset_revision")
    if not isinstance(benchmark_revision, str) or not _REVISION_RE.fullmatch(benchmark_revision):
        raise OfflineBundleError("evaluation-assets manifest lacks a resolved dataset revision")
    benchmark_snapshot = _inside(
        bundle_root,
        evaluation_manifest.get("snapshot_path", ""),
        context="benchmark snapshot path",
    )

    return {
        "training": {
            "assets_manifest": _file_identity(bundle_root, training_manifest_path),
            "pool_manifest": _file_identity(bundle_root, pool_manifest_path),
            "model_revision": model_revision,
            "model_identity": actual_model_identity,
            "training_pool_dataset_identity": directory_identity(pool_dataset),
            "training_pool_records_sha256": pool_manifest["records_sha256"],
        },
        "evaluation": {
            "assets_manifest": _file_identity(bundle_root, evaluation_manifest_path),
            "dataset_revision": benchmark_revision,
            "benchmark_snapshot_identity": directory_identity(benchmark_snapshot),
            "test_rows": 5111,
            "test_records_sha256": evaluation_manifest["test_records_sha256"],
        },
    }


def write_wheelhouse_hashes(bundle_root: Path) -> Path:
    """Write stable GNU-compatible SHA-256 lines for every staged wheel."""
    wheelhouse = bundle_root / "wheelhouse"
    if not wheelhouse.is_dir():
        raise OfflineBundleError(f"wheelhouse is missing: {wheelhouse}")
    files = sorted(path for path in wheelhouse.rglob("*") if path.is_file())
    if not files:
        raise OfflineBundleError("wheelhouse contains no files")
    output = bundle_root / WHEELHOUSE_MANIFEST_NAME
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for path in files:
            relative = path.relative_to(bundle_root).as_posix()
            handle.write(f"{sha256_file(path)}  {relative}\n")
    return output


def _verify_wheelhouse(bundle_root: Path) -> dict[str, Any]:
    manifest_path = bundle_root / WHEELHOUSE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise OfflineBundleError("wheelhouse hash manifest is missing")
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]):
            raise OfflineBundleError(f"invalid wheelhouse hash line {line_number}")
        relative = parts[1].lstrip("*")
        if relative in expected:
            raise OfflineBundleError(f"duplicate wheelhouse entry: {relative}")
        path = _inside(bundle_root, bundle_root / relative, context="wheelhouse entry")
        if path.parent != bundle_root / "wheelhouse" and bundle_root / "wheelhouse" not in path.parents:
            raise OfflineBundleError(f"wheelhouse entry is outside wheelhouse/: {relative}")
        expected[relative] = parts[0]
    actual_paths = sorted(path for path in (bundle_root / "wheelhouse").rglob("*") if path.is_file())
    actual_names = {path.relative_to(bundle_root).as_posix() for path in actual_paths}
    if set(expected) != actual_names:
        missing = sorted(actual_names - set(expected))
        extra = sorted(set(expected) - actual_names)
        raise OfflineBundleError(f"wheelhouse manifest/file mismatch: unlisted={missing}, missing={extra}")
    total_size = 0
    for path in actual_paths:
        relative = path.relative_to(bundle_root).as_posix()
        if sha256_file(path) != expected[relative]:
            raise OfflineBundleError(f"wheelhouse file hash mismatch: {relative}")
        total_size += path.stat().st_size
    if not actual_paths:
        raise OfflineBundleError("wheelhouse contains no files")
    return {
        "manifest": _file_identity(bundle_root, manifest_path),
        "num_files": len(actual_paths),
        "total_size_bytes": total_size,
    }


def _installed_freeze() -> str:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OfflineBundleError("cannot inspect the staged Python environment") from exc
    return "\n".join(completed.stdout.splitlines()) + "\n"


def _verify_environment_lock(bundle_root: Path) -> dict[str, Any]:
    lock_path = bundle_root / ENVIRONMENT_LOCK_NAME
    if not lock_path.is_file() or lock_path.stat().st_size == 0:
        raise OfflineBundleError("requirements.lock is missing or empty")
    if lock_path.read_text(encoding="utf-8") != _installed_freeze():
        raise OfflineBundleError("staged Python environment differs from requirements.lock")
    return _file_identity(bundle_root, lock_path)


def _required_file_identities(bundle_root: Path, source_manifest_path: Path) -> dict[str, Any]:
    required = {
        "environment_lock": bundle_root / ENVIRONMENT_LOCK_NAME,
        "python_executable": bundle_root / "venv" / "bin" / "python",
        "python_venv_config": bundle_root / "venv" / "pyvenv.cfg",
        "source_manifest": source_manifest_path,
        "wheelhouse_manifest": bundle_root / WHEELHOUSE_MANIFEST_NAME,
    }
    return {name: _file_identity(bundle_root, path) for name, path in sorted(required.items())}


def create_bundle_manifest(bundle_root: Path, project_root: Path) -> Path:
    """Seal a fully staged bundle. A failed build never receives the top-level ready marker."""
    bundle_root = bundle_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    manifest_path = bundle_root / BUNDLE_MANIFEST_NAME
    ready_path = bundle_root / READY_MARKER_NAME
    source_manifest_path = bundle_root / SOURCE_MANIFEST_NAME
    if manifest_path.exists() or ready_path.exists() or source_manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing offline-bundle seal")

    environment_lock = _verify_environment_lock(bundle_root)
    wheelhouse = _verify_wheelhouse(bundle_root)
    source = {**_source_identity(bundle_root, project_root), "created_at": utc_now()}
    write_json_exclusive(source_manifest_path, source)
    assets = _asset_identity(bundle_root)
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": "moleculariq_leonardo_offline_bundle",
        "created_at": utc_now(),
        "bundle_root": str(bundle_root),
        "project_root": str(project_root),
        "environment_lock": environment_lock,
        "wheelhouse": wheelhouse,
        "source": source,
        "assets": assets,
        "required_files": _required_file_identities(bundle_root, source_manifest_path),
    }
    write_json_exclusive(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    with ready_path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(manifest_hash + "\n")
    source_manifest_path.chmod(0o444)
    manifest_path.chmod(0o444)
    ready_path.chmod(0o444)
    verify_bundle(bundle_root, project_root)
    return manifest_path


def verify_bundle(bundle_root: Path, project_root: Path) -> dict[str, Any]:
    """Fail closed unless every sealed input still matches the published bundle manifest."""
    bundle_root = bundle_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    manifest_path = bundle_root / BUNDLE_MANIFEST_NAME
    ready_path = bundle_root / READY_MARKER_NAME
    if not manifest_path.is_file() or not ready_path.is_file():
        raise OfflineBundleError("offline bundle is partial: top-level manifest/_READY missing")
    ready_hash = ready_path.read_text(encoding="ascii").strip()
    if not _SHA256_RE.fullmatch(ready_hash) or ready_hash != sha256_file(manifest_path):
        raise OfflineBundleError("offline-bundle ready marker does not match its manifest")
    manifest = _mapping(manifest_path)
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("artifact_type") != "moleculariq_leonardo_offline_bundle"
        or manifest.get("bundle_root") != str(bundle_root)
        or manifest.get("project_root") != str(project_root)
    ):
        raise OfflineBundleError("offline-bundle manifest identity/path mismatch")

    environment_lock = _verify_environment_lock(bundle_root)
    wheelhouse = _verify_wheelhouse(bundle_root)
    source_manifest_path = bundle_root / SOURCE_MANIFEST_NAME
    source_manifest = _mapping(source_manifest_path)
    current_source = _source_identity(bundle_root, project_root)
    recorded_source = {key: value for key, value in source_manifest.items() if key != "created_at"}
    if recorded_source != current_source or manifest.get("source") != source_manifest:
        raise OfflineBundleError("project source or vendored repositories differ from the sealed identity")
    assets = _asset_identity(bundle_root)
    required_files = _required_file_identities(bundle_root, source_manifest_path)
    if manifest.get("environment_lock") != environment_lock:
        raise OfflineBundleError("environment-lock identity differs from the bundle manifest")
    if manifest.get("wheelhouse") != wheelhouse:
        raise OfflineBundleError("wheelhouse identity differs from the bundle manifest")
    if manifest.get("assets") != assets:
        raise OfflineBundleError("staged asset identity differs from the bundle manifest")
    if manifest.get("required_files") != required_files:
        raise OfflineBundleError("one or more required bundle files were mutated")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hash_parser = subparsers.add_parser("hash-wheelhouse")
    hash_parser.add_argument("--bundle-root", type=Path, required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--bundle-root", type=Path, required=True)
    create_parser.add_argument("--project-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle-root", type=Path, required=True)
    verify_parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "hash-wheelhouse":
        result: Any = write_wheelhouse_hashes(args.bundle_root.expanduser().resolve())
    elif args.command == "create":
        result = create_bundle_manifest(args.bundle_root, args.project_root)
    else:
        result = verify_bundle(args.bundle_root, args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
