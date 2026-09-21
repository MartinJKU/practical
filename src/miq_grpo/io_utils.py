"""Small deterministic artifact and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_sha256(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def directory_identity(root: str | Path) -> dict[str, Any]:
    """Hash every material file in a model directory in stable path order."""
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ignored_parts = {".cache", "__pycache__"}
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not ignored_parts.intersection(path.relative_to(directory).parts)
    )
    if not files:
        raise RuntimeError(f"model directory has no material files: {directory}")
    digest = hashlib.sha256()
    components: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(directory).as_posix()
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        components.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": file_hash})
    return {"tree_sha256": digest.hexdigest(), "num_files": len(files), "files": components}


def write_json_exclusive(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def offline_bundle_file_identities() -> dict[str, dict[str, Any]]:
    """Bind a compute artifact to the sealed offline code/environment bundle."""
    files = {
        "offline_bundle_manifest": "MIQ_OFFLINE_BUNDLE_MANIFEST",
        "requirements_lock": "MIQ_ENV_LOCK",
        "source_manifest": "MIQ_SOURCE_MANIFEST",
        "wheelhouse_hash_manifest": "MIQ_WHEELHOUSE_MANIFEST",
    }
    identities: dict[str, dict[str, Any]] = {}
    for label, environment_key in files.items():
        value = os.environ.get(environment_key)
        if not value:
            raise RuntimeError(f"{environment_key} is required for offline-bundle provenance")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{environment_key} does not point to a file: {path}")
        identities[label] = {"path": str(path), "sha256": sha256_file(path)}
    return identities


def source_tree_sha256(root: str | Path) -> str:
    project_root = Path(root).resolve()
    digest = hashlib.sha256()
    excluded = {
        ".git",
        ".venv",
        ".pytest_cache",
        "__pycache__",
        "build",
        "data",
        "dist",
        "logs",
        "report",
        "results",
        "runs",
    }
    for path in sorted(project_root.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.relative_to(project_root).parts):
            continue
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_state(root: str | Path) -> dict[str, Any]:
    project_root = Path(root).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip()), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def _accelerator_provenance() -> dict[str, Any]:
    """Capture report-safe GPU/runtime facts without recording credentials."""
    nvidia_smi: dict[str, Any]
    executable = shutil.which("nvidia-smi")
    if executable is None:
        nvidia_smi = {"available": False, "gpus": []}
    else:
        command = [
            executable,
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            rows = []
            if completed.returncode == 0:
                for line in completed.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",", maxsplit=3)]
                    if len(parts) == 4:
                        rows.append(
                            {
                                "index": parts[0],
                                "name": parts[1],
                                "driver_version": parts[2],
                                "memory_total_mib": parts[3],
                            }
                        )
            nvidia_smi = {
                "available": completed.returncode == 0,
                "returncode": completed.returncode,
                "gpus": rows,
                "stderr": completed.stderr.strip() if completed.returncode else "",
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            nvidia_smi = {"available": False, "gpus": [], "error": type(error).__name__}

    torch_cuda: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "compute_capability": [properties.major, properties.minor],
                        "total_memory_bytes": properties.total_memory,
                    }
                )
        torch_cuda = {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": len(devices),
            "devices": devices,
        }
    except (ImportError, OSError, RuntimeError) as error:
        torch_cuda = {"available": False, "devices": [], "error": type(error).__name__}

    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": nvidia_smi,
        "torch_cuda": torch_cuda,
    }


def runtime_provenance(project_root: str | Path) -> dict[str, Any]:
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_JOB_ACCOUNT",
    )
    # Deliberately whitelist non-secret execution metadata. Never collect the
    # full environment: cloud API keys and Hugging Face tokens may be present.
    execution_keys = (
        "MIQ_EXECUTION_PLATFORM",
        "MIQ_CLOUD_PROVIDER",
        "MIQ_CLOUD_REGION",
        "MIQ_CONTAINER_IMAGE",
        "MIQ_CONTAINER_IMAGE_DIGEST",
        "MIQ_EXECUTION_ID",
        "RUNPOD_POD_ID",
        "RUNPOD_DC_ID",
        "RUNPOD_GPU_COUNT",
    )
    return {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": package_versions(
            (
                "torch",
                "transformers",
                "trl",
                "accelerate",
                "datasets",
                "rdkit",
                "moleculariq-core",
                "lm_eval",
            )
        ),
        "git": git_state(project_root),
        "source_tree_sha256": source_tree_sha256(project_root),
        "slurm": {key: os.environ.get(key) for key in slurm_keys},
        "execution_environment": {key: os.environ.get(key) for key in execution_keys},
        "accelerator": _accelerator_provenance(),
    }
