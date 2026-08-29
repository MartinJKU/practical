"""Connected-node staging for later fully offline Leonardo jobs."""

from __future__ import annotations

import argparse
import hashlib
import os
from importlib import metadata
from pathlib import Path

from .constants import MODEL_ID, OFFICIAL_BENCHMARK_ID, TRAIN_POOL_ID
from .io_utils import directory_identity, records_sha256, utc_now, write_json_exclusive


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable staging root: {path}")


def _configure_hf_home(root: Path) -> Path:
    hf_home = root / "hf_cache"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    return hf_home


def stage_training_assets(output_root: Path, model_revision: str) -> None:
    """Download the base model and export only the official training pool."""
    _refuse_existing(output_root)
    output_root.mkdir(parents=True)
    hf_home = _configure_hf_home(output_root)

    from datasets import Dataset
    from huggingface_hub import HfApi, snapshot_download
    from moleculariq_core import load_molecule_pool

    api = HfApi()
    model_info = api.model_info(MODEL_ID, revision=model_revision)
    pool_info = api.dataset_info(TRAIN_POOL_ID)

    model_path = output_root / "models" / "Qwen2.5-0.5B-Instruct"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=model_info.sha,
        local_dir=model_path,
        repo_type="model",
    )
    model_identity = directory_identity(model_path)

    smiles_values = load_molecule_pool("train", cache_dir=str(hf_home / "datasets"))
    pool_info_after = api.dataset_info(TRAIN_POOL_ID)
    if pool_info_after.sha != pool_info.sha:
        raise RuntimeError("MolecularIQ training pool changed while assets were being staged; retry")
    if not smiles_values or any(
        not isinstance(smiles, str) or not smiles.strip() for smiles in smiles_values
    ):
        raise RuntimeError("Official MolecularIQ training pool is empty or contains an invalid SMILES field")

    rows = [
        {
            "molecule_id": hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:24],
            "smiles": smiles,
        }
        for smiles in smiles_values
    ]
    pool_root = output_root / "train_pool"
    pool_dataset_path = pool_root / "dataset"
    pool_root.mkdir(parents=True)
    Dataset.from_list(rows).save_to_disk(str(pool_dataset_path))
    pool_manifest = {
        "artifact_type": "moleculariq_training_pool_snapshot",
        "created_at": utc_now(),
        "source_dataset": TRAIN_POOL_ID,
        "source_dataset_revision": pool_info.sha,
        "pool_name": "train",
        "num_molecules": len(rows),
        "records_sha256": records_sha256(rows),
        "moleculariq_core_version": metadata.version("moleculariq-core"),
        "benchmark_data_present": False,
    }
    write_json_exclusive(pool_root / "manifest.json", pool_manifest)
    (pool_root / "_READY").touch(exist_ok=False)

    manifest = {
        "artifact_type": "moleculariq_grpo_training_assets",
        "created_at": utc_now(),
        "model_id": MODEL_ID,
        "requested_model_revision": model_revision,
        "resolved_model_revision": model_info.sha,
        "model_path": str(model_path.resolve()),
        "model_identity": model_identity,
        "training_pool_path": str(pool_dataset_path.resolve()),
        "training_pool_manifest": str((pool_root / "manifest.json").resolve()),
        "training_pool_revision": pool_info.sha,
        "hf_home": str(hf_home.resolve()),
        "official_benchmark_staged": False,
    }
    write_json_exclusive(output_root / "assets_manifest.json", manifest)
    (output_root / "_READY").touch(exist_ok=False)


def stage_evaluation_assets(output_root: Path) -> None:
    """Stage the held-out benchmark in an evaluation-only cache."""
    _refuse_existing(output_root)
    output_root.mkdir(parents=True)
    hf_home = _configure_hf_home(output_root)

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    benchmark_info = api.dataset_info(OFFICIAL_BENCHMARK_ID)
    # Cache the default ``main`` ref because the pinned official task YAML does
    # not expose a revision argument. Offline lm-eval must resolve that same ref.
    snapshot_path = Path(
        snapshot_download(
            repo_id=OFFICIAL_BENCHMARK_ID,
            revision="main",
            repo_type="dataset",
            cache_dir=str(hf_home / "hub"),
        )
    )
    if snapshot_path.name != benchmark_info.sha:
        raise RuntimeError("cached benchmark snapshot differs from the resolved main revision")
    benchmark = load_dataset(
        OFFICIAL_BENCHMARK_ID,
        split="test",
        cache_dir=str(hf_home / "datasets"),
    )
    benchmark_info_after = api.dataset_info(OFFICIAL_BENCHMARK_ID)
    if benchmark_info_after.sha != benchmark_info.sha:
        raise RuntimeError("official benchmark changed while assets were being staged; retry")
    benchmark_rows = [dict(row) for row in benchmark]
    manifest = {
        "artifact_type": "moleculariq_official_evaluation_assets",
        "created_at": utc_now(),
        "dataset_id": OFFICIAL_BENCHMARK_ID,
        "dataset_revision": benchmark_info.sha,
        "test_rows": len(benchmark),
        "test_records_sha256": records_sha256(benchmark_rows),
        "dataset_fingerprint": getattr(benchmark, "_fingerprint", None),
        "snapshot_path": str(snapshot_path),
        "hf_home": str(hf_home.resolve()),
        "training_pool_present": False,
        "evaluation_only": True,
    }
    write_json_exclusive(output_root / "assets_manifest.json", manifest)
    (output_root / "_READY").touch(exist_ok=False)


def _training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=stage_training_assets.__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-revision", default="main")
    return parser


def _evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=stage_evaluation_assets.__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def training_main() -> None:
    args = _training_parser().parse_args()
    stage_training_assets(args.output_root.expanduser().resolve(), args.model_revision)


def evaluation_main() -> None:
    args = _evaluation_parser().parse_args()
    stage_evaluation_assets(args.output_root.expanduser().resolve())
