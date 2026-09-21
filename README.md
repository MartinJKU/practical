# MolecularIQ single-task GRPO study

This is a new, self-contained implementation for training three full-parameter
`Qwen/Qwen2.5-0.5B-Instruct` models with TRL GRPO:

| Run | Frozen training data | Reportable evaluation |
|---|---|---|
| `count_grpo` | MolecularIQ count questions only | Whole official MolecularIQ benchmark |
| `index_grpo` | MolecularIQ index questions only | Whole official MolecularIQ benchmark |
| `constraint_grpo` | MolecularIQ constrained-generation questions only | Whole official MolecularIQ benchmark |
| `baseline` | No training | Whole official MolecularIQ benchmark |

No pre-existing project directory, dataset, checkpoint, or result is consumed.
All output paths are immutable-by-default and refuse silent overwrites.

## Study contract

- Training molecules come only from `ml-jku/moleculariq-trainPool`'s `train`
  pool.
- Questions, targets, constraints, and metadata are generated offline in a CPU
  preprocessing job, then materialized as three frozen Hugging Face Datasets.
- The GRPO loop receives those datasets and only scores new completions. It
  never generates a new question.
- Count and index rewards use stored targets and `moleculariq-core`'s official
  evaluator. Constraint rewards validate any generated SMILES against the
  stored constraints; they do not compare it to one reference SMILES.
- The held-out benchmark is in a separate evaluation cache and is never exposed
  through the training environment.
- Every reportable model is evaluated once on all 5,111 official test questions
  with the pinned `moleculariq_pass_at_k` task and its three repeats. There is no
  subset or benchmark-driven checkpoint selection.

The implementation pins MolecularIQ Core and the official evaluator to Git
commits, records the resolved Qwen snapshot, hashes frozen records and result
files, captures the installed environment and SLURM metadata, and uses the same
system instruction and generation protocol for all four models.

The fixed software anchors are MolecularIQ Core
`a1b89635371c3cd942e44ebeec63ec3665e7743d`, MolecularIQ Eval
`425ecaaa8faf65aa43aa60ec0f584b7b7f060063`, TRL `1.12.0`, and official task
YAML SHA-256
`49c481683afec0dcca7de490ac778983e9ebddd750e5ec809377a07e6e67007d`.

## Layout

```text
configs/                 preprocessing, three experiments, official evaluation
requirements/            connected-node inputs for the offline wheelhouse
scripts/slurm/            CPU preprocessing, GPU smoke/train/eval, CPU reporting
scripts/cloud/            portable RunPod staging and phase runners
src/miq_grpo/             builders, parser, rewards, trainer, evaluator, plots
tests/                    CPU-only integrity and adversarial reward tests
```

Generated artifacts live outside the source tree under the paths supplied via
environment variables.

## 1. Verify the source locally

Python 3.11 is required. The pure tests do not need PyTorch or RDKit:

```bash
cd /path/to/moleculariq_grpo_single_task_20260827
PYTHONPATH=src python3.11 -m pytest -q
python3.11 -m compileall -q src
for script in scripts/*.sh scripts/cloud/*.sh scripts/slurm/*.sbatch; do bash -n "$script"; done
```

## Run immediately on RunPod Secure Cloud

Use an **on-demand Secure Cloud Pod**, not Serverless or an interruptible/Spot
Pod. With only one week available, the preferred GPU is one **A100 80 GB** per
worker. An **L40S 48 GB** is the budget fallback, but only after the complete
GPU smoke gate passes. Create a **250 GB network volume** in a data center with
enough same-GPU inventory and attach it at `/workspace`. Keep the source,
sealed bundle, datasets, checkpoints, official results, and report on that
volume; the container disk is disposable.

Start with one A100 80 GB Pod using a PyTorch image that provides Python 3.11
and SSH. Use that identical image for every later worker. After connecting,
verify the interpreter, GPU, driver, and persistent mount:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h /workspace
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)'
```

If the last command fails because the template has another Python version, make
a Python 3.11 bootstrap interpreter with the template's Conda installation:

```bash
eval "$(conda shell.bash hook)"
conda create -y -n miq-bootstrap python=3.11
conda activate miq-bootstrap
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)'
```

`nvidia-smi` must identify the purchased GPU and a driver that supports CUDA
12.1 or newer. The staged environment installs the pinned CUDA 12.1 PyTorch
build and the smoke phase verifies CUDA from inside that exact environment.

### Update the clone, then stage and run the gates

Use one clean clone at a stable path on the network volume. Pull the portable
runner revision before creating the sealed bundle, and record the exact commit:

```bash
cd /workspace/practical  # replace this if the clone has a different name
git checkout main
git pull --ff-only origin main
test -f scripts/cloud/stage_bundle.sh
test -z "$(git status --porcelain)"
git rev-parse HEAD
export MIQ_PROJECT_ROOT="$(pwd -P)"
```

Do not pull, edit, or replace this clone after the bundle is staged. The bundle
seal binds the installed package and every project source byte to this absolute
path. A later code change therefore requires a new clone path, bundle path, and
run ID.

Choose one unique run ID and use the same exports in every Pod shell. Do not
change the ID, paths, or checked-out source midway through the study.

```bash
# Keep the MIQ_PROJECT_ROOT value exported above.
export MIQ_CLOUD_ROOT=/workspace/miq_cloud
export MIQ_CLOUD_RUN_ID=runpod_a100_20260922_v1
export MIQ_CLOUD_PROVIDER=runpod
cd "$MIQ_PROJECT_ROOT"
```

Stage once on the first connected Pod, then initialize and validate the new
experiment suite:

```bash
export MIQ_BOOTSTRAP_PYTHON="$(command -v python3)"
bash scripts/cloud/stage_bundle.sh
bash scripts/cloud/init_run.sh
bash scripts/cloud/preprocess.sh
bash scripts/cloud/gpu_smoke.sh
bash scripts/cloud/status.sh
```

Do **not** copy the Leonardo bundle or its virtual environment. Its manifest,
installed scripts, and virtual environment are bound to absolute Leonardo paths
and source bytes. `stage_bundle.sh` creates a fresh RunPod bundle at
`$MIQ_CLOUD_ROOT/offline_bundles/offline_bundle_v1`, seals it, and every later
phase verifies it before doing work. Do not edit or update the source after
staging. A code or configuration change requires a new bundle path and a new
run ID; never replace artifacts under an existing identity.

If using the L40S fallback and smoke fails because 48 GB is insufficient,
deploy A100 80 GB workers and initialize a new run ID. Keep the verified RunPod
bundle at its unchanged absolute path, but do not mix the failed L40S suite
with the new A100 suite or weaken the training configuration to make smoke
pass.

### Train three models in parallel

Only continue after `gpu_smoke.sh` succeeds for count, index, and constraint.
Deploy two additional one-GPU Pods of the **same GPU type**, attaching the same
network volume during deployment. In all three Pods, repeat the common exports
above and run one command each inside a persistent terminal such as `tmux`:

```bash
# Pod 1
bash scripts/cloud/train.sh count

# Pod 2
bash scripts/cloud/train.sh index

# Pod 3
bash scripts/cloud/train.sh constraint
```

The default resume mode is `auto`. After a Pod interruption, attach the same
volume to a replacement Pod of the same GPU type, restore the same exports, and
rerun the same command; it resumes from the newest valid checkpoint. Do not use
`--fresh` to bypass a partial production run. Inspect progress from any Pod:

```bash
bash scripts/cloud/status.sh
```

The phase logs are under
`$MIQ_CLOUD_ROOT/experiments/$MIQ_CLOUD_RUN_ID/logs`. Use `tail -F` on the file
reported for the active phase when closer monitoring is needed. Do not start
official evaluation until status shows all three `training_complete.json`
manifests as valid.

### Evaluate in parallel and build the report

For the shortest wall clock, use four concurrent one-GPU workers. Run the
baseline and all three trained models on the **same GPU type**—all L40S or all
A100 80 GB—and with the same run ID and network volume:

```bash
# One command per Pod
bash scripts/cloud/evaluate.sh baseline
bash scripts/cloud/evaluate.sh count_grpo
bash scripts/cloud/evaluate.sh index_grpo
bash scripts/cloud/evaluate.sh constraint_grpo
```

Each command performs one full official 5,111-document MolecularIQ evaluation.
These are final held-out measurements, not a debugging or checkpoint-selection
loop. When `status.sh` reports all four evaluations complete, run once:

```bash
bash scripts/cloud/report.sh
bash scripts/cloud/status.sh
```

The immutable suite is rooted at
`$MIQ_CLOUD_ROOT/experiments/$MIQ_CLOUD_RUN_ID`; its `report/` directory holds
the CSVs, PNG/PDF figures, manifest, and `_COMPLETE` marker described below.
Download the complete suite—including frozen data, three model runs, raw
official results, and report—using RunPod's SFTP/SCP or network-volume cloud
sync. Verify the downloaded hashes before deleting remote data.

**Billing warning:** as soon as the report and artifacts are safely downloaded,
stop or terminate every GPU Pod in the RunPod console. A separately created
network volume survives Pod termination, so storage charges continue until the
volume itself is deleted. Never delete the volume before verifying the local
copy.

### One-week execution schedule

| Deadline | Required milestone |
|---|---|
| Day 1 | Create volume/Pod, upload source, stage bundle, preprocess, pass smoke |
| Days 1–3 | Run count, index, and constraint training concurrently |
| Days 3–5 | Run baseline and three official evaluations concurrently |
| Day 6 | Build report, inspect manifests/plots, download and verify everything |
| Day 7 | Reserved for checkpoint resume, failed-Pod replacement, or transfer |

Do not serialize all seven GPU jobs on one Pod: the conservative reservation
envelope leaves essentially no recovery time inside one week. Parallel workers
write to separate task/model directories on the shared network volume.

## 2. Copy and stage on a connected Leonardo login node

First check the live CINECA module and queue policy (`module spider python/3.11`,
`sinfo`, and `saldo -b`). Leonardo's unconfigured `python3` may still be Python
3.6, so load the deep-learning profile and an available Python 3.11 module
before choosing a brand-new persistent path. The module name below is the known
Leonardo Python 3.11 module; use the exact name reported by `module spider` if it
has changed. The staging command intentionally refuses an existing destination.

```bash
module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0
python --version  # must report Python 3.11.x

export MIQ_PROJECT_ROOT="$WORK/moleculariq_grpo_single_task_20260827"
cd "$MIQ_PROJECT_ROOT"

export MIQ_BOOTSTRAP_PYTHON="$(command -v python)"
export MIQ_OFFLINE_BUNDLE_ROOT="$WORK/miq_artifacts/offline_bundle_v1"
bash scripts/stage_offline_bundle.sh
```

This connected-login-node step builds and hashes a complete Python wheelhouse,
checks out the exact MolecularIQ repositories, resolves and downloads the Qwen
snapshot, and warms two separate Hugging Face caches. It uses the CUDA 12.1
PyTorch 2.5.1 wheel, compatible with Leonardo's documented 535.54.03 driver.
It then seals the resolved package lock, every wheel, source tree, vendor Git
tree, installed project/Core/evaluator package trees, model snapshot, training
pool, and benchmark snapshot in a top-level manifest and `_READY` marker.
Submission and every later SLURM phase reject a partial or mutated bundle and
enforce all Hugging Face/Transformers/Datasets offline flags.

Do not use Booster `/tmp` for these artifacts. Put the bundle, datasets,
checkpoints, logs, benchmark outputs, and report on persistent `$WORK` or the
appropriate persistent project storage.

## 3. Freeze the experiment before looking at benchmark results

Review these files before the first reportable run:

- `configs/preprocessing.yaml`: 20,000 examples per family and the allowed
  official property sets.
- `configs/experiments/*.yaml`: three seeds, full fine-tuning, one epoch,
  `num_generations=8`, fixed reward weights, FP32 trainable master parameters,
  and BF16 mixed-precision compute. Keeping the master parameters in FP32 is
  deliberate at the configured `1e-6` learning rate.
- `configs/evaluation.yaml`: one baseline plus three trained checkpoints, all
  evaluated with the same official task.

If you change an experiment choice, create a new versioned dataset/run/result
root. Do not use official benchmark outcomes to decide any of these values.

## 4. Submit the complete pipeline

Use separate Booster and DCGP account names if your allocation requires the
usual DCGP `_0` suffix. Every destination below should be new.

```bash
export MIQ_DCGP_ACCOUNT="YOUR_DCGP_ACCOUNT"
export MIQ_BOOSTER_ACCOUNT="YOUR_BOOSTER_ACCOUNT"
export MIQ_DATA_ROOT="$WORK/miq_artifacts/frozen_datasets_v1"
export MIQ_RUN_ROOT="$WORK/miq_artifacts/runs_v1"
export MIQ_EVAL_RESULTS_ROOT="$WORK/miq_artifacts/official_results_v1"
export MIQ_REPORT_ROOT="$WORK/miq_artifacts/report_v1"

bash scripts/submit_pipeline.sh
```

The dependency chain is:

```text
CPU frozen-data build -> 1-GPU smoke/reload gate -> 3-model training array
-> 4-model whole-benchmark evaluation array -> report tables and plots
```

The smoke job exercises count, index, and constrained generation separately. It
validates package/API versions, CUDA visibility, dataset hashes, reward oracle
cases, a forced stop/checkpoint/resume sequence, two real GRPO optimizer steps,
fresh-process reload, changed parameters versus the baseline, and nonzero
within-group reward variance for every task. It then switches to the isolated
evaluation cache and checks the pinned task, all 5,111 cached document hashes,
and baseline loading without generating or scoring benchmark answers. Monitor
with `squeue` and inspect final states with `sacct`; the submit script prints all
five job IDs.

## Preemption and explicit resume

The training array requests a `USR1` warning ten minutes before walltime. The
trainer saves at the next step boundary, stops, and the job exits `99` instead
of pretending to be complete. Resume only the affected array element from its
exact checkpoint. For example, for count (array index 0):

```bash
export MIQ_DATA_BUNDLE="$(<"$MIQ_RUN_ROOT/pipeline_state/dataset_bundle_path.txt")"
export MIQ_MODEL_PATH="$(python -c 'import json,os; print(json.load(open(os.environ["MIQ_OFFLINE_BUNDLE_ROOT"] + "/training_assets/assets_manifest.json"))["model_path"])')"
export MIQ_MODEL_REVISION="$(python -c 'import json,os; print(json.load(open(os.environ["MIQ_OFFLINE_BUNDLE_ROOT"] + "/training_assets/assets_manifest.json"))["resolved_model_revision"])')"
export MIQ_PIPELINE_STATE="$MIQ_RUN_ROOT/pipeline_state"
export MIQ_RESUME_COUNT_CHECKPOINT="$MIQ_RUN_ROOT/qwen25_05b_miq_count_grpo_v1/checkpoint-N"

cd "$MIQ_PROJECT_ROOT"
sbatch --account "$MIQ_BOOSTER_ACCOUNT" --array=0 scripts/slurm/train_array.sbatch
```

Use `MIQ_RESUME_INDEX_CHECKPOINT` with array index 1 or
`MIQ_RESUME_CONSTRAINT_CHECKPOINT` with index 2. A resume is accepted only when
the frozen configuration and run directory match. Wait until all three
production run directories contain a valid `training_complete.json`.

The evaluation and report jobs submitted by the original pipeline depend on the
original training-array job. Once an array element exits `99`, that old
dependency cannot be repaired by a separately submitted resume. Read the old
job IDs, confirm that the evaluation and report jobs are only stale downstream
jobs, and cancel them:

```bash
grep -E '^(eval|report)=' "$MIQ_PIPELINE_STATE/submitted_jobs.txt"
squeue -j OLD_EVAL_JOB_ID,OLD_REPORT_JOB_ID
scancel OLD_EVAL_JOB_ID OLD_REPORT_JOB_ID
```

Do not cancel a running official evaluation, and do not use the recovery helper
if any model has already completed official evaluation. After every resumed
training job has completed, keep the original environment variables and submit
a fresh finalization chain:

```bash
cd "$MIQ_PROJECT_ROOT"
bash scripts/submit_evaluation_and_report.sh
```

The helper refuses unless all three non-smoke completion manifests match their
expected experiment/task identities, base-model snapshot, and frozen dataset
artifacts. It also requires clean, unused evaluation/report roots, complete
offline assets, the pinned clean evaluator checkout, and no active old
downstream jobs. It submits a new four-model evaluation array and a report job
with `afterok` on that array. The original job record is left untouched; the new
job IDs and the superseded IDs are written once under
`$MIQ_PIPELINE_STATE/evaluation_and_report_submission/`.

## Outputs for the report

After the final dependency succeeds, `MIQ_REPORT_ROOT` contains:

- `official_overall_metrics.csv` (`pass_at_1`, `pass_at_3`, `avg_accuracy`),
- `official_task_family_metrics.csv`,
- `official_supercategory_metrics.csv`,
- `official_paired_deltas_with_ci.csv` (deterministic 2,000-replicate paired
  document bootstrap, including 95% confidence intervals),
- `training_metrics.csv`,
- PNG and PDF versions of `overall_official_metrics`,
  `accuracy_by_task_family`, `paired_overall_delta_vs_baseline`,
  `paired_task_family_deltas_vs_baseline`, and `training_reward_curves`,
- `report_manifest.json` with hashes for every input run, raw result, table, and
  figure, plus a final `_COMPLETE` marker.

Reporting builds in a sibling temporary directory and atomically publishes
only after all four model/protocol/checkpoint bindings, all 5,111 paired sample
rows, CSVs, confidence intervals, and plots succeed. A failed report leaves no
partially published `MIQ_REPORT_ROOT`.

Treat the whole official benchmark as a one-time final measurement. If a run
fails operationally, retain its failure artifact and rerun for that documented
reason—never choose among successful benchmark runs by score.
